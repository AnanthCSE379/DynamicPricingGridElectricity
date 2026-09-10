"""
run_multiweek_seasonal_benchmark.py
Comprehensive Multi-Week & 4-Season Grid Pricing Benchmark:
Evaluates 5 Pricing Regimes across 4 Distinct Seasonal Weeks and a Continuous 2-Week Winter Peak:
1. Winter Peak Week: Jan 13, 2014 - Jan 19, 2014 (336 half-hours)
2. Spring Transition Week: Apr 15, 2013 - Apr 21, 2013 (336 half-hours)
3. Summer Low-Demand Week: Jul 15, 2013 - Jul 21, 2013 (336 half-hours)
4. Autumn Ramp Week: Oct 14, 2013 - Oct 20, 2013 (336 half-hours)
5. Continuous 2-Week Winter Stress Test: Jan 13, 2014 - Jan 26, 2014 (672 half-hours)

Regimes Compared:
1. Static Flat Tariff as per LCL (14.23 p/kWh)
2. GRU Forecast + Static Peak Pricing (28.00 p/kWh threshold)
3. Double GRU + Math Continuous Dynamic Pricing
4. GRU Forecast + Fine-Tuned MLX Agent Hybrid
5. Pure Agentic (Agentic AI Forecast + Agentic AI Pricing)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from src.pure_dataset            import add_pure_features, apply_pure_scaler, LOOKBACK, TARGET_COL
from src.dataset                 import _add_cyclical_features, apply_scaler
from src.dynamic_pricing_controller import DynamicPricingController
from src.mlx_agent               import HybridMLXController, PureMLXController
from src.grid_economics          import compute_halfhour_economics, wholesale_price_at, FEEDER_THRESHOLD
from train_pure_benchmark        import PureGRU
from src.models                  import GRUModel as ConsumerTwin

N_HOUSEHOLDS = 1000
LCL_FLAT_TARIFF = 14.23
STATIC_PEAK_TARIFF = 28.00
HISTORY_LEN = 12

OUT_DIR = os.path.join(ROOT_DIR, "plots/6way_benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

SEASONAL_WINDOWS = {
    "Winter": {
        "start": "2014-01-13 00:00:00",
        "days": 7,
        "desc": "Winter Peak Week (Cold Snap, High Heating & Peak Spot Prices)"
    },
    "Spring": {
        "start": "2013-04-15 00:00:00",
        "days": 7,
        "desc": "Spring Transition Week (Moderate Demand, Solar Emergence)"
    },
    "Summer": {
        "start": "2013-07-15 00:00:00",
        "days": 7,
        "desc": "Summer Low-Demand Week (High Solar Suppression, Basal Load)"
    },
    "Autumn": {
        "start": "2013-10-14 00:00:00",
        "days": 7,
        "desc": "Autumn Ramp Week (Rising Evening Lighting & Heating)"
    },
    "Winter_2Weeks": {
        "start": "2014-01-13 00:00:00",
        "days": 14,
        "desc": "Continuous 2-Week Winter Peak Test (14 Days / 672 Half-Hours)"
    }
}

def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_models(device):
    print("[Models] Loading models and scalers...", flush=True)
    pure_model = PureGRU()
    pure_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "checkpoints/BestPureModel.pt"), map_location=device))
    pure_model.to(device).eval()
    pure_scaler = joblib.load(os.path.join(ROOT_DIR, "checkpoints/pure_scaler.pkl"))

    consumer = ConsumerTwin()
    consumer.load_state_dict(torch.load(os.path.join(ROOT_DIR, "checkpoints/GRU.pt"), map_location=device))
    consumer.to(device).eval()
    p2_scaler = joblib.load(os.path.join(ROOT_DIR, "checkpoints/scaler.pkl"))

    return pure_model, pure_scaler, consumer, p2_scaler

def extract_window_features(df_full, start_str, days, pure_scaler, p2_scaler):
    df_pure = add_pure_features(df_full).with_columns(pl.col(TARGET_COL).forward_fill())
    df_p2   = _add_cyclical_features(df_full).with_columns(pl.col(TARGET_COL).forward_fill())
    
    start_dt = pd.to_datetime(start_str)
    history_start = start_dt - pd.Timedelta(hours=24) # 48 half-hours history
    end_dt = start_dt + pd.Timedelta(days=days)
    
    pure_sub = df_pure.filter((pl.col("DateTime") >= pl.lit(str(history_start)).str.to_datetime()) & (pl.col("DateTime") < pl.lit(str(end_dt)).str.to_datetime()))
    p2_sub   = df_p2.filter((pl.col("DateTime") >= pl.lit(str(history_start)).str.to_datetime()) & (pl.col("DateTime") < pl.lit(str(end_dt)).str.to_datetime()))
    
    arr_pure = apply_pure_scaler(pure_sub, pure_scaler)
    arr_p2   = apply_scaler(p2_sub, p2_scaler)
    ts_eval  = pure_sub.select("DateTime").to_series().to_list()[LOOKBACK:]
    
    return arr_pure, arr_p2, ts_eval

def gru_predict_one(model, scaler, arr, idx, device):
    window = torch.tensor(arr[idx:idx+LOOKBACK], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(window).cpu().item()
    return float(scaler.inverse_transform([[pred_scaled]])[0][0])

def consumer_predict_one(model, scaler, arr_p2, idx, tariff, device):
    window = arr_p2[idx:idx+LOOKBACK].copy()
    window[-1, -1] = tariff
    t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(t).cpu().item()
    return float(np.clip(scaler.inverse_transform([[pred_scaled]])[0][0], 0.01, None))

def simulate_campaign(season_name, config, df_full, pure_model, pure_scaler, consumer, p2_scaler, device, agent_hybrid, agent_pure):
    print(f"\n{'='*75}\n  Simulating Campaign: {season_name} ({config['desc']})\n{'='*75}", flush=True)
    arr_pure, arr_p2, ts_eval = extract_window_features(df_full, config["start"], config["days"], pure_scaler, p2_scaler)
    n = len(ts_eval)
    
    ctrl_math = DynamicPricingController()

    regimes = {
        "regime1_static_lcl": {
            "label": "1. Static Flat (LCL 14.23p)",
            "tariffs": [], "demands": [], "forecasts": [], "wholesale": [], "profits": [],
            "revenues": [], "wholesale_costs": [], "congestion_costs": []
        },
        "regime2_gru_static_peak": {
            "label": "2. GRU + Static Peak (28p)",
            "tariffs": [], "demands": [], "forecasts": [], "wholesale": [], "profits": [],
            "revenues": [], "wholesale_costs": [], "congestion_costs": []
        },
        "regime3_gru_dynamic_math": {
            "label": "3. Double GRU + Math Dynamic",
            "tariffs": [], "demands": [], "forecasts": [], "wholesale": [], "profits": [],
            "revenues": [], "wholesale_costs": [], "congestion_costs": []
        },
        "regime4_gru_agent_hybrid": {
            "label": "4. GRU + Agentic AI (Preemptive)",
            "tariffs": [], "demands": [], "forecasts": [], "wholesale": [], "profits": [],
            "revenues": [], "wholesale_costs": [], "congestion_costs": []
        },
        "regime5_pure_agent": {
            "label": "5. Pure Agentic AI",
            "tariffs": [], "demands": [], "forecasts": [], "wholesale": [], "profits": [],
            "revenues": [], "wholesale_costs": [], "congestion_costs": []
        },
        "regime6_hybrid_advance": {
            "label": "6. Hybrid Agent (Advance Broadcast)",
            "tariffs": [], "demands": [], "forecasts": [], "wholesale": [], "profits": [],
            "revenues": [], "wholesale_costs": [], "congestion_costs": []
        }
    }

    next_t6 = LCL_FLAT_TARIFF

    times = []

    for i in range(n):
        dt = datetime.fromisoformat(str(ts_eval[i]))
        time_str = dt.strftime("%A %H:%M")
        wh = wholesale_price_at(dt)
        times.append(str(dt))

        # 1. PureGRU load forecast
        gru_fc = gru_predict_one(pure_model, pure_scaler, arr_pure, i, device)

        # 2. Historical loads for Pure MLX Agent
        start = max(0, i + LOOKBACK - HISTORY_LEN)
        end   = i + LOOKBACK
        hist_scaled = arr_pure[start:end, 0]
        history = pure_scaler.inverse_transform(hist_scaled.reshape(-1, 1)).flatten().tolist()
        if len(history) < HISTORY_LEN:
            history = ([history[0]] * (HISTORY_LEN - len(history))) + history

        # --- Regime 1: Static Flat ---
        t1 = LCL_FLAT_TARIFF
        d1 = consumer_predict_one(consumer, p2_scaler, arr_p2, i, t1, device)
        e1 = compute_halfhour_economics(d1, t1, wh, N_HOUSEHOLDS)
        regimes["regime1_static_lcl"]["tariffs"].append(t1)
        regimes["regime1_static_lcl"]["demands"].append(d1)
        regimes["regime1_static_lcl"]["forecasts"].append(d1)
        regimes["regime1_static_lcl"]["wholesale"].append(wh)
        regimes["regime1_static_lcl"]["profits"].append(e1["net_profit_gbp"])
        regimes["regime1_static_lcl"]["revenues"].append(e1["revenue_gbp"])
        regimes["regime1_static_lcl"]["wholesale_costs"].append(e1["wholesale_cost_gbp"])
        regimes["regime1_static_lcl"]["congestion_costs"].append(e1["congestion_cost_gbp"])

        # --- Regime 2: GRU + Static Peak ---
        t2 = STATIC_PEAK_TARIFF if gru_fc > FEEDER_THRESHOLD else LCL_FLAT_TARIFF
        d2 = consumer_predict_one(consumer, p2_scaler, arr_p2, i, t2, device)
        e2 = compute_halfhour_economics(d2, t2, wh, N_HOUSEHOLDS)
        regimes["regime2_gru_static_peak"]["tariffs"].append(t2)
        regimes["regime2_gru_static_peak"]["demands"].append(d2)
        regimes["regime2_gru_static_peak"]["forecasts"].append(gru_fc)
        regimes["regime2_gru_static_peak"]["wholesale"].append(wh)
        regimes["regime2_gru_static_peak"]["profits"].append(e2["net_profit_gbp"])
        regimes["regime2_gru_static_peak"]["revenues"].append(e2["revenue_gbp"])
        regimes["regime2_gru_static_peak"]["wholesale_costs"].append(e2["wholesale_cost_gbp"])
        regimes["regime2_gru_static_peak"]["congestion_costs"].append(e2["congestion_cost_gbp"])

        # --- Regime 3: Double GRU + Math ---
        t3 = ctrl_math.compute_tariff(gru_fc)
        d3 = consumer_predict_one(consumer, p2_scaler, arr_p2, i, t3, device)
        e3 = compute_halfhour_economics(d3, t3, wh, N_HOUSEHOLDS)
        regimes["regime3_gru_dynamic_math"]["tariffs"].append(t3)
        regimes["regime3_gru_dynamic_math"]["demands"].append(d3)
        regimes["regime3_gru_dynamic_math"]["forecasts"].append(gru_fc)
        regimes["regime3_gru_dynamic_math"]["wholesale"].append(wh)
        regimes["regime3_gru_dynamic_math"]["profits"].append(e3["net_profit_gbp"])
        regimes["regime3_gru_dynamic_math"]["revenues"].append(e3["revenue_gbp"])
        regimes["regime3_gru_dynamic_math"]["wholesale_costs"].append(e3["wholesale_cost_gbp"])
        regimes["regime3_gru_dynamic_math"]["congestion_costs"].append(e3["congestion_cost_gbp"])

        # --- Regime 4: Hybrid Agentic AI ---
        t4, _ = agent_hybrid.compute_tariff(time_str, wh, gru_fc, history=history)
        d4 = consumer_predict_one(consumer, p2_scaler, arr_p2, i, t4, device)
        e4 = compute_halfhour_economics(d4, t4, wh, N_HOUSEHOLDS)
        regimes["regime4_gru_agent_hybrid"]["tariffs"].append(t4)
        regimes["regime4_gru_agent_hybrid"]["demands"].append(d4)
        regimes["regime4_gru_agent_hybrid"]["forecasts"].append(gru_fc)
        regimes["regime4_gru_agent_hybrid"]["wholesale"].append(wh)
        regimes["regime4_gru_agent_hybrid"]["profits"].append(e4["net_profit_gbp"])
        regimes["regime4_gru_agent_hybrid"]["revenues"].append(e4["revenue_gbp"])
        regimes["regime4_gru_agent_hybrid"]["wholesale_costs"].append(e4["wholesale_cost_gbp"])
        regimes["regime4_gru_agent_hybrid"]["congestion_costs"].append(e4["congestion_cost_gbp"])

        # --- Regime 5: Pure Agentic AI ---
        p5, t5, _ = agent_pure.forecast_and_price(time_str, wh, history)
        d5 = consumer_predict_one(consumer, p2_scaler, arr_p2, i, t5, device)
        e5 = compute_halfhour_economics(d5, t5, wh, N_HOUSEHOLDS)
        regimes["regime5_pure_agent"]["tariffs"].append(t5)
        regimes["regime5_pure_agent"]["demands"].append(d5)
        regimes["regime5_pure_agent"]["forecasts"].append(p5)
        regimes["regime5_pure_agent"]["wholesale"].append(wh)
        regimes["regime5_pure_agent"]["profits"].append(e5["net_profit_gbp"])
        regimes["regime5_pure_agent"]["revenues"].append(e5["revenue_gbp"])
        regimes["regime5_pure_agent"]["wholesale_costs"].append(e5["wholesale_cost_gbp"])
        regimes["regime5_pure_agent"]["congestion_costs"].append(e5["congestion_cost_gbp"])

        # --- Regime 6: Hybrid Agentic AI (Advance Broadcast) ---
        d6 = consumer_predict_one(consumer, p2_scaler, arr_p2, i, next_t6, device)
        e6 = compute_halfhour_economics(d6, next_t6, wh, N_HOUSEHOLDS)
        regimes["regime6_hybrid_advance"]["tariffs"].append(next_t6)
        regimes["regime6_hybrid_advance"]["demands"].append(d6)
        regimes["regime6_hybrid_advance"]["forecasts"].append(gru_fc)
        regimes["regime6_hybrid_advance"]["wholesale"].append(wh)
        regimes["regime6_hybrid_advance"]["profits"].append(e6["net_profit_gbp"])
        regimes["regime6_hybrid_advance"]["revenues"].append(e6["revenue_gbp"])
        regimes["regime6_hybrid_advance"]["wholesale_costs"].append(e6["wholesale_cost_gbp"])
        regimes["regime6_hybrid_advance"]["congestion_costs"].append(e6["congestion_cost_gbp"])
        # The Hybrid Agent already calculated the tariff (t4) based on the forecast. 
        # We simply store it to be applied 1 step later for the Advance Broadcast regime.
        next_t6 = t4

        if (i + 1) % 48 == 0 or i == n - 1:
            day_num = (i + 1) // 48
            print(f"  Day {day_num:2d} ({i+1:3d}/{n} HH) | Net Profit so far: Flat=£{sum(regimes['regime1_static_lcl']['profits']):,.0f}, Math=£{sum(regimes['regime3_gru_dynamic_math']['profits']):,.0f}, Hyb=£{sum(regimes['regime4_gru_agent_hybrid']['profits']):,.0f}", flush=True)

    # Summarize this campaign
    campaign_summary = {}
    base_demands = np.array(regimes["regime1_static_lcl"]["demands"])
    peak_mask   = base_demands > FEEDER_THRESHOLD
    valley_mask = base_demands < 0.18

    for k, data in regimes.items():
        tariffs  = np.array(data["tariffs"])
        demands  = np.array(data["demands"])
        profits  = np.array(data["profits"])
        revenues = np.array(data["revenues"])
        w_costs  = np.array(data["wholesale_costs"])
        c_costs  = np.array(data["congestion_costs"])

        total_profit = float(profits.sum())
        total_rev    = float(revenues.sum())
        total_wcost  = float(w_costs.sum())
        total_ccost  = float(c_costs.sum())
        margin_pct   = (total_profit / total_rev * 100) if total_rev > 0 else 0.0
        avg_rate     = float((tariffs * demands).sum() / demands.sum())
        price_vol    = float(np.std(tariffs))
        max_tariff   = float(np.max(tariffs))
        stress_hours = float(np.sum(demands > FEEDER_THRESHOLD)) / 2.0

        # Behavioral metrics
        if np.sum(peak_mask) > 0:
            peak_curtail_pct = float(((base_demands[peak_mask].sum() - demands[peak_mask].sum()) / base_demands[peak_mask].sum()) * 100.0)
        else:
            peak_curtail_pct = 0.0

        if np.sum(valley_mask) > 0:
            valley_boost_pct = float(((demands[valley_mask].sum() - base_demands[valley_mask].sum()) / base_demands[valley_mask].sum()) * 100.0)
        else:
            valley_boost_pct = 0.0

        shock_hours = float(np.sum(tariffs > 25.0)) / 2.0

        campaign_summary[k] = {
            "label": data["label"],
            "total_net_profit_gbp": round(total_profit, 2),
            "gross_revenue_gbp": round(total_rev, 2),
            "wholesale_cost_gbp": round(total_wcost, 2),
            "congestion_cost_gbp": round(total_ccost, 2),
            "profit_margin_pct": round(margin_pct, 2),
            "customer_total_bill_gbp": round(total_rev, 2),
            "customer_effective_rate_p_kwh": round(avg_rate, 2),
            "max_tariff_p": round(max_tariff, 2),
            "tariff_volatility_std": round(price_vol, 2),
            "peak_curtailment_pct": round(peak_curtail_pct, 2),
            "valley_consumption_boost_pct": round(valley_boost_pct, 2),
            "price_shock_hours": round(shock_hours, 1),
            "grid_stress_hours": round(stress_hours, 1),
            "tariffs": data["tariffs"],
            "demands": data["demands"],
            "forecasts": data["forecasts"],
            "profits": data["profits"],
            "times": times
        }

    return campaign_summary

def main():
    device = get_device()
    pure_model, pure_scaler, consumer, p2_scaler = load_models(device)
    
    print("[Data] Loading full feeder aggregate parquet...", flush=True)
    parquet_path = os.path.join(ROOT_DIR, "data/feeder_aggregate_halfhourly.parquet")
    df_full = pl.read_parquet(parquet_path).sort("DateTime")
    
    agent_hybrid = HybridMLXController()
    agent_pure   = PureMLXController()

    all_campaigns = {}

    for season_name, config in SEASONAL_WINDOWS.items():
        summary = simulate_campaign(
            season_name, config, df_full, pure_model, pure_scaler, consumer, p2_scaler, device, agent_hybrid, agent_pure
        )
        all_campaigns[season_name] = summary

    out_file = os.path.join(OUT_DIR, "seasonal_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(all_campaigns, f, indent=2)

    print("\n" + "=" * 110)
    print("  MULTI-WEEK & 4-SEASON BENCHMARK MASTER SCORECARD")
    print("=" * 110)
    for season_name in ["Winter", "Spring", "Summer", "Autumn", "Winter_2Weeks"]:
        print(f"\n--- Campaign: {season_name} ({SEASONAL_WINDOWS[season_name]['desc']}) ---")
        print(f"  {'Model / Regime':<36}{'Net Profit':<14}{'Cust Rate':<14}{'Curtail %':<12}{'Stress Hrs'}")
        print("  " + "-" * 85)
        for k, s in all_campaigns[season_name].items():
            print(f"  {s['label']:<36}£{s['total_net_profit_gbp']:>9,.2f}    {s['customer_effective_rate_p_kwh']:>5.2f} p/kWh   {s['peak_curtailment_pct']:>6.1f}%     {s['grid_stress_hours']:>5.1f}h")
    print("\n" + "=" * 110)
    print(f"All multi-week and seasonal results saved to {out_file}!")

if __name__ == "__main__":
    main()
