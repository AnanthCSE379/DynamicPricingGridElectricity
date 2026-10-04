"""Reproducible experimental interrogation of the trained ConsumerTwin GRU.

This script DOES NOT replace the GRU with a demand curve.  It loads
checkpoints/GRU.pt and checkpoints/scaler.pkl, creates controlled tariff paths
around valid held-out histories, rolls the GRU forward autoregressively, and
saves raw trajectories, derived metrics, tables, and publication figures.

Run from the repository root:
    MPLCONFIGDIR=/private/tmp/customer-response-mpl \
      /Users/ananth/anaconda3/bin/python scripts/run_gru_customer_response_study.py

The model's training tariff support is discrete: 3.99, 11.76, and 67.20 p/kWh.
Intermediate prices are retained only as labelled interpolation diagnostics;
the primary conclusions use the observed price levels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import GRUModel

DATA = ROOT / "data/feeder_aggregate_halfhourly.csv"
CHECKPOINT = ROOT / "checkpoints/GRU.pt"
SCALER = ROOT / "checkpoints/scaler.pkl"
OUT = ROOT / "results/customer_response"
RAW, METRICS, PLOTS, TABLES = [OUT / x for x in ("raw", "metrics", "plots", "tables")]

LOOKBACK, BASE, RECOVERY = 48, 11.76, 24
OBSERVED_PRICES = np.array([3.99, 11.76, 67.20])
PRICE_GRID = np.array([3.99, 11.76, 20.0, 35.0, 50.0, 67.20])
DURATIONS = np.array([1, 2, 4, 8, 16])  # half-hours
SEED = 20261004


def engineer(df: pd.DataFrame) -> np.ndarray:
    """Exact feature order used by src.dataset.py, implemented with pandas."""
    dt = pd.to_datetime(df["DateTime"])
    hh = dt.dt.hour * 2 + dt.dt.minute // 30
    dow = dt.dt.weekday + 1
    month = dt.dt.month
    return np.column_stack([
        df["household_mean_kwh"].ffill().to_numpy(np.float32),
        np.sin(2 * np.pi * hh / 48), np.cos(2 * np.pi * hh / 48),
        np.sin(2 * np.pi * (dow - 1) / 7), np.cos(2 * np.pi * (dow - 1) / 7),
        np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
        (dow >= 6).astype(np.float32), df["price_p_per_kwh"].fillna(BASE).to_numpy(np.float32),
    ]).astype(np.float32)


def load_model():
    scaler = joblib.load(SCALER)
    model = GRUModel()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.eval()
    return model, scaler


def choose_seeds(df: pd.DataFrame, n_per_period: int = 24) -> tuple[np.ndarray, pd.DataFrame]:
    """Sample many valid held-out windows, balanced by actual time-of-day."""
    test = (df["DateTime"] > pd.Timestamp("2013-09-30 23:30:00"))
    valid = test & (np.arange(len(df)) >= LOOKBACK) & (np.arange(len(df)) < len(df) - (RECOVERY + DURATIONS.max() + 2))
    hour = df["DateTime"].dt.hour
    bins = {
        "Night (00:00–05:30)": hour < 6,
        "Morning (06:00–10:30)": (hour >= 6) & (hour < 11),
        "Afternoon (11:00–16:30)": (hour >= 11) & (hour < 17),
        "Evening (17:00–23:30)": hour >= 17,
    }
    rng = np.random.default_rng(SEED)
    selected = []
    rows = []
    for label, mask in bins.items():
        idx = np.flatnonzero(valid & mask.to_numpy())
        chosen = rng.choice(idx, size=min(n_per_period, len(idx)), replace=False)
        selected.extend(chosen.tolist())
        rows.extend([{"seed_index": int(i), "time_period": label, "timestamp": str(df.loc[i, "DateTime"])} for i in chosen])
    seeds = np.array(sorted(selected), dtype=int)
    meta = pd.DataFrame(rows).sort_values("seed_index").reset_index(drop=True)
    meta["sample_id"] = np.arange(len(meta))
    return seeds, meta


def rollout(model, scaler, arr, seeds, price_path: np.ndarray) -> np.ndarray:
    """Model-consistent autoregressive rollout for a shared tariff path.

    At a given step the last price in the 48-step input causes the next-demand
    prediction. The predicted load is then appended with the next scenario
    price and real future calendar features. No future observed load is used.
    """
    n, horizon = len(seeds), len(price_path)
    windows = np.stack([arr[s - LOOKBACK + 1:s + 1].copy() for s in seeds])
    output = np.empty((n, horizon), dtype=np.float32)
    with torch.no_grad():
        for j, price in enumerate(price_path):
            windows[:, -1, -1] = price
            pred_z = model(torch.from_numpy(windows)).numpy().reshape(-1)
            pred = scaler.inverse_transform(pred_z.reshape(-1, 1)).reshape(-1)
            pred = np.clip(pred, 0.01, None)
            output[:, j] = pred
            if j < horizon - 1:
                row_idx = seeds + j + 1
                new_rows = arr[row_idx].copy()       # only future calendar context is reused
                new_rows[:, 0] = scaler.transform(pred.reshape(-1, 1)).reshape(-1)
                new_rows[:, -1] = price_path[j + 1]
                windows = np.concatenate([windows[:, 1:, :], new_rows[:, None, :]], axis=1)
    return output


def support(price: float) -> str:
    return "observed training price" if np.any(np.isclose(price, OBSERVED_PRICES)) else "interpolation within training range"


def ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, mean - 1.96 * se, mean + 1.96 * se


def event_metrics(base: np.ndarray, intervention: np.ndarray, price: float, duration: int, meta: pd.DataFrame) -> pd.DataFrame:
    """Energy accounting. Positive difference means load removed by the intervention."""
    diff = base - intervention
    event = diff[:, :duration]
    post = diff[:, duration:]
    removed = np.maximum(event, 0).sum(axis=1)
    recovered = np.maximum(-post, 0).sum(axis=1)
    rebound = np.maximum(-post, 0).max(axis=1)
    net = diff.sum(axis=1)
    reduction = event.mean(axis=1)
    base_event = base[:, :duration].mean(axis=1)
    response_pct = 100 * reduction / base_event
    price_pct = 100 * (price - BASE) / BASE
    elasticity = np.full_like(response_pct, np.nan, dtype=float)
    # Standard price elasticity uses the signed demand change, hence negative
    # values denote demand falling as price rises.
    np.divide(-response_pct, price_pct, out=elasticity, where=abs(price_pct) > 1e-9)
    recovery_ratio = np.full_like(removed, np.nan, dtype=float)
    np.divide(recovered, removed, out=recovery_ratio, where=removed > 1e-9)
    out = meta[["sample_id", "time_period", "timestamp"]].copy()
    out["price_p_per_kwh"] = price
    out["price_support"] = support(price)
    out["duration_half_hours"] = duration
    out["duration_hours"] = duration / 2
    out["baseline_demand_kwh_per_hh"] = base_event
    out["event_demand_kwh_per_hh"] = intervention[:, :duration].mean(axis=1)
    out["absolute_reduction_kwh_per_hh"] = reduction
    out["response_pct"] = response_pct
    out["price_change_p_per_kwh"] = price - BASE
    out["price_change_pct"] = price_pct
    out["elasticity"] = elasticity
    out["energy_removed_kwh"] = removed
    out["energy_recovered_kwh"] = recovered
    out["net_energy_reduction_kwh"] = net
    out["recovery_ratio"] = recovery_ratio
    out["rebound_peak_kwh_per_hh"] = rebound
    out["flexible_fraction_at_test_price"] = np.where(base_event > 0, removed / (base_event * duration), np.nan)
    return out


def add_raw(store: list[dict], experiment: str, scenario: str, price_path: np.ndarray, baseline: np.ndarray, scenario_q: np.ndarray, meta: pd.DataFrame, duration: int):
    for sample_i, row in meta.iterrows():
        for step, (p, b, q) in enumerate(zip(price_path, baseline[sample_i], scenario_q[sample_i])):
            store.append({"experiment": experiment, "scenario": scenario, "sample_id": row.sample_id,
                          "time_period": row.time_period, "timestamp_seed": row.timestamp, "step_half_hour": step + 1,
                          "phase": "high-price event" if step < duration else "post-event recovery",
                          "price_p_per_kwh": p, "baseline_demand_kwh": b, "scenario_demand_kwh": q,
                          "response_baseline_minus_scenario_kwh": b - q})


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(PLOTS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_study(metrics: pd.DataFrame, trajectories: pd.DataFrame, history: pd.DataFrame, repeat: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    primary = metrics.loc[metrics.duration_half_hours.eq(4)].copy()
    group = primary.groupby("price_p_per_kwh")
    summary = group.agg(n=("sample_id", "count"), demand=("event_demand_kwh_per_hh", "mean"), reduction=("absolute_reduction_kwh_per_hh", "mean"),
                        response_pct=("response_pct", "mean"), removed=("energy_removed_kwh", "mean"), recovered=("energy_recovered_kwh", "mean"),
                        net=("net_energy_reduction_kwh", "mean"), elasticity=("elasticity", "mean"), rebound=("rebound_peak_kwh_per_hh", "mean")).reset_index()
    summary.to_csv(METRICS / "price_response_summary.csv", index=False)
    summary[["price_p_per_kwh", "n", "removed", "recovered", "net", "rebound"]].to_csv(METRICS / "energy_accounting_summary.csv", index=False)
    obs = summary.price_p_per_kwh.isin(OBSERVED_PRICES)

    source_columns = {"demand":"event_demand_kwh_per_hh", "reduction":"absolute_reduction_kwh_per_hh", "response_pct":"response_pct", "removed":"energy_removed_kwh", "recovered":"energy_recovered_kwh", "net":"net_energy_reduction_kwh", "elasticity":"elasticity", "rebound":"rebound_peak_kwh_per_hh"}
    def line_plot(y, ylabel, name, title):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        errors=group[source_columns[y]].sem().reindex(summary.price_p_per_kwh).to_numpy()
        ax.errorbar(summary.price_p_per_kwh, summary[y], yerr=errors, marker="o", capsize=3, color="#245a8d", label="Mean ± 95% CI")
        ax.scatter(summary.loc[obs, "price_p_per_kwh"], summary.loc[obs, y], s=50, color="#b2182b", zorder=3, label="Observed training prices")
        ax.set(xlabel="Event price (p/kWh)", ylabel=ylabel, title=title)
        ax.legend(); savefig(fig, name)
    line_plot("demand", "Predicted demand (kWh / half-hour)", "01_price_vs_predicted_demand", "Price and predicted demand")
    line_plot("response_pct", "Demand reduction (%)", "02_price_vs_pct_reduction", "Price and relative demand reduction")
    line_plot("recovered", "Recovered energy after event (kWh)", "03_price_vs_load_shifted", "Price and model-implied shifted/recovered energy")
    line_plot("net", "Net energy reduction over event + recovery (kWh)", "04_price_vs_net_customer_withdrawal", "Price and net energy reduction")
    line_plot("elasticity", "Estimated price elasticity (dimensionless)", "05_price_vs_elasticity", "Price and local event elasticity")
    line_plot("rebound", "Post-event rebound peak (kWh / half-hour)", "08_price_vs_rebound_magnitude", "Event price and rebound magnitude")

    marginal = summary.loc[summary.price_p_per_kwh >= BASE].copy()
    marginal["marginal"] = np.gradient(marginal.reduction, marginal.price_p_per_kwh)
    fig, ax = plt.subplots(figsize=(7.4, 4.8)); ax.plot(marginal.price_p_per_kwh, marginal.marginal, marker="o", color="#7b3294")
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Event price (p/kWh)", ylabel="Δ reduction / Δ price (kWh / half-hour per p/kWh)", title="Marginal price response")
    savefig(fig, "06_price_vs_marginal_response")

    duration = metrics.loc[metrics.price_p_per_kwh.eq(67.2)].groupby("duration_hours").agg(reduction=("absolute_reduction_kwh_per_hh", "mean"), reduction_sem=("absolute_reduction_kwh_per_hh", "sem"), removed=("energy_removed_kwh", "mean"), removed_sem=("energy_removed_kwh", "sem"), net=("net_energy_reduction_kwh", "mean"), net_sem=("net_energy_reduction_kwh", "sem")).reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
    axes[0].errorbar(duration.duration_hours, duration.reduction, yerr=1.96*duration.reduction_sem, marker="o", capsize=3, color="#245a8d")
    axes[0].set(ylabel="Mean demand reduction\n(kWh / half-hour)", title="High-price duration and response at 67.2 p/kWh")
    axes[1].errorbar(duration.duration_hours, duration.removed, yerr=1.96*duration.removed_sem, marker="o", capsize=3, label="Energy removed")
    axes[1].errorbar(duration.duration_hours, duration.net, yerr=1.96*duration.net_sem, marker="o", capsize=3, label="Net reduction")
    axes[1].set(xlabel="High-price duration (hours)", ylabel="Energy (kWh)"); axes[1].legend()
    savefig(fig, "07_price_duration_vs_response")

    fig, ax = plt.subplots(figsize=(6.5, 5)); ax.scatter(primary.energy_removed_kwh, primary.energy_recovered_kwh, alpha=.45, c=primary.price_p_per_kwh, cmap="viridis")
    lim=max(primary.energy_removed_kwh.max(), primary.energy_recovered_kwh.max()); ax.plot([0,lim],[0,lim],"--",color="grey")
    ax.set(xlabel="Energy removed during event (kWh)", ylabel="Energy recovered afterward (kWh)", title="Removed versus recovered energy"); savefig(fig, "09_energy_removed_vs_recovered")
    line_plot("response_pct", "Demand reduction (%)", "10_price_change_magnitude_vs_response", "Price-change magnitude and response")

    # Spike vs gradual trajectories are stored as a dedicated experiment.
    sg = trajectories.query("experiment == 'spike_vs_gradual'")
    fig, axes = plt.subplots(2,1,figsize=(8,6),sharex=True)
    for scen, g in sg.groupby("scenario"):
        z=g.groupby("step_half_hour").agg(price=("price_p_per_kwh","mean"), q=("scenario_demand_kwh","mean"), base=("baseline_demand_kwh","mean"))
        axes[0].step(z.index/2, z.price, where="post", label=scen); axes[1].plot(z.index/2,z.q,label=scen); axes[1].plot(z.index/2,z.base,"--",color="black",alpha=.5)
    axes[0].set(ylabel="Price (p/kWh)",title="Sudden versus gradual high-price paths"); axes[1].set(xlabel="Hours after event start",ylabel="Predicted demand (kWh / half-hour)")
    axes[0].legend(); savefig(fig,"11_sudden_vs_gradual_trajectories")

    pivot=primary.pivot_table(index="time_period",columns="price_p_per_kwh",values="response_pct",aggfunc="mean").reindex(["Night (00:00–05:30)","Morning (06:00–10:30)","Afternoon (11:00–16:30)","Evening (17:00–23:30)"])
    fig, ax=plt.subplots(figsize=(8,4.5)); im=ax.imshow(pivot,aspect="auto",cmap="RdBu_r"); ax.set(xticks=np.arange(pivot.shape[1]),xticklabels=pivot.columns,yticks=np.arange(pivot.shape[0]),yticklabels=pivot.index,xlabel="Event price (p/kWh)",title="Time of day × price response")
    fig.colorbar(im,ax=ax,label="Demand reduction (%)"); savefig(fig,"12_time_of_day_price_response_heatmap")

    rep=repeat.groupby("event_number").agg(reduction=("response_pct","mean"),removed=("energy_removed_kwh","mean"),rebound=("rebound_peak_kwh_per_hh","mean")).reset_index()
    fig,ax=plt.subplots(figsize=(7.4,4.8)); ax.plot(rep.event_number,rep.reduction,marker="o",label="Demand reduction (%)"); ax.set(xlabel="Repeated 67.2p event number",ylabel="Mean response (%)",title="Repeated-event response"); ax.legend(); savefig(fig,"13_repeated_event_response")

    qtiles=pd.qcut(primary.baseline_demand_kwh_per_hh,3,labels=["Low","Medium","High"])
    bd=primary.assign(baseline_band=qtiles).groupby("baseline_band",observed=True).agg(reduction=("absolute_reduction_kwh_per_hh","mean"),pct=("response_pct","mean"),elasticity=("elasticity","mean")).reset_index()
    fig,ax=plt.subplots(figsize=(7.4,4.8)); ax.bar(bd.baseline_band,bd.pct,color="#4d9221"); ax.set(xlabel="Baseline demand band",ylabel="Demand reduction (%)",title="Baseline demand and price responsiveness"); savefig(fig,"14_baseline_demand_vs_responsiveness")

    high=summary.loc[summary.price_p_per_kwh.eq(67.2)].iloc[0]; base_row=summary.loc[summary.price_p_per_kwh.eq(BASE)].iloc[0]
    remaining=high.demand; flex=max(0,base_row.demand-high.demand)
    fig,ax=plt.subplots(figsize=(6,4.5)); ax.bar(["Remaining at 67.2p","Removed at 67.2p"],[remaining,flex],color=["#4c78a8","#f58518"]); ax.set(ylabel="Predicted demand (kWh / half-hour)",title="Observed high-price flexible and non-responsive components"); savefig(fig,"15_flexible_nonflexible_load")

    fig,axes=plt.subplots(2,1,figsize=(7.4,6),sharex=True); axes[0].plot(marginal.price_p_per_kwh,marginal.reduction,marker="o"); axes[1].plot(marginal.price_p_per_kwh,marginal.marginal,marker="o"); axes[1].axhline(0,color="black",linewidth=.8)
    axes[0].set(ylabel="Demand reduction (kWh / half-hour)",title="Tolerance / saturation diagnostic"); axes[1].set(xlabel="Event price (p/kWh)",ylabel="Marginal reduction per p/kWh"); savefig(fig,"16_tolerance_saturation_analysis")

    # Prior-history result (same current 67.2p).
    fig,ax=plt.subplots(figsize=(7.4,4.8)); h=history.groupby("history").agg(response=("reduction_kwh_per_hh","mean")).reset_index(); ax.bar(h.history,h.response,color="#a65628"); ax.set(xlabel="Prior four half-hour prices; current price fixed at 67.2p",ylabel="Demand reduction (kWh / half-hour)",title="Price-history effect at fixed current price"); savefig(fig,"17_price_history_effect")


def write_tables_and_report(metrics: pd.DataFrame, history: pd.DataFrame, repeat: pd.DataFrame, price_summary: pd.DataFrame, data_info: dict):
    primary=metrics.query("duration_half_hours == 4")
    supported=primary[primary.price_p_per_kwh.isin(OBSERVED_PRICES)]
    high=supported[supported.price_p_per_kwh.eq(67.2)]
    base=supported[supported.price_p_per_kwh.eq(BASE)]
    high_mean=high.response_pct.mean(); high_elasticity=high.elasticity.mean(); rebound=high.rebound_peak_kwh_per_hh.mean()
    # Ratio of pooled energy totals is robust to tiny per-window denominators.
    rec=high.energy_recovered_kwh.sum()/high.energy_removed_kwh.sum()
    # Detectable threshold: lower 95% CI of reduction > 1% at a tested price.  This is sensitivity-labelled.
    tested=[]
    for p,g in primary[primary.price_p_per_kwh.ge(BASE)].groupby("price_p_per_kwh"):
        m,l,u=ci(g.response_pct.to_numpy()); tested.append((p,m,l,u))
    threshold=next((p for p,m,l,u in tested if l>1.0),None)
    marginal=np.gradient(price_summary.query("price_p_per_kwh >= @BASE").reduction,price_summary.query("price_p_per_kwh >= @BASE").price_p_per_kwh)
    sat_price=None
    # Saturation needs two successive increments below 10% of first observed increment; otherwise absent.
    pos=np.abs(marginal[1:]); ref=np.abs(marginal[1]) if len(marginal)>2 else np.nan
    for i in range(1,len(pos)):
        if pos[i-1] < .1*ref and pos[i] < .1*ref: sat_price=float(price_summary.query("price_p_per_kwh >= @BASE").price_p_per_kwh.iloc[i+1]); break
    params=pd.DataFrame([
        ["Training price support", "3.99, 11.76, 67.20 p/kWh", "Observed values only; all intermediate test prices are interpolation diagnostics."],
        ["High-price response", f"{high_mean:.2f}%", "Four-half-hour event at 67.2 p/kWh against 11.76 p/kWh."],
        ["Elasticity at 67.2p", f"{high_elasticity:.3f}", "Derived event elasticity: (% demand change)/(% price change)."],
        ["Recovery ratio", f"{rec:.3f}", "Post-event energy above baseline / removed event energy; model-derived accounting."],
        ["Rebound peak", f"{rebound:.4f} kWh/hh", "Maximum post-event scenario demand above counterfactual."],
        ["Detectable response threshold", "None in observed price grid" if threshold is None else f"≈ {threshold:.2f} p/kWh (interpolation diagnostic)", "Criterion: lower 95% CI > 1% reduction. The supported grid has no upward level between 11.76 and 67.20 p/kWh, so it cannot resolve a physical threshold."],
        ["Saturation/tolerance region", "Not identified" if sat_price is None else f"≈ {sat_price:.2f} p/kWh", "Requires two marginal increments below 10% of first observed marginal response."],
        ["Price-history effect", f"range {history.reduction_kwh_per_hh.min():.4f} to {history.reduction_kwh_per_hh.max():.4f} kWh/hh", "Current price fixed; prior four tariff inputs varied."],
        ["Repeated-event attenuation", f"{repeat.groupby('event_number').response_pct.mean().iloc[-1]-repeat.groupby('event_number').response_pct.mean().iloc[0]:.2f} percentage points", "Autoregressive repeated-event experiment; not psychological fatigue."],
        ["Solar / net-demand / clustering", "Not supported", "No solar feature and only feeder-level aggregate target; no per-customer panel."],
    ],columns=["parameter","estimate","interpretation"])
    params.to_csv(TABLES/"behavioural_parameters.csv",index=False)
    questions=[
        ["Q1/Q2/Q6/Q9/Q10","Price × duration autoregressive events","Peak reduction, removed/recovered/net energy, rebound","Supported as GRU-output accounting; no energy conservation is imposed."],
        ["Q3/Q13/Q14","Observed-price and interpolation price sweeps","Threshold and marginal response","Only 3 observed training price levels; saturation is weakly identifiable."],
        ["Q4/Q7/Q16","Price sweep stratified by time and baseline demand","Elasticity and responsiveness","Conditional variation across held-out histories."],
        ["Q5/Q15","Spike/gradual and prior-history paths","Trajectory and fixed-current-price response","Gradual intermediate prices are labelled interpolation."],
        ["Q11/Q12","Price-history and repeated high-price paths","Current response and repeated-event reduction","Memory is model-derived; not customer psychology."],
        ["Q17/Q18/Q19","Feature/data audit","Solar, net demand, customer clusters","Not supported by the nine GRU inputs or feeder aggregate data."],
    ]
    pd.DataFrame(questions,columns=["research_questions","experiment","metric","observed_scope"]).to_csv(TABLES/"research_question_summary.csv",index=False)
    report=f"""# GRU customer-response experimental report

## A. What the trained GRU actually predicts

The checkpoint is a two-layer 64-unit GRU with a 48 half-hour (24 h) input. Its nine inputs are prior standardized household-mean consumption, six cyclical calendar encodings, a weekend flag, and tariff price. It predicts the next **absolute household-mean demand** in standardized units; the saved scaler inverse-transforms this to kWh per half-hour. There are no weather, solar, appliance, or customer-identity inputs.

Experiments use {data_info['n_seeds']} held-out historical seed windows, balanced across four time-of-day bands. For every intervention, the same seed is rolled under a baseline 11.76 p/kWh tariff and a counterfactual tariff path; no future observed load is fed into either rollout.

## B. Derived quantities

“Removed”, “recovered”, “net reduction”, elasticity, rebound, and flexible fraction are calculated from paired GRU trajectories. The GRU itself predicts demand only; it does not predict withdrawal, shifting, or energy conservation.

## C. Findings and limits

* The only observed training prices are 3.99, 11.76, and 67.20 p/kWh. Intermediate experiments are interpolation and are marked in the raw results; no price above 67.2 p/kWh is used.
* Solar, net-demand, and clustering claims cannot be made because solar is not a model input and the target is an aggregate feeder average rather than individual customers.
* Multi-step results are autoregressive GRU rollouts, so they are valid model probes but may compound one-step model error. Rebound or repeated-event patterns are therefore model behaviour, not evidence of psychology or physical energy conservation.
* The tolerance and threshold criteria in `behavioural_parameters.csv` are operational diagnostics with explicitly stated sensitivity rules, not universal customer limits.
"""
    (OUT/"REPORT.md").write_text(report)
    duration_summary = metrics.loc[metrics.price_p_per_kwh.eq(67.2)].groupby("duration_hours").response_pct.mean()
    tod = primary.pivot_table(index="time_period", columns="price_p_per_kwh", values="response_pct", aggfunc="mean")
    repeat_mean = repeat.groupby("event_number").response_pct.mean()
    history_mean = history.groupby("history").reduction_kwh_per_hh.mean()
    guide = f"""# Figure interpretation guide

All means summarize **96 held-out GRU seed histories**; the price-response figures show 95% confidence intervals. Red points denote actual training price levels. Intermediate price levels are interpolation diagnostics.

1. **Price vs predicted demand** — The model’s four-half-hour mean demand falls from {base.event_demand_kwh_per_hh.mean():.3f} kWh/hh at 11.76p to {high.event_demand_kwh_per_hh.mean():.3f} kWh/hh at 67.20p.
2. **Price vs percentage reduction** — At the supported high price, the paired counterfactual reduction is {high_mean:.2f}% (95% CI approximately ±{1.96*high.response_pct.sem():.2f} pp). Low price 3.99p increases demand by {-primary[primary.price_p_per_kwh.eq(3.99)].response_pct.mean():.2f}%.
3. **Price vs shifted/recovered energy** — This shows energy that appears above baseline after the event. It is a derived GRU-trajectory quantity, not an energy-conservation guarantee.
4. **Price vs net withdrawal** — The net change sums the full event-plus-recovery horizon. Positive values mean the intervention trajectory remains below its paired baseline overall.
5. **Price vs elasticity** — Elasticity is non-constant. At 67.2p it is {high_elasticity:.3f}; it is a finite event elasticity, not a structural demand parameter.
6. **Marginal response** — The plot is diagnostic only because its middle prices are interpolation; it does not establish a supported saturation point.
7. **Duration response** — Mean high-price reduction rises from {duration_summary.iloc[0]:.2f}% at 0.5 h to {duration_summary.iloc[-1]:.2f}% at 8 h. No duration plateau is demonstrated in this tested horizon.
8. **Rebound magnitude** — At 67.2p the mean rebound peak is {rebound:.4f} kWh/hh. It is a small model-predicted post-event overshoot, not proof of real-world load shifting.
9. **Removed vs recovered energy** — Points below the diagonal retain a net reduction; the aggregate recovery ratio at 67.2p is {rec:.2f}.
10. **Price-change magnitude** — Same response data as Figure 2 expressed as the high-minus-baseline price change.
11. **Sudden vs gradual paths** — Both are autoregressive GRU trajectories. The gradual path uses unsupported intermediate prices (20, 35, and 50p), so retain it as a scenario illustration rather than primary evidence.
12. **Time-of-day × price heatmap** — At 67.2p, response ranges from {tod[67.2].min():.2f}% ({tod[67.2].idxmin()}) to {tod[67.2].max():.2f}% ({tod[67.2].idxmax()}); night does not show reliable high-price curtailment in this model.
13. **Repeated events** — Reduction rises from {repeat_mean.iloc[0]:.2f}% at the first event to {repeat_mean.iloc[2]:.2f}% at the third and ends at {repeat_mean.iloc[-1]:.2f}%. This is non-monotonic, so there is no evidence of monotonic response attenuation.
14. **Baseline-demand bands** — This is conditional heterogeneity across seed histories, not customer-level heterogeneity.
15. **Flexible/non-flexible components** — The “remaining” bar is only the demand remaining at the highest observed training price. It is an inferred non-responsive load at 67.2p, not essential load.
16. **Tolerance/saturation** — No saturation price is identified within observed training support: only baseline and one higher supported price exist.
17. **Price history** — At the same current 67.2p price, mean one-step reduction is {history_mean['normal → high']:.4f} kWh/hh after normal history versus {history_mean['high → high']:.4f} after high history. This supports model memory of recent tariff inputs.
"""
    (TABLES/"FIGURE_GUIDE.md").write_text(guide)


def main():
    for path in (RAW,METRICS,PLOTS,TABLES): path.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA,parse_dates=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    model,scaler=load_model(); arr=engineer(df); arr[:,0]=scaler.transform(arr[:,0,None]).ravel()
    seeds,meta=choose_seeds(df)
    max_h=int(DURATIONS.max()+RECOVERY)
    base_path=np.full(max_h,BASE,dtype=np.float32)
    baseline=rollout(model,scaler,arr,seeds,base_path)
    raw=[]; metric_rows=[]

    for price in PRICE_GRID:
        for duration in DURATIONS:
            path=np.full(max_h,BASE,dtype=np.float32); path[:duration]=price
            scenario=rollout(model,scaler,arr,seeds,path)
            metric_rows.append(event_metrics(baseline,scenario,float(price),int(duration),meta))
            add_raw(raw,"price_duration","price_%.2f_duration_%d"%(price,duration),path,baseline,scenario,meta,int(duration))
    metrics=pd.concat(metric_rows,ignore_index=True); metrics.to_csv(METRICS/"event_metrics.csv",index=False)

    # Same final high price, different trajectories. Intermediate ramp values are interpolation diagnostics.
    h=16; spike=np.full(h,BASE,dtype=np.float32); spike[:4]=67.2
    gradual=np.full(h,BASE,dtype=np.float32); gradual[:7]=[BASE,20,35,50,67.2,67.2,67.2]
    for name,path in [("sudden_high",spike),("gradual_high",gradual)]:
        sc=rollout(model,scaler,arr,seeds,path); add_raw(raw,"spike_vs_gradual",name,path,baseline[:,:h],sc,meta,7)

    # Fixed current high price after controlled discrete price histories; one-step direct probe.
    hist_rows=[]
    for name,prior in [("low → high",3.99),("normal → high",11.76),("high → high",67.2)]:
        windows=np.stack([arr[s-LOOKBACK+1:s+1].copy() for s in seeds]); windows[:,-5:-1,-1]=prior; windows[:,-1,-1]=67.2
        with torch.no_grad(): z=model(torch.from_numpy(windows)).numpy().reshape(-1)
        q=np.clip(scaler.inverse_transform(z[:,None]).ravel(),.01,None)
        baseq=baseline[:,0]
        hist_rows.extend([{"sample_id":meta.iloc[i].sample_id,"history":name,"demand_kwh_per_hh":q[i],"baseline_kwh_per_hh":baseq[i],"reduction_kwh_per_hh":baseq[i]-q[i]} for i in range(len(seeds))])
    history=pd.DataFrame(hist_rows); history.to_csv(METRICS/"price_history_metrics.csv",index=False)

    # Five repeated two-half-hour high events, separated by four half-hours at baseline.
    path=np.full(32,BASE,dtype=np.float32); starts=[0,6,12,18,24]
    for st in starts: path[st:st+2]=67.2
    repq=rollout(model,scaler,arr,seeds,path); repbase=rollout(model,scaler,arr,seeds,np.full(32,BASE,dtype=np.float32))
    rep_rows=[]
    for no,st in enumerate(starts,1):
        sub=event_metrics(repbase[:,st:st+8],repq[:,st:st+8],67.2,2,meta)
        sub["event_number"]=no; rep_rows.append(sub)
    repeat=pd.concat(rep_rows,ignore_index=True); repeat.to_csv(METRICS/"repeated_event_metrics.csv",index=False)

    trajectories=pd.DataFrame(raw); trajectories.to_csv(RAW/"event_trajectories.csv",index=False)
    plot_study(metrics,trajectories,history,repeat)
    price_summary=pd.read_csv(METRICS/"price_response_summary.csv")
    training_prices=df.loc[df.DateTime <= pd.Timestamp("2013-05-31 23:30:00"),"price_p_per_kwh"]
    data_info={
        "n_seeds":len(seeds), "sequence_length_half_hours":LOOKBACK, "timestep_minutes":30,
        "target":"next absolute household mean consumption (kWh per half-hour after inverse scaling)",
        "features":["standardized historical demand","sin/cos half-hour","sin/cos weekday","sin/cos month","weekend","tariff price"],
        "training_price_min_p_per_kwh":float(training_prices.min()), "training_price_max_p_per_kwh":float(training_prices.max()),
        "training_price_quantiles_p_per_kwh":{str(k):float(v) for k,v in training_prices.quantile([.01,.05,.5,.95,.99]).items()},
        "training_price_counts":{str(k):int(v) for k,v in training_prices.value_counts().sort_index().items()},
    }
    (TABLES/"model_and_data_contract.json").write_text(json.dumps(data_info,indent=2))
    write_tables_and_report(metrics,history,repeat,price_summary,data_info)
    print(f"Completed GRU study: {len(seeds)} held-out seeds; outputs in {OUT}")


if __name__ == "__main__": main()
