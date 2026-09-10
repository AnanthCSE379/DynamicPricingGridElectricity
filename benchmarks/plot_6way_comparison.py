"""
plot_multiweek_seasonal_comparison.py
Generates 6 dedicated publication-ready charts for the Multi-Week & 4-Season Grid Benchmark:
1. 01_continuous_2week_winter_timeline.png
2. 02_seasonal_tariff_heatmap.png
3. 03_seasonal_demand_and_transformer_stress.png
4. 04_seasonal_operator_financial_breakdown.png
5. 05_seasonal_customer_bill_and_elasticity.png
6. 06_seasonal_pareto_frontier.png
Saved exclusively in plots/seasonal_multiweek_benchmark/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(ROOT_DIR, "plots/6way_benchmark")
RESULTS_FILE = os.path.join(OUT_DIR, "seasonal_benchmark_results.json")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    "regime1_static_lcl":       "#7f7f7f",  # Neutral Gray
    "regime2_gru_static_peak":  "#1f77b4",  # Muted Blue
    "regime3_gru_dynamic_math": "#ff7f0e",  # Safety Orange
    "regime4_gru_agent_hybrid": "#2ca02c",  # Vivid Green (Star)
    "regime5_pure_agent":       "#9467bd",  # Royal Purple
    "regime6_hybrid_advance":   "#d62728",  # Red
}

SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_COLORS = {
    "Winter": "#1f77b4",
    "Spring": "#2ca02c",
    "Summer": "#ff7f0e",
    "Autumn": "#d62728"
}

def load_data():
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)

def plot_1_continuous_winter_timeline(data):
    w2 = data["Winter_2Weeks"]
    times = [datetime.fromisoformat(t) for t in w2["regime1_static_lcl"]["times"]]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    # 1. Tariffs
    ax = axes[0]
    ax.plot(times, w2["regime1_static_lcl"]["tariffs"], color=COLORS["regime1_static_lcl"], label="1. Static Flat", alpha=0.7, linestyle=":")
    ax.plot(times, w2["regime3_gru_dynamic_math"]["tariffs"], color=COLORS["regime3_gru_dynamic_math"], label="3. Math", alpha=0.7)
    ax.plot(times, w2["regime4_gru_agent_hybrid"]["tariffs"], color=COLORS["regime4_gru_agent_hybrid"], label="4. Hybrid (Pre)", linewidth=2.2)
    ax.plot(times, w2["regime6_hybrid_advance"]["tariffs"], color=COLORS["regime6_hybrid_advance"], label="6. Hybrid (Adv)", linewidth=2.2, linestyle="--")
    ax.axhline(25.0, color="red", linestyle="--", alpha=0.5, label="Price Shock Threshold (25p)")
    ax.set_ylabel("Tariff (p/kWh)")
    ax.set_title("Panel A: Dispatched Retail Tariffs over Continuous 14-Day Winter Peak", fontweight="bold")
    ax.legend(loc="upper right", ncol=4, frameon=True)
    ax.grid(True, alpha=0.3)

    # 2. Demand vs Threshold
    ax = axes[1]
    ax.plot(times, w2["regime1_static_lcl"]["demands"], color=COLORS["regime1_static_lcl"], label="Baseline Demand", alpha=0.6, linestyle=":")
    ax.plot(times, w2["regime4_gru_agent_hybrid"]["demands"], color=COLORS["regime4_gru_agent_hybrid"], label="Managed (Preemptive)", linewidth=2.0)
    ax.plot(times, w2["regime6_hybrid_advance"]["demands"], color=COLORS["regime6_hybrid_advance"], label="Managed (Advance)", linewidth=2.0, linestyle="--")
    ax.axhline(0.25, color="red", linestyle="--", linewidth=1.8, label="Transformer Overload Limit (0.25 kWh/hh)")
    ax.set_ylabel("Load (kWh/hh)")
    ax.set_title("Panel B: Substation Feeder Loading & Peak Shaving Performance", fontweight="bold")
    ax.legend(loc="upper right", ncol=3, frameon=True)
    ax.grid(True, alpha=0.3)

    # 3. Cumulative Operator Net Profit
    ax = axes[2]
    for r_key in ["regime1_static_lcl", "regime2_gru_static_peak", "regime3_gru_dynamic_math", "regime4_gru_agent_hybrid", "regime5_pure_agent", "regime6_hybrid_advance"]:
        cum_profit = np.cumsum(w2[r_key]["profits"])
        ax.plot(times, cum_profit, color=COLORS[r_key], label=w2[r_key]["label"], linewidth=2.2 if "hybrid" in r_key else 1.8)
    ax.set_ylabel("Cumulative Profit (£)")
    ax.set_title("Panel C: Cumulative Operator Net Profit (£) over 14 Days (1,000 Homes)", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, frameon=True)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %b %d'))
    plt.xticks(rotation=0)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "01_continuous_2week_winter_timeline.png")
    plt.savefig(p)
    plt.close()
    print("  Saved 01_continuous_2week_winter_timeline.png")

def plot_2_seasonal_tariff_heatmap(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for i, s in enumerate(SEASONS):
        ax = axes[i]
        times = [datetime.fromisoformat(t) for t in data[s]["regime1_static_lcl"]["times"]]
        hh_indices = [(t.hour * 2 + t.minute // 30) for t in times]
        
        # Calculate mean diurnal tariff per half-hour slot (0 to 47)
        diurnal_flat = [np.mean([t for j, t in enumerate(data[s]["regime1_static_lcl"]["tariffs"]) if hh_indices[j] == slot]) for slot in range(48)]
        diurnal_math = [np.mean([t for j, t in enumerate(data[s]["regime3_gru_dynamic_math"]["tariffs"]) if hh_indices[j] == slot]) for slot in range(48)]
        diurnal_hyb  = [np.mean([t for j, t in enumerate(data[s]["regime4_gru_agent_hybrid"]["tariffs"]) if hh_indices[j] == slot]) for slot in range(48)]
        diurnal_pure = [np.mean([t for j, t in enumerate(data[s]["regime5_pure_agent"]["tariffs"]) if hh_indices[j] == slot]) for slot in range(48)]
        diurnal_adv  = [np.mean([t for j, t in enumerate(data[s]["regime6_hybrid_advance"]["tariffs"]) if hh_indices[j] == slot]) for slot in range(48)]

        hours = np.linspace(0, 23.5, 48)
        ax.plot(hours, diurnal_flat, color=COLORS["regime1_static_lcl"], linestyle=":", label="1. Static Flat")
        ax.plot(hours, diurnal_math, color=COLORS["regime3_gru_dynamic_math"], label="3. Double GRU + Math")
        ax.plot(hours, diurnal_hyb, color=COLORS["regime4_gru_agent_hybrid"], linewidth=2.5, label="4. Hybrid (Preemptive)")
        ax.plot(hours, diurnal_pure, color=COLORS["regime5_pure_agent"], linestyle="--", label="5. Pure Agent")
        ax.plot(hours, diurnal_adv, color=COLORS["regime6_hybrid_advance"], linewidth=2.5, linestyle="--", label="6. Hybrid (Advance)")
        ax.axhline(25.0, color="red", linestyle="--", alpha=0.4, label="Price Shock (25p)")

        ax.set_title(f"{s} Season Diurnal Tariff Profile", fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Tariff (p/kWh)")
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 4)])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "02_seasonal_tariff_heatmap.png")
    plt.savefig(p)
    plt.close()
    print("  Saved 02_seasonal_tariff_heatmap.png")

def plot_3_seasonal_demand_and_stress(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Transformer Stress Hours by Season
    regime_keys = ["regime1_static_lcl", "regime2_gru_static_peak", "regime3_gru_dynamic_math", "regime4_gru_agent_hybrid", "regime5_pure_agent", "regime6_hybrid_advance"]
    labels = ["1. Flat", "2. Peak", "3. Math", "4. Hybrid(Pre)", "5. Pure", "6. Hybrid(Adv)"]
    width = 0.14
    
    x = np.arange(len(SEASONS))
    width = 0.14

    for j, r in enumerate(regime_keys):
        stress_vals = [data[s][r]["grid_stress_hours"] for s in SEASONS]
        ax1.bar(x + j * width - width * 2.5, stress_vals, width, label=labels[j], color=COLORS[r])

    ax1.set_title("Panel A: Substation Overload Duration (>0.25 kWh/hh)", fontweight="bold")
    ax1.set_ylabel("Stress Duration (Hours / Week)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SEASONS)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Peak Appliance Curtailment %
    for j, r in enumerate(regime_keys[1:]): # skip flat
        curtail_vals = [data[s][r]["peak_curtailment_pct"] for s in SEASONS]
        ax2.bar(x + j * 0.16 - 0.32, curtail_vals, 0.16, label=labels[j+1], color=COLORS[r])

    ax2.set_title("Panel B: Peak Appliance Load Curtailment (%) by Season", fontweight="bold")
    ax2.set_ylabel("Peak Load Shaved (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SEASONS)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "03_seasonal_demand_and_transformer_stress.png")
    plt.savefig(p)
    plt.close()
    print("  Saved 03_seasonal_demand_and_transformer_stress.png")

def plot_4_seasonal_financials(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    regime_keys = ["regime1_static_lcl", "regime2_gru_static_peak", "regime3_gru_dynamic_math", "regime4_gru_agent_hybrid", "regime5_pure_agent", "regime6_hybrid_advance"]
    labels = ["1. Flat", "2. Peak", "3. Math", "4. Hybrid(Pre)", "5. Pure", "6. Hybrid(Adv)"]
    x = np.arange(len(SEASONS))
    width = 0.14

    # Panel 1: Net Operating Profit (£)
    for j, r in enumerate(regime_keys):
        profits = [data[s][r]["total_net_profit_gbp"] for s in SEASONS]
        ax1.bar(x + j * width - width * 2.5, profits, width, label=labels[j], color=COLORS[r])

    ax1.set_title("Panel A: Operator Weekly Net Profit (£) Across 4 Seasons", fontweight="bold")
    ax1.set_ylabel("Net Profit (£ / Week)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SEASONS)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Profit Margin %
    for j, r in enumerate(regime_keys):
        margins = [data[s][r]["profit_margin_pct"] for s in SEASONS]
        ax2.bar(x + j * width - width * 2.5, margins, width, label=labels[j], color=COLORS[r])

    ax2.set_title("Panel B: Operator Operating Margin (%) Across 4 Seasons", fontweight="bold")
    ax2.set_ylabel("Operating Margin (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SEASONS)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "04_seasonal_operator_financial_breakdown.png")
    plt.savefig(p)
    plt.close()
    print("  Saved 04_seasonal_operator_financial_breakdown.png")

def plot_5_seasonal_customer_bills(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    regime_keys = ["regime1_static_lcl", "regime2_gru_static_peak", "regime3_gru_dynamic_math", "regime4_gru_agent_hybrid", "regime5_pure_agent", "regime6_hybrid_advance"]
    labels = ["1. Flat", "2. Peak", "3. Math", "4. Hybrid(Pre)", "5. Pure", "6. Hybrid(Adv)"]
    x = np.arange(len(SEASONS))
    width = 0.14

    # Panel 1: Customer Unit Rate (p/kWh)
    for j, r in enumerate(regime_keys):
        rates = [data[s][r]["customer_effective_rate_p_kwh"] for s in SEASONS]
        ax1.bar(x + j * width - width * 2.5, rates, width, label=labels[j], color=COLORS[r])

    ax1.set_title("Panel A: Customer Effective Unit Rate (p/kWh) Across Seasons", fontweight="bold")
    ax1.set_ylabel("Unit Rate (p/kWh)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SEASONS)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Valley Filling Consumption Boost %
    for j, r in enumerate(regime_keys[1:]): # skip flat
        boosts = [data[s][r]["valley_consumption_boost_pct"] for s in SEASONS]
        ax2.bar(x + j * 0.16 - 0.32, boosts, 0.16, label=labels[j+1], color=COLORS[r])

    ax2.set_title("Panel B: Overnight Valley Appliance Participation (+%)", fontweight="bold")
    ax2.set_ylabel("Valley Load Boost (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SEASONS)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "05_seasonal_customer_bill_and_elasticity.png")
    plt.savefig(p)
    plt.close()
    print("  Saved 05_seasonal_customer_bill_and_elasticity.png")

def plot_6_seasonal_pareto_frontier(data):
    fig, ax = plt.subplots(figsize=(10, 7))

    markers = {"Winter": "o", "Spring": "s", "Summer": "^", "Autumn": "D"}
    regime_keys = ["regime1_static_lcl", "regime2_gru_static_peak", "regime3_gru_dynamic_math", "regime4_gru_agent_hybrid", "regime5_pure_agent", "regime6_hybrid_advance"]

    for s in SEASONS:
        for r in regime_keys:
            profit = data[s][r]["total_net_profit_gbp"]
            rate   = data[s][r]["customer_effective_rate_p_kwh"]
            stress = max(1.0, data[s][r]["grid_stress_hours"])
            # Size inversely proportional to stress
            size = max(50, 400 - stress * 15)

            ax.scatter(rate, profit, s=size, color=COLORS[r], marker=markers[s], alpha=0.85, edgecolors="black", linewidth=1.2)

    # Annotate key points
    ax.text(data["Winter"]["regime4_gru_agent_hybrid"]["customer_effective_rate_p_kwh"] + 0.1,
            data["Winter"]["regime4_gru_agent_hybrid"]["total_net_profit_gbp"] + 20,
            "Winter: Preemptive (Protects Grid)", fontweight="bold", color=COLORS["regime4_gru_agent_hybrid"])
    ax.text(data["Winter"]["regime6_hybrid_advance"]["customer_effective_rate_p_kwh"] + 0.1,
            data["Winter"]["regime6_hybrid_advance"]["total_net_profit_gbp"] - 150,
            "Winter: Advance (Maximizes Profit)", fontweight="bold", color=COLORS["regime6_hybrid_advance"])

    ax.set_title("Cross-Seasonal Pareto Frontier: Operator Profit vs Customer Rate", fontweight="bold")
    ax.set_xlabel("Customer Effective Rate (p/kWh) — Lower is Better for Consumers")
    ax.set_ylabel("Operator Weekly Net Profit (£) — Higher is Better for Utility")
    ax.grid(True, alpha=0.3)

    # Custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Winter Week', markerfacecolor='gray', markersize=9),
        Line2D([0], [0], marker='s', color='w', label='Spring Week', markerfacecolor='gray', markersize=9),
        Line2D([0], [0], marker='^', color='w', label='Summer Week', markerfacecolor='gray', markersize=9),
        Line2D([0], [0], marker='D', color='w', label='Autumn Week', markerfacecolor='gray', markersize=9),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "06_seasonal_pareto_frontier.png")
    plt.savefig(p)
    plt.close()
    print("  Saved 06_seasonal_pareto_frontier.png")

def main():
    print("Loading multi-week & seasonal simulation results...", flush=True)
    data = load_data()

    print("Generating Plot 1: Continuous 2-Week Winter Timeline...")
    plot_1_continuous_winter_timeline(data)

    print("Generating Plot 2: Seasonal Tariff Heatmap...")
    plot_2_seasonal_tariff_heatmap(data)

    print("Generating Plot 3: Seasonal Demand and Transformer Stress...")
    plot_3_seasonal_demand_and_stress(data)

    print("Generating Plot 4: Seasonal Financial Breakdown...")
    plot_4_seasonal_financials(data)

    print("Generating Plot 5: Seasonal Customer Bills & Elasticity...")
    plot_5_seasonal_customer_bills(data)

    print("Generating Plot 6: Seasonal Pareto Frontier...")
    plot_6_seasonal_pareto_frontier(data)

    print(f"\nAll 6 multi-week seasonal comparison plots saved to {OUT_DIR}!")

if __name__ == "__main__":
    main()
