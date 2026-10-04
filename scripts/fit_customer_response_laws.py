"""Fit simple behavioural laws to controlled CustomerTwin tariff experiments.

This script is deliberately downstream of analyze_customer_response.py:

    python scripts/analyze_customer_response.py
    python scripts/fit_customer_response_laws.py

It converts the neural model's price-response predictions into a customer
"withdrawal fraction" W(p) = 1 - q(p) / q(p0), where p0 = 14.23 p/kWh.
It then compares three interpretable candidate laws:

    Logarithmic: W = A log(1 + x)
    Power:       W = A x^B
    Saturating:  W = W_max (1 - exp(-K x))

with x = p / p0 - 1.  These are empirical summaries of a trained predictive
model, not estimates of a causal consumer demand function.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "plots/customer_response/response_summary.json"
OUT = ROOT / "plots/customer_response"
BASE_PRICE = 14.23


def rmse(observed: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - fitted) ** 2)))


def fit_laws(x: np.ndarray, w: np.ndarray) -> dict:
    """Fit simple non-negative forms. x and w must exclude the baseline point."""
    if np.any(w <= 0):
        # A positive withdrawal law cannot represent this context.  Reporting a
        # failed fit is more honest than turning a rebound into "elasticity".
        return {}
    # W = A log(1+x), fitted through zero.
    z = np.log1p(x)
    a_log = float(np.dot(z, w) / np.dot(z, z))
    log_pred = a_log * z

    # log(W) = log(A) + B log(x), valid when withdrawal is positive.
    b_power, log_a_power = np.polyfit(np.log(x), np.log(w), 1)
    a_power = float(np.exp(log_a_power))
    power_pred = a_power * x ** b_power

    # Grid search K; W_max is the least-squares coefficient at each K.
    best = None
    for k in np.geomspace(0.02, 20.0, 4000):
        basis = 1.0 - np.exp(-k * x)
        w_max = max(0.0, float(np.dot(basis, w) / np.dot(basis, basis)))
        pred = w_max * basis
        candidate = (rmse(w, pred), float(k), w_max, pred)
        if best is None or candidate[0] < best[0]:
            best = candidate
    sat_error, k_sat, w_max_sat, sat_pred = best

    return {
        "logarithmic": {
            "formula": "W(x) = A * ln(1 + x)", "A": a_log,
            "rmse": rmse(w, log_pred), "predicted": log_pred,
        },
        "power": {
            "formula": "W(x) = A * x^B", "A": a_power, "B": float(b_power),
            "rmse": rmse(w, power_pred), "predicted": power_pred,
        },
        "saturating_exponential": {
            "formula": "W(x) = W_max * (1 - exp(-K*x))", "W_max": w_max_sat, "K": k_sat,
            "rmse": sat_error, "predicted": sat_pred,
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(INPUT.read_text())
    results = {"definition": {"baseline_price_p_per_kwh": BASE_PRICE,
                               "cost_spike": "x = p / 14.23 - 1",
                               "withdrawal_fraction": "W = 1 - q(p) / q(14.23)"},
               "contexts": {}}

    contexts = summary["price_sweep"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    colors = {"logarithmic": "#457b9d", "power": "#e76f51", "saturating_exponential": "#2a9d8f"}

    for ax, (name, record) in zip(axes.flat, contexts.items()):
        prices = np.asarray(record["prices_p_per_kwh"], dtype=float)
        demand = np.asarray(record["mean_demand_kwh_per_hh"], dtype=float)
        baseline_index = int(np.argmin(np.abs(prices - BASE_PRICE)))
        q0 = demand[baseline_index]
        x_all = prices / BASE_PRICE - 1.0
        w_all = 1.0 - demand / q0
        use = x_all > 0
        x, w = x_all[use], w_all[use]
        fits = fit_laws(x, w)
        best_name = min(fits, key=lambda key: fits[key]["rmse"]) if fits else None

        dense_x = np.linspace(0, x.max(), 200)
        ax.scatter(x_all, w_all * 100, color="#222222", s=30, zorder=4, label="CustomerTwin outputs")
        for law, fit in fits.items():
            if law == "logarithmic":
                dense_y = fit["A"] * np.log1p(dense_x)
            elif law == "power":
                dense_y = fit["A"] * dense_x ** fit["B"]
            else:
                dense_y = fit["W_max"] * (1 - np.exp(-fit["K"] * dense_x))
            suffix = "  best" if law == best_name else ""
            ax.plot(dense_x, dense_y * 100, color=colors[law], linewidth=2,
                    label=f"{law.replace('_', ' ')}{suffix}")
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.25)
        note = (f"best: {best_name.replace('_', ' ')}\nRMSE = {fits[best_name]['rmse']*100:.2f} pp"
                if best_name else "No monotonic withdrawal law\n(price effect reverses at high prices)")
        ax.text(0.03, 0.94, note, transform=ax.transAxes, va="top", fontsize=9)

        serial_fits = {}
        for law, fit in fits.items():
            serial_fits[law] = {
                k: (round(float(v), 8) if isinstance(v, (float, np.floating)) else v)
                for k, v in fit.items() if k != "predicted"
            }
        results["contexts"][name] = {
            "n_intervals": record["n_intervals"], "baseline_demand_kwh_per_hh": q0,
            "prices_p_per_kwh": prices.tolist(), "withdrawal_fraction": w_all.round(8).tolist(),
            "best_law": best_name or "not_monotonic", "fits": serial_fits,
        }

    axes[1, 0].set_xlabel("Sudden cost spike, x = p / 14.23 − 1")
    axes[1, 1].set_xlabel("Sudden cost spike, x = p / 14.23 − 1")
    axes[0, 0].set_ylabel("Predicted withdrawal (%)")
    axes[1, 0].set_ylabel("Predicted withdrawal (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Which simple law best describes CustomerTwin withdrawal after a price spike?", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(OUT / "withdrawal_law_fits.png", dpi=220)
    plt.close(fig)

    (OUT / "withdrawal_law_fits.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# CustomerTwin: high-level behavioural laws",
        "",
        "Define the sudden cost spike as `x = p / 14.23 - 1` and the model-implied withdrawal fraction as `W = 1 - q(p) / q(14.23)`.",
        "",
        "These equations describe controlled interventions on the saved GRU checkpoint. They are conditional model summaries, not causal estimates from a customer experiment.",
        "",
    ]
    for name, item in results["contexts"].items():
        lines.extend([f"## {name}", ""])
        if item["best_law"] == "power":
            f = item["fits"]["power"]
            lines.extend([
                f"For price increases above 14.23 p/kWh, the best simple approximation is:",
                "",
                f"`W(x) = {f['A']:.6f} · x^{f['B']:.6f}`  (RMSE {f['rmse']*100:.2f} percentage points)",
                "",
                f"Therefore `q(p) ≈ {item['baseline_demand_kwh_per_hh']:.6f} · [1 − W(p / 14.23 − 1)]` kWh/half-hour.",
                "",
            ])
        else:
            lines.extend([
                "No simple monotonic withdrawal equation is supported: the model's high-price response reverses sign in this context.",
                "",
            ])
    (OUT / "CUSTOMER_BEHAVIORAL_LAWS.md").write_text("\n".join(lines))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
