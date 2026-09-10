"""
src/grid_economics.py
Wholesale market pricing model and half-hourly financial accounting engine.
Computes grid revenue, wholesale procurement cost, congestion penalties,
and net profit for any pricing regime.
"""

import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# UK Wholesale Spot Price Curve (p/kWh)
# Synthetic diurnal pattern calibrated to UK Day-Ahead market averages
# ─────────────────────────────────────────────────────────────────────────────
WHOLESALE_HH_BASE = np.array([
    5.2, 4.9, 4.7, 4.5, 4.4, 4.3,   # 00:00 – 02:30  (overnight low)
    4.4, 4.6, 5.0, 5.8, 6.5, 7.2,   # 03:00 – 05:30  (early ramp)
    8.5, 9.8, 9.5, 9.0, 8.5, 8.2,   # 06:00 – 08:30  (morning peak)
    7.8, 7.5, 7.2, 7.0, 6.8, 6.5,   # 09:00 – 11:30  (midday dip)
    6.2, 5.9, 5.8, 5.6, 5.8, 6.2,   # 12:00 – 14:30  (solar suppression)
    7.0, 8.5, 11.0,15.0, 22.0,28.0, # 15:00 – 17:30  (evening ramp-up)
    32.0,30.0,25.0,20.0,16.0,12.0,  # 18:00 – 20:30  (peak stress)
    9.5, 8.0, 7.0, 6.2, 5.8, 5.4,   # 21:00 – 23:30  (wind-down)
], dtype=np.float32)  # shape: [48]

# Seasonal scaling factors (Winter heavier, Summer lighter)
SEASONAL_SCALE = {
    12: 1.35, 1: 1.40, 2: 1.30,   # Winter
    3: 1.10,  4: 1.00, 5: 0.95,   # Spring
    6: 0.88,  7: 0.85, 8: 0.87,   # Summer
    9: 0.95, 10: 1.10, 11: 1.20,  # Autumn
}

# Congestion / Substation Stress Cost: quadratic penalty per (kWh/hh)² above threshold.
# Calibrated so that peak congestion costs ≈ 5-15% of gross retail revenue per slot.
CONGESTION_GAMMA = 2.0   # £ per (kWh/hh)² per 1000 households per half-hour
FEEDER_THRESHOLD = 0.25    # kWh/hh per household: grid stress onset


def wholesale_price_at(dt: datetime) -> float:
    """
    Returns half-hourly wholesale electricity spot price in pence/kWh
    for a given datetime, accounting for time-of-day and seasonality.
    """
    hh_idx = dt.hour * 2 + dt.minute // 30
    base   = WHOLESALE_HH_BASE[hh_idx]
    scale  = SEASONAL_SCALE.get(dt.month, 1.0)
    # Add small Gaussian noise to simulate day-to-day price variability
    noise  = np.random.normal(0, base * 0.05)
    return float(np.clip(base * scale + noise, 1.5, 80.0))


def compute_halfhour_economics(
    demand_kwh:     float,
    retail_price_p: float,
    wholesale_p:    float,
    n_households:   int = 1000,
) -> dict:
    """
    Computes financial metrics for one half-hour slot.
    Returns values scaled to per 1000 households.

    Args:
        demand_kwh:     Average demand per household (kWh/hh)
        retail_price_p: Dispatched retail tariff (pence/kWh)
        wholesale_p:    Wholesale spot price (pence/kWh)
        n_households:   Number of households in the feeder (default: 1000)
    """
    total_demand_kwh = demand_kwh * n_households

    # Revenue: retail tariff collected from customers
    revenue_p = retail_price_p * total_demand_kwh

    # Wholesale procurement cost: what the utility pays to buy energy
    wholesale_cost_p = wholesale_p * total_demand_kwh

    # Congestion / substation penalty: quadratic cost above threshold
    excess = max(0.0, demand_kwh - FEEDER_THRESHOLD)
    congestion_cost_p = CONGESTION_GAMMA * (excess ** 2) * n_households * 100  # convert to pence

    # Net profit in pence; convert to pounds
    net_profit_p = revenue_p - wholesale_cost_p - congestion_cost_p

    # Gross margin (%)
    margin_pct = (net_profit_p / revenue_p * 100) if revenue_p > 1e-6 else 0.0

    return {
        "demand_kwh":         demand_kwh,
        "retail_price_p":     retail_price_p,
        "wholesale_p":        wholesale_p,
        "total_demand_kwh":   total_demand_kwh,
        "revenue_gbp":        revenue_p / 100,
        "wholesale_cost_gbp": wholesale_cost_p / 100,
        "congestion_cost_gbp":congestion_cost_p / 100,
        "net_profit_gbp":     net_profit_p / 100,
        "margin_pct":         margin_pct,
    }
