"""
src/dynamic_pricing_controller.py
Forecast-Driven Dynamic Pricing Controller.
Maps unconstrained baseline load forecast (from Stage 1 Pure Model)
into an optimized dynamic tariff P*_{t+1} in pence/kWh.
"""

import numpy as np

class DynamicPricingController:
    """
    Dynamic tariff pricing policy:
    - Normal baseline: 11.76 p/kWh
    - Peak stress cap: 67.20 p/kWh
    - Off-peak / valley floor: 3.99 p/kWh
    """
    def __init__(
        self,
        p_normal: float = 11.76,
        p_high: float = 67.20,
        p_low: float = 3.99,
        l_stress: float = 0.24,  # kWh/hh threshold where grid stress begins (~80th percentile)
        l_peak: float = 0.34,    # kWh/hh severe congestion threshold
        l_valley: float = 0.16,  # kWh/hh valley threshold
        max_slew_rate: float = 25.0, # max allowed price delta per half-hour
    ):
        self.p_normal = p_normal
        self.p_high = p_high
        self.p_low = p_low
        self.l_stress = l_stress
        self.l_peak = l_peak
        self.l_valley = l_valley
        self.max_slew_rate = max_slew_rate
        self.last_price = p_normal

    def compute_tariff(self, forecasted_load_kwh: float, is_smooth: bool = True) -> float:
        """
        Calculates dynamic tariff based on forecasted load.
        """
        if forecasted_load_kwh <= self.l_valley:
            # Low tariff during valley/excess hours
            target_price = self.p_low
        elif forecasted_load_kwh <= self.l_stress:
            # Normal baseline tariff
            target_price = self.p_normal
        else:
            # Grid congestion warning: scale price up to 67.20 p/kWh
            ratio = (forecasted_load_kwh - self.l_stress) / max(1e-4, (self.l_peak - self.l_stress))
            ratio = min(1.0, max(0.0, ratio))
            target_price = self.p_normal + (self.p_high - self.p_normal) * ratio

        # Slew rate limitation to prevent erratic oscillations
        if is_smooth:
            delta = target_price - self.last_price
            if abs(delta) > self.max_slew_rate:
                target_price = self.last_price + np.sign(delta) * self.max_slew_rate

        self.last_price = target_price
        return float(np.clip(target_price, self.p_low, self.p_high))

    def reset(self):
        self.last_price = self.p_normal
