"""Read household trajectories and the demand history preceding a recorded action."""

import numpy as np
import pandas as pd


def household_index(episode, household_id):
    ids = episode["context"]["household_ids"].astype(str)
    found = np.flatnonzero(ids == str(household_id))
    if len(found) != 1:
        raise ValueError("Select a household present in this episode.")
    return int(found[0])


def household_timeline(episode, household_id):
    """Keep interval energy, average power and cumulative accounts distinct."""
    h = household_index(episode, household_id)
    steps, arrays, context = episode["steps"], episode["arrays"], episode["context"]
    dt = float(episode["metadata"]["interval_hours"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive interval duration required.")
    n = len(steps)
    region = int(context["regions"][h])
    values = {f"{name}_kwh": array[:, h] for name, array in arrays.items()}
    values["reference_kwh"] = context["reference"][:n, h]
    result = pd.DataFrame(values)
    for name in ("requested", "raw", "delivered", "essential", "reference"):
        result[f"{name}_kw"] = result[f"{name}_kwh"] / dt
    previous = np.r_[context["initial_history"][h, -1], arrays["requested"][:-1, h]]
    result["previous_kw"] = previous / dt
    result["power_change_kw"] = result.requested_kw - result.previous_kw
    result["hour"] = steps.hour.to_numpy()
    result["price_nok_kwh"] = steps.price.to_numpy()
    result["recorded_regional_price_nok_kwh"] = context["market_prices"][region, :n]
    result["household_charge_nok"] = result.delivered_kwh * result.price_nok_kwh
    result["reference_charge_nok"] = (
        result.reference_kwh * result.recorded_regional_price_nok_kwh
    )
    result["unserved_kwh"] = np.maximum(result.requested_kwh - result.delivered_kwh, 0)
    for name in (
        "requested_kwh",
        "delivered_kwh",
        "reference_kwh",
        "household_charge_nok",
        "unserved_kwh",
    ):
        result[f"cumulative_{name}"] = result[name].astype(np.float64).cumsum()
    # This is a cohort reward, never an invented per-household allocation of it.
    result["cohort_reward"] = steps.reward.to_numpy()
    result["cumulative_cohort_reward"] = result.cohort_reward.cumsum()
    return result


def household_history(episode, household_id, hour):
    """Return the visible demand window BEFORE the selected action.

    Append earlier requested demand, which is the response simulator's feedback;
    never expose the selected hour's result, later requests or curtailed delivery.
    """
    if not isinstance(hour, (int, np.integer)) or not 1 <= hour <= len(
        episode["steps"]
    ):
        raise ValueError("Choose a recorded episode hour.")
    h = household_index(episode, household_id)
    initial = episode["context"]["initial_history"][h]
    past = episode["arrays"]["requested"][: hour - 1, h]
    values = np.r_[initial, past][-len(initial) :]
    hours = np.arange(hour - len(initial), hour)
    return pd.DataFrame(
        {
            "Relative hour": hours,
            "Demand kWh": values,
            "Average power kW": values / episode["metadata"]["interval_hours"],
            "Source": np.where(hours <= 0, "Starting history", "Simulated request"),
        }
    )
