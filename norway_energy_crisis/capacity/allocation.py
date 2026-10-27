"""Essential-first allocation of one region's hourly energy budget."""

from dataclasses import dataclass

import numpy as np

from norway_energy_crisis.clean_data import require

METHODS = ("fair", "proportional", "fcfs")


@dataclass(frozen=True)
class AllocationResult:
    delivered: np.ndarray
    essential_delivered: np.ndarray
    flexible_delivered: np.ndarray
    essential_unserved: np.ndarray
    flexible_unserved: np.ndarray
    remaining_kwh: float
    essential_infeasible: bool


def allocate(
    request_kwh,
    essential_kwh,
    budget_kwh,
    requested_before=None,
    served_before=None,
    method="fair",
    order=None,
):
    """Allocate one pool only; history affects priority but never creates backlog."""
    requested = np.asarray(request_kwh, dtype=np.float64)
    essential = np.asarray(essential_kwh, dtype=np.float64)
    require(
        requested.ndim == 1 and essential.shape == requested.shape,
        "Request/essential vectors must match.",
    )
    require(
        np.isfinite(requested).all()
        and np.isfinite(essential).all()
        and (essential >= 0).all()
        and (requested >= essential).all(),
        "Requests must be finite and at least the nonnegative essential floor.",
    )
    require(
        np.isfinite(budget_kwh) and budget_kwh >= 0 and method in METHODS,
        "Invalid budget or allocator.",
    )
    flexible = requested - essential
    essential_total = float(essential.sum())
    infeasible = bool(essential_total > budget_kwh)
    if infeasible:
        essential_served = essential * (budget_kwh / essential_total)
        flexible_served = np.zeros_like(flexible)
    else:
        essential_served = essential.copy()
        remaining = min(max(budget_kwh - essential_total, 0.0), float(flexible.sum()))
        if remaining >= flexible.sum():
            flexible_served = flexible.copy()
        elif remaining == 0:
            flexible_served = np.zeros_like(flexible)
        elif method == "proportional":
            flexible_served = flexible * (remaining / flexible.sum())
        elif method == "fcfs":
            order = np.asarray(order, dtype=int)
            require(
                order.shape == requested.shape
                and np.array_equal(np.sort(order), np.arange(len(requested))),
                "FCFS requires a complete synthetic permutation.",
            )
            before = np.r_[0.0, np.cumsum(flexible[order])[:-1]]
            flexible_served = np.zeros_like(flexible)
            flexible_served[order] = np.minimum(
                flexible[order], np.maximum(remaining - before, 0.0)
            )
        else:
            prior_requests = (
                np.zeros_like(flexible)
                if requested_before is None
                else np.asarray(requested_before, dtype=float)
            )
            prior_served = (
                np.zeros_like(flexible)
                if served_before is None
                else np.asarray(served_before, dtype=float)
            )
            require(
                prior_requests.shape == flexible.shape
                and prior_served.shape == flexible.shape
                and np.isfinite(prior_requests).all()
                and np.isfinite(prior_served).all()
                and (prior_requests >= 0).all()
                and (prior_served >= 0).all()
                and (prior_served <= prior_requests + 1e-8).all(),
                "Invalid allocation history.",
            )
            denominator = prior_requests + flexible
            low, high = 0.0, 1.0
            # At level one every current request is filled. Bisection never
            # reallocates old unserved demand; each result remains capped by f_i.
            for _ in range(55):
                level = (low + high) / 2
                candidate = np.clip(level * denominator - prior_served, 0.0, flexible)
                if candidate.sum() > remaining:
                    high = level
                else:
                    low = level
            flexible_served = np.clip(low * denominator - prior_served, 0.0, flexible)
    delivered = essential_served + flexible_served
    require(
        (delivered >= 0).all()
        and (delivered <= requested + 1e-8).all()
        and delivered.sum() <= budget_kwh + 1e-8,
        "Allocation violated energy conservation or capacity.",
    )
    return AllocationResult(
        delivered,
        essential_served,
        flexible_served,
        np.maximum(essential - essential_served, 0.0),
        np.maximum(flexible - flexible_served, 0.0),
        max(float(budget_kwh - delivered.sum()), 0.0),
        infeasible,
    )


def jain(values):
    """Distinguish empty eligible groups from groups receiving no service."""
    values = np.asarray(values, dtype=np.float64)
    require(
        np.isfinite(values).all() and (values >= 0).all(), "Invalid fairness values."
    )
    if not len(values):
        return None
    if not np.any(values):
        return 0.0
    return float(values.sum() ** 2 / (len(values) * np.square(values).sum()))


def service(reference_energy, delivered_energy):
    reference_energy = np.asarray(reference_energy, dtype=np.float64)
    delivered_energy = np.asarray(delivered_energy, dtype=np.float64)
    require(
        reference_energy.shape == delivered_energy.shape
        and np.isfinite(reference_energy).all()
        and np.isfinite(delivered_energy).all()
        and (reference_energy >= 0).all()
        and (delivered_energy >= 0).all(),
        "Invalid reference/delivered energy.",
    )
    active = reference_energy > 1e-6
    ratios = np.minimum(delivered_energy[active] / reference_energy[active], 1.0)
    return dict(
        loss=float(np.mean(1.0 - ratios)) if len(ratios) else 0.0,
        fairness=jain(ratios),
        eligible_households=int(active.sum()),
        service_ratios=ratios.tolist(),
    )
