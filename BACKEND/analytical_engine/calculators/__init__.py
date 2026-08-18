"""Deterministic signal calculators for the Analytical Engine.

These modules are CALCULATORS, not agent tools and not orchestrator tools:

    input attributes -> deterministic calculation -> supporting signal

They never call an LLM, an orchestrator, or another agent; they never request
additional evidence; they never modify the ML risk score; they never generate
a second fraud score.

Shared arithmetic helpers live here; the claim and provider calculators are
fully separated (claim/provider isolation).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..state import SignalRecord

EPS = 1e-9


def num(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    """Read a numeric attribute; NaN / non-numeric -> default."""
    value = row.get(key, default)
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, value)))


def median_of(medians: dict[str, float], key: str, fallback: float) -> float:
    value = medians.get(key)
    return fallback if value is None else float(value)


def upside_deviation(value: float, median: float) -> float:
    """Relative amount ABOVE the training median, as percent (clamped at 0).

    When the training median is ~0 (e.g. Claims_Last_7_Days), any positive value
    already exceeds the median; the raw value is scaled linearly instead.
    """
    if median is None or abs(median) < EPS:
        return max(value, 0.0) * 10.0
    return max(value / (median + EPS) - 1.0, 0.0) * 100.0


def downside_deviation(value: float, median: float) -> float:
    """Relative amount BELOW the training median, as percent (clamped at 0)."""
    if median is None or abs(median) < EPS:
        return 0.0
    return max((median - value) / (abs(median) + EPS), 0.0) * 100.0


def abs_deviation(value: float, median: float) -> float:
    """Absolute relative deviation from the training median, as percent."""
    if median is None or abs(median) < EPS:
        return 0.0
    return abs(value - median) / (abs(median) + EPS) * 100.0


def severity_for(value: float) -> str:
    """Signal severity bands, aligned with the centralized risk-rating bands."""
    if value >= 75:
        return "High"
    if value >= 45:
        return "Medium"
    return "Low"


def make_signal(
    signal: str,
    value: float,
    interpretation: str,
    supporting_attributes: list[str],
    row: dict[str, Any],
) -> SignalRecord:
    value = round(clip(value), 1)
    attribute_values = {
        attr: round(v, 4) for attr in supporting_attributes if (v := num(row, attr)) is not None
    }
    return SignalRecord(
        signal=signal,
        value=value,
        interpretation=interpretation,
        supporting_attributes=supporting_attributes,
        severity=severity_for(value),
        attribute_values=attribute_values,
    )


def fmt_money(value: float | None) -> str:
    return f"${value:,.0f}" if value is not None else "n/a"


def fmt(value: float | None, nd: int = 2) -> str:
    return f"{value:,.{nd}f}" if value is not None else "n/a"
