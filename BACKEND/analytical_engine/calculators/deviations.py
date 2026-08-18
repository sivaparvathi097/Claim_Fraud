"""Top feature deviations vs the training medians stored in the artifact.

A feature deviation is its OWN output type — distinct from the ML model score
and from Analytical Engine signals. The three are never combined into another
hidden risk score.
"""
from __future__ import annotations

from typing import Any

from . import EPS, num


def top_feature_deviations(
    row: dict[str, Any], medians: dict[str, float], k: int = 5
) -> list[dict[str, Any]]:
    """Rank numeric features by relative deviation from the training median."""
    scored: list[tuple[float, str, float, float]] = []
    for col, median in medians.items():
        value = num(row, col)
        if value is None:
            continue
        denom = abs(median) + EPS
        rel = abs(value - median) / denom
        scored.append((rel, col, value, float(median)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "feature": col,
            "value": round(value, 4),
            "training_median": round(median, 4),
            "deviation": round(rel, 4),
        }
        for rel, col, value, median in scored[:k]
    ]
