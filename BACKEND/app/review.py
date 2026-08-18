"""Human review store — the final decision layer after LLM reasoning.

    Selected case -> LLM reasoning -> HUMAN REVIEW -> Accept / Reject

The human reviewer is the final decision-maker. A review record stores the
decision next to the analysis result; it NEVER overwrites the ML prediction
or the ML risk score, and it is never converted back into a model prediction.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

_LOCK = threading.Lock()
# (entity_type, case_id) -> review record
_RECORDS: dict[tuple[str, str], dict[str, Any]] = {}

_VALID_ACTIONS = {"accept": "Accepted", "reject": "Rejected"}


def record_review(case_id: str, entity_type: str, action: str) -> dict[str, Any]:
    """Store an Accept/Reject decision. Returns the stored record."""
    status = _VALID_ACTIONS[action]
    record = {
        "case_id": case_id,
        "entity_type": entity_type,
        "review_status": status,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_action": action,
    }
    with _LOCK:
        _RECORDS[(entity_type, case_id)] = record
    return record


def get_review(entity_type: str, case_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _RECORDS.get((entity_type, case_id))


def all_reviews() -> list[dict[str, Any]]:
    with _LOCK:
        return list(_RECORDS.values())
