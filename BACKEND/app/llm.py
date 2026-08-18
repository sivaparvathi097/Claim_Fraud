"""LLM explanation boundary — the layer AFTER deterministic routing.

    Selected case (an ALREADY-GENERATED analysis result incl. its routing
    decision)
        -> LLM reasoning (explanation only, called for EVERY route,
          including auto_approve)
        -> human review (Accept / Reject) where the route requires it

Hard rules enforced by this module:

* The ROUTING LAYER decides the route; the LLM only EXPLAINS it. The LLM is
  called for every route (auto_approve / fast_track / full_investigation)
  with route-specific instructions, but its output never decides or changes
  the route. Any route the LLM returns is compared against the deterministic
  route and IGNORED on mismatch (see app/main.py).
* The LLM receives the already-generated analysis result: ML output, raw and
  calibrated risk, TrustFlags, Analytical Engine signals/evidence, and the
  routing decision. ML inference, calibration, the Trust Gate and the
  Analytical Engine are NEVER rerun on this path.
* The training target is never passed to the LLM as an inference fact
  (``target`` / ``Target_Label`` / ``Potential Fraud`` are stripped from
  anything that reaches the prompt).
* The LLM is an explanation layer: it never calculates a new risk score and
  never overrides the ML prediction. The structured response echoes the
  prediction / risk scores / rating / route straight from the INPUT result.
* No hardcoded API keys — configuration comes from environment variables:

      FWA_LLM_API_KEY    (required — no key => clear configuration error)
      FWA_LLM_BASE_URL   (OpenAI-compatible chat-completions base URL)
      FWA_LLM_MODEL      (model name)

* If the provider is not configured or the call fails, the endpoint raises a
  clear error. It never fabricates reasoning, and the case keeps its
  deterministic route.
* COST NOTE: the LLM is called for EVERY explained case — batch explanation
  cost and latency scale with the total number of analyzed cases, including
  auto_approve cases. This is intentional, not a compute-saving optimization.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

API_KEY_ENV = "FWA_LLM_API_KEY"
BASE_URL_ENV = "FWA_LLM_BASE_URL"
MODEL_ENV = "FWA_LLM_MODEL"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT_SECONDS = 90

# Training-target columns that must never reach the LLM as inference facts.
_TARGET_KEYS = {"target", "target_label", "potential fraud"}


class LLMConfigurationError(RuntimeError):
    """The LLM provider is not configured (missing API key / base URL)."""


class LLMReasoningError(RuntimeError):
    """The configured LLM call failed or returned unusable output."""


# ----------------------------------------------------------------- configuration
def _api_key() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def _base_url() -> str:
    return os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL


def _model() -> str:
    return os.environ.get(MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL


def llm_status() -> dict[str, Any]:
    """Configuration status (safe to expose — never leaks the key)."""
    return {
        "configured": _api_key() is not None,
        "provider": "openai-compatible",
        "model": _model(),
        "base_url": _base_url(),
        "api_key_env": API_KEY_ENV,
    }


# ----------------------------------------------------------------- context build
def _strip_targets(payload: Any) -> Any:
    """Recursively drop training-target keys from anything sent to the LLM."""
    if isinstance(payload, dict):
        return {
            k: _strip_targets(v)
            for k, v in payload.items()
            if str(k).strip().lower() not in _TARGET_KEYS and v is not None
        }
    if isinstance(payload, list):
        return [_strip_targets(item) for item in payload if item is not None]
    return payload


def _signal_items(signals: list) -> list[dict[str, Any]]:
    return [
        {
            "signal": s.signal,
            "value": s.value,
            "severity": s.severity,
            "interpretation": s.interpretation,
            "supporting_attributes": s.supporting_attributes,
            "attribute_values": s.attribute_values,
        }
        for s in signals
    ]


def _upstream_and_routing(analysis: Any) -> dict[str, Any]:
    """Phase 3/4 context: raw + calibrated risk, TrustFlags and the
    deterministic routing decision — all already generated, read-only."""
    routing = getattr(analysis, "routing", None)
    routing_block = {
        "route": routing.route,
        "reasons": list(routing.reasons),
        "requires_human_review": routing.requires_human_review,
        "predicate_results": dict(routing.predicate_results),
    } if routing is not None else None
    flags = getattr(analysis, "trust_flags", None)
    # Keep the context plain-JSON serializable (the prompt json.dumps it).
    if flags is not None and not isinstance(flags, dict):
        flags = flags.model_dump() if hasattr(flags, "model_dump") else dict(flags)
    return {
        "raw_ml_risk_score": getattr(analysis, "raw_ml_risk_score", None),
        "calibrated_risk_score": getattr(analysis, "calibrated_risk_score", None),
        "prediction_label": getattr(analysis, "prediction_label", None)
        or getattr(analysis, "predicted_class", None),
        "risk_rating": getattr(analysis, "risk_rating", None) or getattr(analysis, "rating", None),
        "trust_flags": flags,
        "routing": routing_block,
    }


def build_claim_context(analysis: Any) -> dict[str, Any]:
    """LLM input for a claim = the ALREADY-GENERATED analysis result.

    Contains IDs, prediction + label, raw + calibrated risk, TrustFlags,
    class probabilities, Analytical Engine signals and evidence, the claim
    attributes needed for explanation, and the deterministic routing
    decision. Never the training target.
    """
    return _strip_targets({
        "case_id": analysis.claim_id,
        "entity_type": "claim",
        "ml_output": {
            "prediction": analysis.predicted_class,
            "prediction_id": analysis.predicted_class_id,
            "risk_score": analysis.risk_score,
            "risk_rating": analysis.rating,
            "class_probabilities": analysis.class_probabilities,
        },
        "upstream_and_routing": _upstream_and_routing(analysis),
        "analytical_signals": _signal_items(analysis.signals),
        "evidence": _signal_items(analysis.evidence),
        "claim_attributes": {
            "claim_id": analysis.claim_id,
            "provider_id": analysis.provider_id,
            "beneficiary_id": analysis.beneficiary_id,
            "claim_type": analysis.claim_type,
            "submitted_amount": analysis.submitted_amount,
            "allowed_amount": analysis.allowed_amount,
            "payment_amount": analysis.payment_amount,
            "deductible_amount": analysis.deductible_amount,
            "service_count": analysis.service_count,
            "duration_days": analysis.duration_days,
        },
    })


def build_provider_context(analysis: Any) -> dict[str, Any]:
    """LLM input for a provider = the ALREADY-GENERATED analysis result."""
    return _strip_targets({
        "case_id": analysis.provider_npi,
        "entity_type": "provider",
        "ml_output": {
            "prediction": analysis.predicted_class,
            "prediction_id": analysis.predicted_class_id,
            "risk_score": analysis.risk_score,
            "risk_rating": analysis.rating,
            "class_probabilities": analysis.class_probabilities,
        },
        "upstream_and_routing": _upstream_and_routing(analysis),
        "analytical_signals": _signal_items(analysis.signals),
        "evidence": _signal_items(analysis.evidence),
        "provider_attributes": {
            "provider_npi": analysis.provider_npi,
            "provider_state": analysis.provider_state,
            "provider_type": analysis.provider_type,
            "total_beneficiaries": analysis.total_beneficiaries,
            "total_services": analysis.total_services,
            "claim_count": analysis.claim_count,
            "weighted_avg_payment": analysis.weighted_avg_payment,
            "services_per_beneficiary": analysis.services_per_beneficiary,
            "peer_deviation_score": analysis.peer_deviation_score,
        },
    })


# ----------------------------------------------------------------- prompting
SYSTEM_PROMPT = """You are the explanation layer of a healthcare Fraud/Waste/Abuse \
risk investigation system. You receive an ALREADY-GENERATED analysis result from a \
trained XGBoost + LightGBM hybrid model plus deterministic Analytical Engine signals.

Your ONLY job is to convert that existing result into human-readable reasoning.

Strict rules:
- NEVER calculate, estimate or suggest a new or different risk score. The ML risk \
score in the input is final.
- NEVER override or question the ML prediction. Echo it as given.
- NEVER claim fraud is legally proven. Use cautious language such as "potentially \
suspicious", "the model indicates elevated risk", "supporting signals include".
- Clearly distinguish: (1) the model prediction, (2) the analytical evidence, and \
(3) the fact that the FINAL decision belongs to a human reviewer.
- Base every statement on the provided signals, evidence and attributes. Do not \
invent facts.
- The ROUTING decision in the input was made by the deterministic routing layer. \
You NEVER decide, change or question the route — you only EXPLAIN it. Echo the \
input route back unchanged in the "route" field. Do not invent routes.

The explanation must cover: the prediction, the risk level, why the model assigned \
this risk, the most important supporting signals, how those signals relate to the \
submitted case, and the relevant financial / behavioral / utilization / peer \
patterns where available, plus important caveats.

Respond with STRICT JSON only (no markdown fences), exactly this shape:
{
  "route": "the deterministic route from the input, unchanged",
  "summary": "one concise paragraph",
  "reasoning": ["explanation point 1", "explanation point 2", ...],
  "supporting_signals": ["names of the signals actually used"],
  "evidence_used": ["short description of the evidence items relied on"],
  "caveats": ["caveat 1", ...]
}"""

# Route-specific explanation instructions (Phase 5). The LLM is called for
# EVERY route — including auto_approve — with the matching instruction below.
# The wording follows the Phase 5 specification: the LLM explains the
# deterministic routing outcome, it never decides it.
ROUTE_INSTRUCTIONS: dict[str | None, str] = {
    "auto_approve": (
        "Explain in plain language why the case satisfied the configured "
        "automated approval policy. Reference the actual passed routing "
        "predicates, the calibrated risk score, the TrustFlags and the "
        "supporting Analytical Engine signals. Do NOT say that the case is "
        "mathematically proven legitimate or definitely legitimate; use "
        "wording such as 'the case satisfied the configured automated "
        "policy'. No human review is required for this route."
    ),
    "fast_track": (
        "Explain why the deterministic routing policy placed the case into "
        "fast-track. Identify the relevant risk, trust and Analytical Engine "
        "signals from the input. Explain why human review is still required. "
        "Do NOT change or suggest a different route."
    ),
    "full_investigation": (
        "Explain why the deterministic policy escalated the case to full "
        "investigation. Identify the strongest risk factors, trust concerns "
        "and supporting Analytical Engine evidence from the input. Explain "
        "why human review is required. Do NOT claim confirmed fraud unless "
        "the underlying prediction/evidence actually supports it — prefer "
        "'potentially suspicious'. Do NOT change or suggest a different route."
    ),
}

_FALLBACK_INSTRUCTION = (
    "Explain the deterministic routing decision recorded in the input, "
    "referencing the routing reasons, predicates, calibrated risk, TrustFlags "
    "and Analytical Engine signals. Do not change or suggest a different route."
)


def route_instruction(route: str | None) -> str:
    """Route-specific explanation instruction for the LLM user prompt."""
    return ROUTE_INSTRUCTIONS.get(route, _FALLBACK_INSTRUCTION)


def _user_prompt(context: dict[str, Any]) -> str:
    routing = (context.get("upstream_and_routing") or {}).get("routing") or {}
    route = routing.get("route")
    return (
        "Explain the following already-generated analysis result. Do not rerun any "
        "model; the ML output, signals and routing decision below are final.\n\n"
        f"Route-specific instruction ({route}): {route_instruction(route)}\n\n"
        + json.dumps(context, indent=2, default=str)
    )


# ----------------------------------------------------------------- call
def _parse_reasoning(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMReasoningError(f"LLM returned unstructured output: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
        raise LLMReasoningError("LLM output missing required structured fields")

    def as_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return []

    llm_route = data.get("route")
    return {
        "summary": data["summary"].strip(),
        "reasoning": as_str_list(data.get("reasoning")) or [data["summary"].strip()],
        "supporting_signals": as_str_list(data.get("supporting_signals")),
        "evidence_used": as_str_list(data.get("evidence_used")),
        "caveats": as_str_list(data.get("caveats")),
        # The route as ECHOED by the LLM. Informational only — app/main.py
        # compares it against the deterministic route and ignores mismatches;
        # the LLM route never becomes the source of truth.
        "llm_route": str(llm_route).strip() if isinstance(llm_route, str) else None,
    }


def request_reasoning(context: dict[str, Any]) -> dict[str, Any]:
    """Call the configured OpenAI-compatible provider for structured reasoning.

    Raises LLMConfigurationError when no API key is configured and
    LLMReasoningError on transport/parse failures — never a fabricated result.
    """
    api_key = _api_key()
    if api_key is None:
        raise LLMConfigurationError(
            "LLM provider is not configured: set the "
            f"{API_KEY_ENV} environment variable (optionally {BASE_URL_ENV} and "
            f"{MODEL_ENV}). No reasoning is fabricated."
        )

    body = json.dumps({
        "model": _model(),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(context)},
        ],
    }).encode()
    request = urllib.request.Request(
        f"{_base_url().rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise LLMReasoningError(f"LLM provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMReasoningError(f"LLM provider unreachable: {exc.reason}") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMReasoningError("LLM provider returned an unexpected payload") from exc
    return _parse_reasoning(content)


def model_provider_label() -> str:
    return f"openai-compatible:{_model()}"
