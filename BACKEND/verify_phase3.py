"""Phase 3 terminal verification — LLM reasoning boundary + human review.

Tests the backend DIRECTLY from the terminal (spec sections 14-17):

  * GET /api/health
  * claim single / provider single   (1 real claim, 1 real provider)
  * claim batch / provider batch     (10 + 10 real records)
  * POST /api/claims/explain, POST /api/providers/explain
  * POST /api/review/accept, POST /api/review/reject (+ stored state)
  * LLM input contract: receives the ML risk score + AE signals, NEVER the
    training target, NEVER computes a second risk score
  * score integrity: review decisions never change prediction/risk/signals

Run from the BACKEND directory:   python verify_phase3.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
os.chdir(BASE)
sys.path.insert(0, str(BASE))

API = "http://localhost:8000"
PASSED = 0
FAILED = 0


def check(name: str, condition: bool) -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}")


def json_safe(record: dict) -> dict:
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in record.items()
    }


def api(path: str, payload: dict | None = None, method: str | None = None):
    """Returns (status_code, parsed_json_or_None, raw_text)."""
    url = f"{API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method or ("POST" if data else "GET"),
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            text = resp.read().decode()
            return resp.status, json.loads(text) if text else None, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(text), text
        except json.JSONDecodeError:
            return exc.code, None, text


print("=" * 72)
print("PHASE 3 — LLM BOUNDARY + HUMAN REVIEW TERMINAL VERIFICATION")
print("=" * 72)

claim_records = pd.read_csv(BASE / "data" / "claimsfinal_with_target.csv", nrows=10).to_dict(orient="records")
provider_records = pd.read_csv(BASE / "data" / "Provider_with_potential_fraud.csv", nrows=10).to_dict(orient="records")

# Server runs WITHOUT an LLM key — the explain endpoints must fail clearly.
env = {k: v for k, v in os.environ.items() if k != "FWA_LLM_API_KEY"}
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
    cwd=str(BASE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
)

try:
    up = False
    for _ in range(60):
        status, body, _ = api("/api/health")
        if status == 200:
            up = True
            break
        time.sleep(2)
    print("\n[14] HEALTH")
    check("GET /api/health -> 200", up and body and body.get("status") == "ok")

    # ------------------------------------------------------------ singles
    print("\n[15] SINGLE ANALYSIS — 1 real claim + 1 real provider")
    s1, c1, _ = api("/api/claims/score", {"features": json_safe(claim_records[0])})
    check("claim single HTTP 200", s1 == 200)
    check("claim single schema (prediction/risk/rating/probabilities/signals/evidence)",
          all(k in c1 for k in ("claim_id", "predicted_class", "risk_score", "rating",
                                "class_probabilities", "signals", "evidence")))
    check("claim single prediction is a real class label",
          c1["predicted_class"] in ("Legitimate", "Fraud", "Waste", "Abuse"))
    check("claim single risk score in 0-100", 0 <= c1["risk_score"] <= 100)
    check("claim single signals present", len(c1["signals"]) >= 10)
    check("claim single evidence is a list", isinstance(c1["evidence"], list))

    s2, p1, _ = api("/api/providers/score", {"features": json_safe(provider_records[0])})
    check("provider single HTTP 200", s2 == 200)
    check("provider single schema",
          all(k in p1 for k in ("provider_npi", "predicted_class", "risk_score", "rating",
                                "class_probabilities", "signals", "evidence")))
    check("provider single prediction label",
          p1["predicted_class"] in ("Legitimate", "Suspicious"))
    check("provider single signals present", len(p1["signals"]) >= 10)

    # ------------------------------------------------------------ batches
    print("\n[15] BATCH ANALYSIS — 10 claims + 10 providers")
    s3, b1, _ = api("/api/claims/batch", {"rows": [json_safe(r) for r in claim_records], "use_dataset": False})
    check("claim batch HTTP 200", s3 == 200)
    check("claim batch results count == 10", len(b1["results"]) == 10)
    check("claim batch queue count == 10", len(b1["queue"]) == 10)
    q_risks = [item["risk_score"] for item in b1["queue"]]
    check("claim batch queue ordered by ML risk desc", q_risks == sorted(q_risks, reverse=True))
    check("claim batch summary present",
          all(k in b1["summary"] for k in ("total_records", "average_risk", "high_risk_count",
                                           "unique_claims", "unique_providers", "unique_beneficiaries")))
    check("claim batch rows carry prediction/risk/signals/evidence",
          all("predicted_class" in r and "signals" in r and "evidence" in r for r in b1["results"]))

    s4, b2, _ = api("/api/providers/batch", {"rows": [json_safe(r) for r in provider_records], "use_dataset": False})
    check("provider batch HTTP 200", s4 == 200)
    check("provider batch results count == 10", len(b2["results"]) == 10)
    check("provider batch queue count == 10", len(b2["queue"]) == 10)
    pq_risks = [item["risk_score"] for item in b2["queue"]]
    check("provider batch queue ordered by ML risk desc", pq_risks == sorted(pq_risks, reverse=True))
    check("provider batch summary present",
          all(k in b2["summary"] for k in ("total_records", "total_providers",
                                           "suspicious_providers", "average_risk")))

    # ------------------------------------------------------------ LLM endpoints
    print("\n[16] LLM ENDPOINTS — already-generated results, no rerun")
    st, _, raw = api("/api/claims/explain", {"analysis": c1})
    check("claim explain fails clearly without API key (HTTP 503)", st == 503)
    check("claim explain error names the missing configuration",
          "FWA_LLM_API_KEY" in raw and "not configured" in raw.lower())
    check("claim explain fabricates NO reasoning (no summary/reasoning fields)",
          '"summary"' not in raw and '"reasoning"' not in raw)

    st2, _, raw2 = api("/api/providers/explain", {"analysis": p1})
    check("provider explain fails clearly without API key (HTTP 503)", st2 == 503)
    check("provider explain fabricates NO reasoning", '"summary"' not in raw2)

    st3, stat, _ = api("/api/llm/status")
    check("GET /api/llm/status -> 200 and configured=false",
          st3 == 200 and stat is not None and stat.get("configured") is False)

    # LLM input contract — offline (app.llm builds the context, no HTTP needed)
    from app import llm as llm_boundary  # noqa: E402
    from app.schemas import ClaimScoreResult  # noqa: E402

    ctx = llm_boundary.build_claim_context(ClaimScoreResult(**c1))
    ctx_json = json.dumps(ctx)
    check("LLM input carries the case id", ctx["case_id"] == c1["claim_id"])
    check("LLM input carries the ML risk score",
          ctx["ml_output"]["risk_score"] == c1["risk_score"])
    check("LLM input carries the Analytical Engine signals",
          len(ctx["analytical_signals"]) == len(c1["signals"]))
    check("LLM input carries the evidence package",
          len(ctx["evidence"]) == len(c1["evidence"]))
    stripped = llm_boundary._strip_targets(
        {"target": 1, "Target_Label": 2, "Potential Fraud": 3, "keep": {"x": 1}})
    check("training target is stripped from LLM input",
          stripped == {"keep": {"x": 1}})
    check("no target column in the claim context",
          all(tok not in ctx_json.lower() for tok in ("target_label", "potential fraud")))
    check("LLM context contains no second risk score field",
          "second_score" not in ctx_json and list(ctx["ml_output"]) ==
          ["prediction", "prediction_id", "risk_score", "risk_rating", "class_probabilities"])

    # ------------------------------------------------------------ human review
    print("\n[17] HUMAN REVIEW — accept / reject")
    claim_case = c1["claim_id"]
    provider_case = p1["provider_npi"]
    ra, acc, _ = api("/api/review/accept", {"case_id": claim_case, "entity_type": "claim"})
    check("POST /api/review/accept -> 200", ra == 200)
    check("accept stores case_id/entity_type/review_status/reviewed_at/reviewer_action",
          all(k in acc for k in ("case_id", "entity_type", "review_status",
                                 "reviewed_at", "reviewer_action")))
    check("accept stores review_status=Accepted + reviewer_action=accept",
          acc["review_status"] == "Accepted" and acc["reviewer_action"] == "accept")

    rr, rej, _ = api("/api/review/reject", {"case_id": provider_case, "entity_type": "provider"})
    check("POST /api/review/reject -> 200", rr == 200)
    check("reject stores review_status=Rejected + reviewer_action=reject",
          rej["review_status"] == "Rejected" and rej["reviewer_action"] == "reject")

    gl, stored, _ = api(f"/api/review/claim/{claim_case}")
    check("stored review retrievable via GET", gl == 200 and stored["review_status"] == "Accepted")

    # Score integrity — review never changes ML output, signals or evidence.
    s5, c1_after, _ = api("/api/claims/score", {"features": json_safe(claim_records[0])})
    check("after review: prediction unchanged", c1_after["predicted_class"] == c1["predicted_class"])
    check("after review: risk score unchanged", c1_after["risk_score"] == c1["risk_score"])
    check("after review: signals unchanged", c1_after["signals"] == c1["signals"])
    check("after review: evidence unchanged", c1_after["evidence"] == c1["evidence"])
    s6, p1_after, _ = api("/api/providers/score", {"features": json_safe(provider_records[0])})
    check("after review: provider risk score unchanged", p1_after["risk_score"] == p1["risk_score"])
    check("after review: provider prediction unchanged",
          p1_after["predicted_class"] == p1["predicted_class"])

finally:
    server.terminate()
    server.wait(timeout=15)

print("\n" + "=" * 72)
print(f"RESULT: {PASSED} passed, {FAILED} failed")
print("=" * 72)
sys.exit(1 if FAILED else 0)
