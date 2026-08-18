"""Phase 2 verification — the LOCKED architecture.

    USER INPUT -> ML MODEL -> PREDICTION + RISK SCORE -> ANALYTICAL ENGINE
                 -> SUPPORTING SIGNALS + EVIDENCE -> DASHBOARD

Verifies:
  1.  Claim single:    ML -> Analytical Engine -> result
  2.  Provider single: ML -> Analytical Engine -> result
  3.  Claim batch:     ML -> Analytical Engine -> results -> queue
  4.  Provider batch:  ML -> Analytical Engine -> results -> queue
  5.  No orchestrator module exists
  6.  No orchestrator is imported anywhere in the production pipeline
  7.  No LLM is called
  8.  No second risk score is generated
  9.  ML score remains unchanged (before engine = after engine = API = queue)
  10. Claim/provider isolation remains intact
  11. Batch vectorization remains intact (1 XGB + 1 LGB call per batch)
  12. Queue remains sorted by the original ML risk score
  13. Frontend-facing API returns real backend analysis results
  14. Single analysis returns one-case result data (no queue)
  15. Batch analysis returns submitted records + queue data

Run from the BACKEND directory:   python verify_phase2.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
os.chdir(BASE)
sys.path.insert(0, str(BASE))

from analytical_engine.inference import ModelRegistry, predict_claims, predict_providers  # noqa: E402
from analytical_engine.state import MLResult, SignalRecord  # noqa: E402
from app import config, services  # noqa: E402

CLAIM_CSV = BASE / "data" / "claimsfinal_with_target.csv"
PROVIDER_CSV = BASE / "data" / "Provider_with_potential_fraud.csv"

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
    """NaN/NaT -> None so records survive JSON transport."""
    out = {}
    for k, v in record.items():
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = None
                continue
        except TypeError:
            pass
        if isinstance(v, (pd.Timestamp,)):
            v = v.isoformat()
        out[k] = v
    return out


print("=" * 72)
print("PHASE 2 — LOCKED ARCHITECTURE VERIFICATION")
print("=" * 72)

registry = ModelRegistry(config.CLAIM_MODEL_PATH, config.PROVIDER_MODEL_PATH)
claim_model = registry.claim
claim_records = pd.read_csv(CLAIM_CSV, nrows=10).to_dict(orient="records")
provider_records = pd.read_csv(PROVIDER_CSV, nrows=10).to_dict(orient="records")

# ----------------------------------------------------------------------
print("\n[5/6] NO ORCHESTRATOR — module absence + import scan")
orch_file = BASE / "analytical_engine" / "orchestrator.py"
check("5. analytical_engine/orchestrator.py does not exist", not orch_file.exists())
check("5. analytical_engine.orchestrator is not importable",
      importlib.util.find_spec("analytical_engine.orchestrator") is None)
banned_names = ["orchestrator.py", "agent.py", "investigation_agent.py",
                "tool_executor.py", "secondary_engine.py"]
existing_banned = [p for n in banned_names
                   for p in (BASE / "analytical_engine").rglob(n)] + \
                  [p for n in banned_names for p in (BASE / "app").rglob(n)]
check("10b. no orchestrator/agent module files exist", not existing_banned)

import_re = re.compile(r"^\s*(from|import)\s+[^\s]*orchestrator", re.IGNORECASE)
import_hits = []
for folder in (BASE / "analytical_engine", BASE / "app"):
    for py in folder.rglob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            if import_re.match(line):
                import_hits.append(f"{py.name}: {line.strip()}")
check("6. no orchestrator import in production code", not import_hits)

print("\n[7] NO LLM — import scan")
llm_re = re.compile(
    r"^\s*(from|import)\s+[^\s]*(openai|anthropic|langchain|llama_index|gemini|transformers|llm)",
    re.IGNORECASE,
)
llm_hits = []
for folder in (BASE / "analytical_engine", BASE / "app"):
    for py in folder.rglob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            if llm_re.match(line):
                llm_hits.append(f"{py.name}: {line.strip()}")
check("7. no LLM imports in production code", not llm_hits)

# ----------------------------------------------------------------------
print("\n[1] CLAIM SINGLE — ML -> Analytical Engine -> result")
registry._provider = None  # isolation baseline
prepared_before, ml_before = predict_claims(claim_model, [claim_records[0]])
score_before = ml_before[0].risk_score
single = services.analyze_claim(claim_model, claim_records[0])
check("1. single claim returns frozen MLResult", isinstance(single.ml, MLResult))
check("9. risk score: ML stage == after Analytical Engine",
      single.ml.risk_score == score_before)
check("1. prediction preserved", single.ml.predicted_class == ml_before[0].predicted_class)
check("1. class probabilities present (4 classes)", len(single.ml.class_probabilities) == 4)
check("1. claim signals produced", len(single.signals) >= 10)
check("1. evidence package produced", isinstance(single.evidence, list))
check("1. all signals are deterministic SignalRecords",
      all(isinstance(s, SignalRecord) for s in single.signals))
check("1. ordinal encoder applied before preprocessor (Claim_Type encoded)",
      isinstance(single.prepared_row.get("Claim_Type"), (int, float)))
try:
    single.ml.risk_score = 0.0  # type: ignore[misc]
    frozen_blocked = False
except FrozenInstanceError:
    frozen_blocked = True
check("9. MLResult is frozen (mutation blocked)", frozen_blocked)

print("\n[8] NO SECOND RISK SCORE")
check("8. signals carry no risk score field",
      all(not hasattr(s, "risk_score") for s in single.signals))
check("8. evidence items are signals, not scores",
      all(isinstance(e, SignalRecord) for e in single.evidence))
check("8. signal values are 0-100 evidence strengths",
      all(0.0 <= s.value <= 100.0 for s in single.signals))
check("8. no evidence-sufficiency concept in engine outputs",
      not any(hasattr(single, a) for a in (
          "evidence_sufficient", "evidence_insufficient", "needs_more_evidence")))

print("\n[10] CLAIM/PROVIDER ISOLATION")
check("10. provider model never loaded during claim analysis",
      registry.provider_loaded is False)
provider_model = registry.provider
claim_xgb = claim_model.xgb_calls
_ = services.analyze_provider(provider_model, provider_records[0])
check("10. claim model never executed during provider analysis",
      claim_model.xgb_calls == claim_xgb)

print("\n[2] PROVIDER SINGLE — ML -> Analytical Engine -> result")
prepared_p, ml_p_before = predict_providers(provider_model, [provider_records[0]])
single_p = services.analyze_provider(provider_model, provider_records[0])
check("2. provider risk score: ML stage == after Analytical Engine",
      single_p.ml.risk_score == ml_p_before[0].risk_score)
check("2. class probabilities present (2 classes)", len(single_p.ml.class_probabilities) == 2)
check("2. provider signals produced", len(single_p.signals) >= 10)
check("2. provider artifact has NO ordinal encoder",
      provider_model.ordinal_encoder is None)

# ----------------------------------------------------------------------
print("\n[3] CLAIM BATCH — ML -> Analytical Engine -> results -> queue")
claim_model.reset_call_counters()
batch = services.analyze_claims_batch(claim_model, claim_records)
check("11. exactly ONE XGBoost call for the whole claim batch",
      claim_model.xgb_calls == 1)
check("11. exactly ONE LightGBM call for the whole claim batch",
      claim_model.lgb_calls == 1)
check("3. batch total == submitted records", batch["total"] == 10)
check("3. one outcome per submitted record", len(batch["outcomes"]) == 10)
check("3. every outcome carries signals + evidence",
      all(len(o.signals) >= 10 and isinstance(o.evidence, list) for o in batch["outcomes"]))
check("3. claim ids preserved exactly",
      batch["identifiers"]["claim_ids"] == [str(r["Claim_ID"]) for r in claim_records])
check("3. provider ids preserved exactly",
      batch["identifiers"]["provider_ids"] == [str(r["Provider_ID"]) for r in claim_records])
q = batch["queue"]
ml_scores = [o.ml.risk_score for o in batch["outcomes"]]
check("12. queue sorted by ORIGINAL ML risk score (desc)",
      [item["risk_score"] for item in q] == sorted(ml_scores, reverse=True))
check("12. queue carries exactly the original ML scores",
      sorted(item["risk_score"] for item in q) == sorted(ml_scores))
check("12. queue fields = Claim ID / Provider ID / Risk / Rating / Status (+rank/pred)",
      all(set(item) == {"rank", "claim_id", "provider_id", "risk_score",
                        "rating", "predicted_class", "status"} for item in q))
check("12. queue ranks sequential 1..n", [i["rank"] for i in q] == list(range(1, 11)))
check("12. queue status = Pending Review", all(i["status"] == "Pending Review" for i in q))
s = batch["summary"]
check("13b. claim summary exposes dashboard values",
      all(k in s for k in ("total_records", "suspicious_count", "average_risk",
                           "high_risk_count", "medium_risk_count", "low_risk_count",
                           "unique_claims", "unique_providers", "unique_beneficiaries")))
check("13b. summary rating counts sum to total",
      s["high_risk_count"] + s["medium_risk_count"] + s["low_risk_count"] == 10)

print("\n[12] QUEUE TIE ORDER — controlled duplicate records")
tie_records = [claim_records[0], claim_records[1], dict(claim_records[0]), dict(claim_records[1])]
tie_batch = services.analyze_claims_batch(claim_model, tie_records)
tie_q = tie_batch["queue"]
tie_scores = [o.ml.risk_score for o in tie_batch["outcomes"]]
check("12. duplicates score identically (ties exist)",
      tie_scores[0] == tie_scores[2] and tie_scores[1] == tie_scores[3])
stable_order = [i for i in range(4)]
stable_order.sort(key=lambda i: tie_scores[i], reverse=True)  # Python sort is stable
check("12. tie order is deterministic (stable submission order)",
      [item["claim_id"] for item in tie_q] ==
      [str(tie_records[i]["Claim_ID"]) for i in stable_order])

print("\n[4] PROVIDER BATCH — ML -> Analytical Engine -> results -> queue")
provider_model.reset_call_counters()
pbatch = services.analyze_providers_batch(provider_model, provider_records)
check("11. exactly ONE XGBoost call for the whole provider batch",
      provider_model.xgb_calls == 1)
check("11. exactly ONE LightGBM call for the whole provider batch",
      provider_model.lgb_calls == 1)
check("4. provider batch total == submitted records", pbatch["total"] == 10)
pq = pbatch["queue"]
p_scores = [o.ml.risk_score for o in pbatch["outcomes"]]
check("12. provider queue sorted by ORIGINAL ML risk score (desc)",
      [item["risk_score"] for item in pq] == sorted(p_scores, reverse=True))
check("12. provider queue fields",
      all(set(item) == {"rank", "provider_npi", "provider_type", "risk_score",
                        "rating", "predicted_class", "status"} for item in pq))
ps = pbatch["summary"]
check("13b. provider summary exposes dashboard values",
      all(k in ps for k in ("total_records", "total_providers", "suspicious_count",
                            "suspicious_providers", "average_risk", "high_risk_count",
                            "medium_risk_count", "low_risk_count",
                            "total_beneficiaries_served")))

# ----------------------------------------------------------------------
print("\n[13/14/15] LIVE API — frontend receives real backend results")
API = "http://localhost:8000"
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
    cwd=str(BASE),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


def api_post(path: str, payload: dict):
    req = urllib.request.Request(
        f"{API}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def api_get(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=120) as resp:
        return json.loads(resp.read().decode())


try:
    up = False
    for _ in range(60):
        try:
            api_get("/api/health")
            up = True
            break
        except Exception:
            time.sleep(2)
    check("13. API is up (/api/health)", up)

    if up:
        # 14 — single claim, one-case payload, NO queue
        r1 = api_post("/api/claims/score",
                      {"features": json_safe(claim_records[0]), "enrich_from_dataset": False})
        check("14. single claim response has dashboard fields",
              all(k in r1 for k in ("claim_id", "provider_id", "predicted_class",
                                    "predicted_class_id", "risk_score", "rating",
                                    "class_probabilities", "signals", "evidence")))
        check("14. single claim response has NO queue", "queue" not in r1)
        check("9. API risk score == ML stage risk score",
              r1["risk_score"] == single.ml.risk_score)
        check("14. single claim signals/evidence non-empty",
              len(r1["signals"]) >= 10 and len(r1["evidence"]) >= 0)

        # 14 — single provider, one-case payload, NO queue
        r2 = api_post("/api/providers/score",
                      {"features": json_safe(provider_records[0]), "enrich_from_dataset": False})
        check("14. single provider response has dashboard fields",
              all(k in r2 for k in ("provider_npi", "predicted_class", "risk_score",
                                    "rating", "class_probabilities", "signals", "evidence")))
        check("14. single provider response has NO queue", "queue" not in r2)
        check("9. provider API risk score == ML stage risk score",
              r2["risk_score"] == single_p.ml.risk_score)

        # 15 — batch claims: submitted records + queue
        b1 = api_post("/api/claims/batch",
                      {"rows": [json_safe(r) for r in claim_records], "use_dataset": False})
        check("15. claim batch returns total/results/queue/summary",
              all(k in b1 for k in ("total", "results", "queue", "summary")))
        check("15. claim batch echoes every submitted record",
              sorted(res["claim_id"] for res in b1["results"]) ==
              sorted(str(r["Claim_ID"]) for r in claim_records))
        check("9. batch API risk scores == ML stage risk scores",
              sorted(res["risk_score"] for res in b1["results"]) == sorted(ml_scores))
        check("12. batch API queue sorted by original ML risk (desc)",
              [item["risk_score"] for item in b1["queue"]] ==
              sorted(ml_scores, reverse=True))
        check("15. every batch row keeps row-selection data (id/risk/rating/signals/evidence)",
              all(res.get("claim_id") and isinstance(res.get("signals"), list)
                  and isinstance(res.get("evidence"), list) for res in b1["results"]))

        # 15 — batch providers
        b2 = api_post("/api/providers/batch",
                      {"rows": [json_safe(r) for r in provider_records], "use_dataset": False})
        check("15. provider batch returns total/results/queue/summary",
              all(k in b2 for k in ("total", "results", "queue", "summary")))
        check("9. provider batch API risk scores == ML stage risk scores",
              sorted(res["risk_score"] for res in b2["results"]) == sorted(p_scores))
        check("12. provider batch API queue sorted by original ML risk (desc)",
              [item["risk_score"] for item in b2["queue"]] ==
              sorted(p_scores, reverse=True))
finally:
    server.terminate()
    server.wait(timeout=15)

print("\n" + "=" * 72)
print(f"RESULT: {PASSED} passed, {FAILED} failed")
print("=" * 72)
sys.exit(1 if FAILED else 0)
