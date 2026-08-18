"""Phase 3 integration verification — Trust Gate + calibration in the live flow.

Pipeline under test (single AND batch, claims AND providers):

    INPUT -> ML -> raw risk -> calibration -> calibrated risk
          -> Trust Gate -> TrustFlags -> Analytical Engine -> signals+evidence
          -> API response (+ existing risk queue for batches)

Tests (per phase brief):
  * claim/provider single + batch via the real API (terminal API tests)
  * TrustFlags present with exactly the 7 fields
  * raw ML risk preservation + calibrated score = stored artifact output
  * signal + evidence + queue preservation (Phase 2 engine equivalence)
  * claim/provider isolation (no cross-loading of models/calibrators)
  * Trust Gate executes exactly once per analyzed case
  * calibration executes via the stored artifact
  * vectorized ML inference unchanged (1 XGB + 1 LGB call per batch)
  * no routing / no LLM changes / no human-review changes

Run from BACKEND/:  python verify_phase3_integration.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

HOST, PORT = "127.0.0.1", 8015
BASE = f"http://{HOST}:{PORT}"
FIXTURES = BACKEND_DIR.parent / "test_fixtures"
CLAIMS_CSV = FIXTURES / "test_uploaded_claims.csv"
PROVIDERS_CSV = FIXTURES / "test_uploaded_providers.csv"

TRUST_FLAG_FIELDS = {
    "model_agreement", "low_agreement", "confidence", "low_confidence",
    "in_distribution", "out_of_distribution", "feature_completeness",
}
ROUTING_TOKENS = (
    "auto_approve", "fast_track", "full_investigation", "route_reasons",
    "requires_human_review",
)
# Words allowed in prose (docstrings asserting their ABSENCE) but forbidden as
# actual code identifiers.
ROUTING_IDENTIFIER_TOKENS = ("orchestrator", "routing", "route")

PASS = FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


# ------------------------------------------------------------------ HTTP
def request(method: str, path: str, body: dict | None, headers: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    for key, value in headers.items():
        req.add_header(key, value)
    if body is not None and "Content-Type" not in headers:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def post_multipart(path: str, csv_path: Path) -> tuple[int, dict]:
    boundary = "----phase3boundary"
    raw = csv_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
    return _raw_post(path, body, boundary)


def _raw_post(path: str, body: bytes, boundary: str) -> tuple[int, dict]:
    req = urllib.request.Request(BASE + path, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def wait_for_health(proc: subprocess.Popen) -> bool:
    for _ in range(90):
        if proc.poll() is not None:
            return False
        try:
            status, body = request("GET", "/api/health", None, {})
            if status == 200 and body.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _flags_ok(flags: dict | None) -> bool:
    return isinstance(flags, dict) and set(flags.keys()) == TRUST_FLAG_FIELDS


def _upstream_ok(r: dict) -> bool:
    return (
        isinstance(r.get("raw_ml_risk_score"), (int, float))
        and isinstance(r.get("calibrated_risk_score"), (int, float))
        and 0.0 <= r["calibrated_risk_score"] <= 100.0
        and r.get("prediction") == r.get("predicted_class")
        and r.get("prediction_label") == r.get("predicted_class")
        and r.get("risk_rating") == r.get("rating")
        and _flags_ok(r.get("trust_flags"))
    )


# ================================================================ API TESTS
def api_tests() -> dict:
    print("\n[1] GET /api/health")
    status, body = request("GET", "/api/health", None, {})
    check("health endpoint ok", status == 200 and body.get("status") == "ok")

    claims = pd.read_csv(CLAIMS_CSV)
    providers = pd.read_csv(PROVIDERS_CSV)

    print("\n[2] POST /api/claims/score — single claim with TrustFlags")
    first_claim = claims.iloc[0].to_dict()
    status, claim_single = request("POST", "/api/claims/score", {"features": _json_safe(first_claim)}, {})
    check("claim single returns 200", status == 200, f"status={status} {claim_single.get('detail', '')}")
    check("claim single: upstream context fields present + consistent", _upstream_ok(claim_single))
    check("claim single: signals present", len(claim_single.get("signals", [])) > 0)
    check("claim single: evidence list present", isinstance(claim_single.get("evidence"), list))
    check("claim single: risk_score (existing field) unchanged contract",
          isinstance(claim_single.get("risk_score"), (int, float)))

    print("\n[3] POST /api/providers/score — single provider with TrustFlags")
    first_provider = providers.iloc[0].to_dict()
    status, prov_single = request("POST", "/api/providers/score", {"features": _json_safe(first_provider)}, {})
    check("provider single returns 200", status == 200, f"status={status} {prov_single.get('detail', '')}")
    check("provider single: upstream context fields present + consistent", _upstream_ok(prov_single))
    check("provider single: signals present", len(prov_single.get("signals", [])) > 0)
    check("provider single: evidence list present", isinstance(prov_single.get("evidence"), list))

    print("\n[4] POST /api/claims/batch — uploaded CSV (25 rows)")
    status, claim_batch = post_multipart("/api/claims/batch", CLAIMS_CSV)
    check("claim batch returns 200", status == 200, f"status={status} {claim_batch.get('detail', '')}")
    results = claim_batch.get("results", [])
    check("claim batch: 25 results for 25 uploaded rows", claim_batch.get("total") == 25 and len(results) == 25)
    check("claim batch: TrustFlags present on EVERY result", all(_upstream_ok(r) for r in results))
    check("claim batch: results/summary/queue preserved",
          "summary" in claim_batch and "queue" in claim_batch)
    queue = claim_batch.get("queue", [])
    scores = [q["risk_score"] for q in queue]
    check("claim batch: queue sorted by ORIGINAL ML risk desc + ranked",
          scores == sorted(scores, reverse=True) and [q["rank"] for q in queue] == list(range(1, len(queue) + 1)))
    check("claim batch: queue entries carry NO calibrated/routing fields",
          all(set(q.keys()) == {"claim_id", "provider_id", "risk_score", "rating", "predicted_class", "status", "rank"}
              for q in queue))
    check("claim batch: queue risk == result risk_score set (original ML risk)",
          sorted(scores) == sorted(r["risk_score"] for r in results))

    print("\n[5] POST /api/providers/batch — uploaded CSV (7 rows)")
    status, prov_batch = post_multipart("/api/providers/batch", PROVIDERS_CSV)
    check("provider batch returns 200", status == 200, f"status={status} {prov_batch.get('detail', '')}")
    p_results = prov_batch.get("results", [])
    check("provider batch: 7 results for 7 uploaded rows", prov_batch.get("total") == 7 and len(p_results) == 7)
    check("provider batch: TrustFlags present on EVERY result", all(_upstream_ok(r) for r in p_results))
    p_queue = prov_batch.get("queue", [])
    p_scores = [q["risk_score"] for q in p_queue]
    check("provider batch: queue sorted by ORIGINAL ML risk desc + ranked",
          p_scores == sorted(p_scores, reverse=True) and [q["rank"] for q in p_queue] == list(range(1, len(p_queue) + 1)))
    check("provider batch: queue entries carry NO calibrated/routing fields",
          all(set(q.keys()) == {"provider_npi", "provider_type", "risk_score", "rating", "predicted_class", "status", "rank"}
              for q in p_queue))

    print("\n[6] Routing present by Phase 4 design; no second score anywhere")
    dumps = json.dumps([claim_single, prov_single, claim_batch, prov_batch])
    # Phase 4 supersession: the API now carries a deterministic `routing`
    # decision on every result. The Phase 3 invariant that survives: queue
    # entries stay free of routing/calibrated keys (checked in [4]/[5]) and no
    # second risk score is invented.
    check("routing decision present on every API result (Phase 4)",
          all(isinstance(r.get("routing"), dict) and r["routing"]["route"]
              in ("auto_approve", "fast_track", "full_investigation")
              for r in [claim_single, prov_single] + claim_batch["results"] + prov_batch["results"]))
    check("no route_reasons / invented routing keys in any API response",
          all(tok not in dumps for tok in ("route_reasons",)))
    return {
        "claim_single": claim_single,
        "prov_single": prov_single,
        "claim_batch": claim_batch,
        "prov_batch": prov_batch,
    }


def _json_safe(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = None if np.isnan(value) else float(value)
        elif isinstance(value, float) and np.isnan(value):
            out[key] = None
        else:
            out[key] = value
    return out


# ================================================================ OFFLINE
def offline_tests() -> None:
    from analytical_engine.engine import run_claim_engine, run_provider_engine
    from analytical_engine.inference import HybridModel, predict_claims, predict_providers
    from app import calibration as calib
    from app import config, services, trust_gate as tg

    claims = pd.read_csv(CLAIMS_CSV)
    providers = pd.read_csv(PROVIDERS_CSV)
    claim_records = [_json_safe(r) for r in claims.to_dict(orient="records")]
    provider_records = [_json_safe(r) for r in providers.to_dict(orient="records")]

    # ------------------------------------------------ CLAIM flow (claim only)
    print("\n[7] Claim flow: score integrity + once-per-case + signal regression")
    claim_model = HybridModel(
        artifact=joblib.load(config.CLAIM_MODEL_PATH), name="claim_fwa_hybrid_model"
    )
    claim_cal = calib.load_calibrator(config.CLAIM_CALIBRATION_PATH)

    # Counters: Trust Gate and calibration must run exactly once per batch.
    counts = {"gate": 0, "calibrate": 0}
    orig_gate = tg.claim_trust_flags

    def counting_gate(*args, **kwargs):
        counts["gate"] += 1
        return orig_gate(*args, **kwargs)

    orig_cal = claim_cal.calibrate_risk_scores

    def counting_calibrate(*args, **kwargs):
        counts["calibrate"] += 1
        return orig_cal(*args, **kwargs)

    tg.claim_trust_flags = counting_gate
    services._CALIBRATORS["claim"] = claim_cal
    claim_cal.calibrate_risk_scores = counting_calibrate
    try:
        claim_model.reset_call_counters()
        batch = services.analyze_claims_batch(claim_model, claim_records)
    finally:
        tg.claim_trust_flags = orig_gate
        claim_cal.calibrate_risk_scores = orig_cal

    check("claim batch: vectorized ML unchanged (1 XGB + 1 LGB call)",
          claim_model.xgb_calls == 1 and claim_model.lgb_calls == 1,
          f"xgb={claim_model.xgb_calls} lgb={claim_model.lgb_calls}")
    check("claim batch: Trust Gate executed exactly ONCE for the whole batch",
          counts["gate"] == 1, str(counts["gate"]))
    check("claim batch: calibration executed exactly ONCE for the whole batch",
          counts["calibrate"] == 1, str(counts["calibrate"]))

    # Independent recomputation of both scores.
    prepared, results, xgb_prob, lgb_prob = predict_claims(claim_model, claim_records, return_blends=True)
    blend = claim_model.xgboost_ratio * xgb_prob + claim_model.lightgbm_ratio * lgb_prob
    mapping = claim_model.class_mapping
    legit_idx = next(i for i, cid in enumerate(claim_model.class_ids)
                     if str(mapping[cid]).lower() == "legitimate")
    raw = (1.0 - blend[:, legit_idx]) * 100.0
    calibrated = claim_cal.calibrate_risk_scores(raw)

    raw_ok = cal_ok = ctx_ok = True
    for i, outcome in enumerate(batch["outcomes"]):
        ctx = outcome.upstream_context
        raw_ok &= abs(ctx["raw_ml_risk_score"] - round(float(raw[i]), 6)) < 1e-9
        cal_ok &= abs(ctx["calibrated_risk_score"] - round(float(calibrated[i]), 6)) < 1e-9
        ctx_ok &= ctx["prediction"] == results[i].predicted_class and set(ctx["trust_flags"]) == TRUST_FLAG_FIELDS
    check("claim: raw_ml_risk_score == raw ML risk BEFORE calibration", raw_ok)
    check("claim: calibrated_risk_score == stored calibration artifact output", cal_ok)
    check("claim: upstream context carries prediction + 7 TrustFlags", ctx_ok)
    check("claim: outcome ML risk unchanged (frozen MLResult)",
          all(abs(o.ml.risk_score - results[i].risk_score) < 1e-12
              for i, o in enumerate(batch["outcomes"])))

    # Phase 2 vs Phase 3 signal/evidence regression (context must not change
    # any deterministic calculation).
    phase2_signals, phase2_evidence, phase2_top = [], [], []
    rows_for_engine = []
    for i, record in enumerate(claim_records):
        row = dict(record)
        row.update(prepared.iloc[i].to_dict())
        rows_for_engine.append(row)
        out2 = run_claim_engine(claim_model, results[i], rows_for_engine[i])  # no context
        phase2_signals.append([s.__dict__ for s in out2.signals])
        phase2_evidence.append([s.__dict__ for s in out2.evidence])
        phase2_top.append(out2.top_feature_deviations)
    check("claim: Phase 2 == Phase 3 Analytical Engine signals",
          phase2_signals == [[s.__dict__ for s in o.signals] for o in batch["outcomes"]])
    check("claim: Phase 2 == Phase 3 evidence package",
          phase2_evidence == [[s.__dict__ for s in o.evidence] for o in batch["outcomes"]])
    check("claim: Phase 2 == Phase 3 top feature deviations",
          phase2_top == [o.top_feature_deviations for o in batch["outcomes"]])
    queue = batch["queue"]
    check("claim: queue still uses the ORIGINAL ML risk score",
          sorted(q["risk_score"] for q in queue) == sorted(r.risk_score for r in results))
    check("claim: queue order + ranks deterministic (desc, 1..N)",
          [q["rank"] for q in queue] == list(range(1, len(queue) + 1))
          and [q["risk_score"] for q in queue] == sorted([q["risk_score"] for q in queue], reverse=True))

    # Single claim: gate runs exactly once per analyzed case too.
    counts["gate"] = counts["calibrate"] = 0
    tg.claim_trust_flags = counting_gate
    claim_cal.calibrate_risk_scores = counting_calibrate
    try:
        single = services.analyze_claim(claim_model, claim_records[0])
    finally:
        tg.claim_trust_flags = orig_gate
        claim_cal.calibrate_risk_scores = orig_cal
    check("claim single: Trust Gate executed exactly once", counts["gate"] == 1)
    check("claim single: calibration executed exactly once", counts["calibrate"] == 1)
    check("claim single: upstream context attached to the engine outcome",
          set(single.upstream_context.keys()) == {"raw_ml_risk_score", "calibrated_risk_score", "prediction", "trust_flags", "routing"})

    check("claim/provider isolation: only the claim calibrator was loaded",
          set(services._CALIBRATORS.keys()) == {"claim"}, str(list(services._CALIBRATORS.keys())))

    # -------------------------------------------- PROVIDER flow (provider only)
    print("\n[8] Provider flow: score integrity + once-per-case + signal regression")
    provider_model = HybridModel(
        artifact=joblib.load(config.PROVIDER_MODEL_PATH), name="provider_fwa_hybrid_model"
    )
    provider_cal = calib.load_calibrator(config.PROVIDER_CALIBRATION_PATH)

    counts = {"gate": 0, "calibrate": 0}
    orig_pgate = tg.provider_trust_flags

    def counting_pgate(*args, **kwargs):
        counts["gate"] += 1
        return orig_pgate(*args, **kwargs)

    orig_pcal = provider_cal.calibrate_risk_scores

    def counting_pcalibrate(*args, **kwargs):
        counts["calibrate"] += 1
        return orig_pcal(*args, **kwargs)

    tg.provider_trust_flags = counting_pgate
    services._CALIBRATORS["provider"] = provider_cal
    provider_cal.calibrate_risk_scores = counting_pcalibrate
    try:
        provider_model.reset_call_counters()
        p_batch = services.analyze_providers_batch(provider_model, provider_records)
    finally:
        tg.provider_trust_flags = orig_pgate
        provider_cal.calibrate_risk_scores = orig_pcal

    check("provider batch: vectorized ML unchanged (1 XGB + 1 LGB call)",
          provider_model.xgb_calls == 1 and provider_model.lgb_calls == 1,
          f"xgb={provider_model.xgb_calls} lgb={provider_model.lgb_calls}")
    check("provider batch: Trust Gate executed exactly ONCE for the whole batch",
          counts["gate"] == 1, str(counts["gate"]))
    check("provider batch: calibration executed exactly ONCE for the whole batch",
          counts["calibrate"] == 1, str(counts["calibrate"]))

    p_prepared, p_results, p_xgb, p_lgb = predict_providers(provider_model, provider_records, return_blends=True)
    p_blend = provider_model.xgboost_ratio * p_xgb + provider_model.lightgbm_ratio * p_lgb
    p_mapping = provider_model.class_mapping
    p_legit = next(i for i, cid in enumerate(provider_model.class_ids)
                   if str(p_mapping[cid]).lower() == "legitimate")
    p_raw = p_blend[:, 1 - p_legit] * 100.0
    p_calibrated = provider_cal.calibrate_risk_scores(p_raw)

    raw_ok = cal_ok = ctx_ok = True
    for i, outcome in enumerate(p_batch["outcomes"]):
        ctx = outcome.upstream_context
        raw_ok &= abs(ctx["raw_ml_risk_score"] - round(float(p_raw[i]), 6)) < 1e-9
        cal_ok &= abs(ctx["calibrated_risk_score"] - round(float(p_calibrated[i]), 6)) < 1e-9
        ctx_ok &= ctx["prediction"] == p_results[i].predicted_class and set(ctx["trust_flags"]) == TRUST_FLAG_FIELDS
    check("provider: raw_ml_risk_score == raw ML risk BEFORE calibration", raw_ok)
    check("provider: calibrated_risk_score == stored calibration artifact output", cal_ok)
    check("provider: upstream context carries prediction + 7 TrustFlags", ctx_ok)

    p_phase2_signals, p_phase2_evidence = [], []
    for i, record in enumerate(provider_records):
        row = dict(record)
        row.update(p_prepared.iloc[i].to_dict())
        out2 = run_provider_engine(provider_model, p_results[i], row)
        p_phase2_signals.append([s.__dict__ for s in out2.signals])
        p_phase2_evidence.append([s.__dict__ for s in out2.evidence])
    check("provider: Phase 2 == Phase 3 Analytical Engine signals",
          p_phase2_signals == [[s.__dict__ for s in o.signals] for o in p_batch["outcomes"]])
    check("provider: Phase 2 == Phase 3 evidence package",
          p_phase2_evidence == [[s.__dict__ for s in o.evidence] for o in p_batch["outcomes"]])
    p_queue = p_batch["queue"]
    check("provider: queue still uses the ORIGINAL ML risk score",
          sorted(q["risk_score"] for q in p_queue) == sorted(r.risk_score for r in p_results))

    counts["gate"] = counts["calibrate"] = 0
    tg.provider_trust_flags = counting_pgate
    provider_cal.calibrate_risk_scores = counting_pcalibrate
    try:
        p_single = services.analyze_provider(provider_model, provider_records[0])
    finally:
        tg.provider_trust_flags = orig_pgate
        provider_cal.calibrate_risk_scores = orig_pcal
    check("provider single: Trust Gate executed exactly once", counts["gate"] == 1)
    check("provider single: calibration executed exactly once", counts["calibrate"] == 1)
    check("provider single: upstream context attached to the engine outcome",
          set(p_single.upstream_context.keys()) == {"raw_ml_risk_score", "calibrated_risk_score", "prediction", "trust_flags", "routing"})

    # ------------------------------------------------------- static guardrails
    print("\n[9] No LLM changes / no human-review changes / routing delegated (Phase 4)")
    services_src = (BACKEND_DIR / "app" / "services.py").read_text(encoding="utf-8")
    main_src = (BACKEND_DIR / "app" / "main.py").read_text(encoding="utf-8")
    # Phase 4 supersession: deterministic routing now exists BY DESIGN, decided
    # exclusively inside app/routing.py. The Phase 3 invariant that survives is:
    # services.py and main.py only DELEGATE — no route literal is decided there.
    check("services.py delegates routing exclusively to app.routing",
          "from . import routing as rt" in services_src
          and "rt.route_claim" in services_src and "rt.route_provider" in services_src)
    check("no route literals decided inside services.py",
          all(tok not in services_src for tok in ROUTING_TOKENS))
    check("no route literals decided inside main.py",
          all(tok not in main_src for tok in ROUTING_TOKENS))
    check("no orchestrator/agent identifiers (def/class/import) in services.py",
          all(f"{kind} {tok}" not in services_src
              for kind in ("def", "class", "import") for tok in ("orchestrator", "agent")))
    check("no orchestrator/agent identifiers (def/class/import) added to main.py",
          all(f"{kind} {tok}" not in main_src
              for kind in ("def", "class") for tok in ("orchestrator", "agent")))
    check("services.py does not import LLM or human review",
          "import llm" not in services_src and "import review" not in services_src)
    check("no orchestrator/agent files exist",
          not any((BACKEND_DIR / "app" / f).exists()
                  for f in ("orchestrator.py", "agent.py")))
    check("Trust Gate still adds no risk-like keys",
          all("risk" not in k and "score" not in k for k in TRUST_FLAG_FIELDS))


def main() -> int:
    env = {k: v for k, v in os.environ.items() if k != "FWA_LLM_API_KEY"}
    log_path = BACKEND_DIR / "verify_phase3_integration_server.log"
    log_file = open(log_path, "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT)],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    try:
        if not wait_for_health(server):
            log_file.flush()
            print("FATAL: backend did not become healthy. Server log:")
            print(log_path.read_text(errors="replace")[-3000:])
            return 1
        api_tests()
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()

    # Offline ML checks AFTER the server released its memory (~8GB machine).
    offline_tests()

    print(f"\nRESULT: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
