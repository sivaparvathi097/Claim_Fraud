"""Phase 4 verification — REAL CSV UPLOAD -> ACTUAL ML INFERENCE.

Proves, against a live server, that the batch endpoints score EXACTLY the
uploaded rows (never a training-dataset sample):

  USER UPLOADS CSV -> READ ACTUAL FILE -> VALIDATE SCHEMA -> ACTUAL RECORDS
      -> VECTORISED ML INFERENCE -> ANALYTICAL ENGINE -> SIGNALS + EVIDENCE
      -> DASHBOARD RESPONSE -> RISK QUEUE

Covers: identifier preservation, vectorisation (1 XGB + 1 LGB call per
batch), Analytical Engine integrity (uploaded values in signals), score
immutability (ML risk == API risk == queue risk), data-leakage protection
(uploaded labels ignored), schema validation errors, legacy JSON path,
single-analysis regression, and LLM/human-review regression.

Run from BACKEND/:  python verify_phase4.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOST, PORT, BASE = "127.0.0.1", 8010, f"http://127.0.0.1:8010"
FIXTURES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_fixtures"))
CLAIMS_CSV = os.path.join(FIXTURES, "test_uploaded_claims.csv")
PROVIDERS_CSV = os.path.join(FIXTURES, "test_uploaded_providers.csv")
CLAIMS_DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "claimsfinal_with_target.csv"))
PROVIDERS_DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "Provider_with_potential_fraud.csv"))

PASS = FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def request(method: str, path: str, body: bytes | None, headers: dict[str, str]):
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    return value


def post_json(path: str, payload: dict):
    return request("POST", path, json.dumps(json_safe(payload)).encode(), {"Content-Type": "application/json"})


def upload(path: str, filename: str, raw: bytes, ctype: str = "text/csv"):
    boundary = "----phase4boundary925e99ce"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {ctype}\r\n\r\n".encode(),
            raw,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return request("POST", path, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})


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


def main() -> int:
    global FAIL
    env = {k: v for k, v in os.environ.items() if k != "FWA_LLM_API_KEY"}
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_phase4_server.log")
    log_file = open(log_path, "w")
    api_ok = False
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT)],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    try:
        if not wait_for_health(server):
            log_file.flush()
            print("FATAL: backend did not become healthy. Server log:")
            print(open(log_path, errors="replace").read()[-3000:])
            return 1

        uploaded_claims = pd.read_csv(CLAIMS_CSV)
        uploaded_providers = pd.read_csv(PROVIDERS_CSV)
        up_claim_ids = {str(v) for v in uploaded_claims["Claim_ID"]}
        up_provider_ids = {str(v) for v in uploaded_claims["Provider_ID"]}
        up_npis = {str(v) for v in uploaded_providers["provider_npi"]}

        # ================================================= 1. CLAIM CSV UPLOAD
        print("\n[1] POST /api/claims/batch — actual uploaded CSV (25 rows)")
        raw = open(CLAIMS_CSV, "rb").read()
        status, body = upload("/api/claims/batch", "test_uploaded_claims.csv", raw)
        check("HTTP 200", status == 200, f"got {status}: {str(body)[:200]}")
        results = body.get("results", [])
        queue = body.get("queue", [])
        summary = body.get("summary", {})
        check("total == uploaded rows (25)", body.get("total") == 25, str(body.get("total")))
        check("result row count == 25", len(results) == 25, str(len(results)))
        check("queue row count == 25", len(queue) == 25, str(len(queue)))

        ret_claim_ids = {str(r["claim_id"]) for r in results}
        ret_provider_ids = {str(r.get("provider_id")) for r in results if r.get("provider_id") is not None}
        check("returned Claim_IDs == uploaded Claim_IDs", ret_claim_ids == up_claim_ids,
              f"missing={sorted(up_claim_ids - ret_claim_ids)[:3]} extra={sorted(ret_claim_ids - up_claim_ids)[:3]}")
        check("returned Provider_IDs are a subset of uploaded Provider_IDs", ret_provider_ids <= up_provider_ids,
              str(sorted(ret_provider_ids - up_provider_ids)[:3]))
        check("every output row has prediction label", all(r.get("predicted_class") for r in results))
        check("every output row has class probabilities",
              all(isinstance(r.get("class_probabilities"), dict) and len(r["class_probabilities"]) == 4 for r in results))
        check("risk scores within [0,100]", all(0.0 <= r["risk_score"] <= 100.0 for r in results))
        check("every row carries deterministic signals", all(len(r.get("signals", [])) >= 1 for r in results))
        check("every row carries evidence", all(len(r.get("evidence", [])) >= 1 for r in results))
        check("no training-only IDs leak in", not (ret_claim_ids - up_claim_ids))

        q_risks = [q["risk_score"] for q in queue]
        check("queue sorted by ML risk desc", q_risks == sorted(q_risks, reverse=True))
        check("queue contains exactly the uploaded records",
              {str(q["claim_id"]) for q in queue} == up_claim_ids)
        check("queue ranks 1..25", [q["rank"] for q in queue] == list(range(1, 26)))
        check("queue default status Pending Review", all(q["status"] == "Pending Review" for q in queue))
        check("summary from uploaded rows (total_records=25, unique_claims=25)",
              summary.get("total_records") == 25 and summary.get("unique_claims") == 25)
        avg = round(sum(r["risk_score"] for r in results) / len(results), 2)
        check("summary average matches uploaded results", abs(summary.get("average_risk", -1) - avg) < 0.05,
              f"{summary.get('average_risk')} vs {avg}")
        claim_resp = body

        # ============================================ 2. PROVIDER CSV UPLOAD
        print("\n[2] POST /api/providers/batch — actual uploaded CSV (7 rows)")
        raw_p = open(PROVIDERS_CSV, "rb").read()
        status, body = upload("/api/providers/batch", "test_uploaded_providers.csv", raw_p)
        check("HTTP 200", status == 200, f"got {status}: {str(body)[:200]}")
        p_results = body.get("results", [])
        p_queue = body.get("queue", [])
        p_summary = body.get("summary", {})
        check("total == uploaded rows (7)", body.get("total") == 7, str(body.get("total")))
        check("result row count == 7", len(p_results) == 7, str(len(p_results)))
        check("queue row count == 7", len(p_queue) == 7, str(len(p_queue)))
        check("returned NPIs == uploaded NPIs",
              {str(r["provider_npi"]) for r in p_results} == up_npis)
        check("risk scores within [0,100]", all(0.0 <= r["risk_score"] <= 100.0 for r in p_results))
        check("every row carries signals + evidence",
              all(len(r.get("signals", [])) >= 1 and len(r.get("evidence", [])) >= 1 for r in p_results))
        pq_risks = [q["risk_score"] for q in p_queue]
        check("queue sorted by ML risk desc", pq_risks == sorted(pq_risks, reverse=True))
        check("queue contains exactly the uploaded providers",
              {str(q["provider_npi"]) for q in p_queue} == up_npis)
        check("queue default status Pending Review", all(q["status"] == "Pending Review" for q in p_queue))
        check("summary from uploaded rows (total_providers=7)", p_summary.get("total_providers") == 7)
        provider_resp = body

        # NOTE: sections [3] AE integrity/score immutability and [4] vectorisation
        # run OFFLINE after the server is stopped (loading the artifacts twice
        # in parallel exceeded available RAM).

        # ================================ 5. DATA-LEAKAGE PROTECTION
        print("\n[5] Data-leakage protection — uploaded labels must not influence inference")
        with_labels = pd.read_csv(CLAIMS_DATASET, nrows=25)  # keeps 'target'
        buf = io.StringIO(); with_labels.to_csv(buf, index=False)
        status, body = upload("/api/claims/batch", "exported_with_target.csv", buf.getvalue().encode())
        check("claim upload WITH target column still 200", status == 200, str(status))
        labelled_risk = {str(r["claim_id"]): r["risk_score"] for r in body.get("results", [])}
        api_risk_now = {str(r["claim_id"]): r["risk_score"] for r in results}
        check("identical risk scores with/without uploaded target",
              all(abs(labelled_risk[k] - api_risk_now[k]) < 1e-9 for k in api_risk_now))

        with_labels_p = pd.read_csv(PROVIDERS_DATASET, nrows=7)  # keeps 'Potential Fraud'
        buf = io.StringIO(); with_labels_p.to_csv(buf, index=False)
        status, body = upload("/api/providers/batch", "exported_with_label.csv", buf.getvalue().encode())
        check("provider upload WITH 'Potential Fraud' column still 200", status == 200, str(status))
        labelled_p_risk = {str(r["provider_npi"]): r["risk_score"] for r in body.get("results", [])}
        api_p_risk = {str(r["provider_npi"]): r["risk_score"] for r in p_results}
        check("identical risk scores with/without uploaded label",
              all(abs(labelled_p_risk[k] - api_p_risk[k]) < 1e-9 for k in api_p_risk))

        # ================================ 6. SCHEMA VALIDATION / ERROR PATHS
        print("\n[6] File + schema validation (clear 400s, no crashes)")
        status, body = upload("/api/claims/batch", "notes.txt", b"hello", "application/octet-stream")
        check("non-CSV extension rejected", status == 400 and "Unsupported file type" in body.get("detail", ""))
        status, body = upload("/api/claims/batch", "empty.csv", b"")
        check("empty CSV rejected", status == 400 and "empty" in body.get("detail", "").lower())
        status, body = upload("/api/claims/batch", "header_only.csv", b"Claim_ID,Claim_Type\n")
        check("header-only CSV rejected", status == 400 and "no data rows" in body.get("detail", ""))

        missing_col = uploaded_claims.drop(columns=["Claim_Type"])
        buf = io.StringIO(); missing_col.to_csv(buf, index=False)
        status, body = upload("/api/claims/batch", "missing_column.csv", buf.getvalue().encode())
        check("missing required column -> 400 naming it",
              status == 400 and "Claim_Type" in body.get("detail", ""))

        no_id = uploaded_claims.drop(columns=["Claim_ID"])
        buf = io.StringIO(); no_id.to_csv(buf, index=False)
        status, body = upload("/api/claims/batch", "no_identifier.csv", buf.getvalue().encode())
        check("missing identifier column -> 400 naming it",
              status == 400 and "Claim_ID" in body.get("detail", ""))

        dup = pd.concat([uploaded_claims.head(3), uploaded_claims.head(1)], ignore_index=True)
        buf = io.StringIO(); dup.to_csv(buf, index=False)
        status, body = upload("/api/claims/batch", "duplicate_ids.csv", buf.getvalue().encode())
        check("duplicate Claim_IDs -> 400", status == 400 and "Duplicate" in body.get("detail", ""))

        bad_num = uploaded_claims.copy(); bad_num["Claim_Submitted_Amount"] = "abc"
        buf = io.StringIO(); bad_num.to_csv(buf, index=False)
        status, body = upload("/api/claims/batch", "bad_numeric.csv", buf.getvalue().encode())
        check("non-numeric numeric column -> 400",
              status == 400 and "Claim_Submitted_Amount" in body.get("detail", ""))

        missing_p = uploaded_providers.drop(columns=["provider_type"])
        buf = io.StringIO(); missing_p.to_csv(buf, index=False)
        status, body = upload("/api/providers/batch", "provider_missing_column.csv", buf.getvalue().encode())
        check("provider missing required column -> 400 naming it",
              status == 400 and "provider_type" in body.get("detail", ""))

        # ====================================== 7. LEGACY JSON PATH REGRESSION
        print("\n[7] Legacy JSON batch path still functional")
        status, body = post_json("/api/claims/batch", {"use_dataset": True, "limit": 12})
        check("JSON claims batch -> 200 with 12 rows", status == 200 and body.get("total") == 12, str(status))
        status, body = post_json("/api/providers/batch", {"use_dataset": True, "limit": 8})
        check("JSON providers batch -> 200 with 8 rows", status == 200 and body.get("total") == 8, str(status))

        # ============================ 8. SINGLE ANALYSIS + LLM/REVIEW REGRESSION
        print("\n[8] Single-analysis + LLM/human-review regression")
        first = {k: v for k, v in uploaded_claims.iloc[0].to_dict().items()
                 if not (isinstance(v, float) and pd.isna(v))}
        status, body = post_json("/api/claims/score", {
            "claim_id": str(first["Claim_ID"]), "enrich_from_dataset": False,
            "features": first,
        })
        check("single claim still scores (200 + risk)", status == 200 and 0 <= body.get("risk_score", -1) <= 100, str(status))
        single_claim = body

        first_p = {k: v for k, v in uploaded_providers.iloc[0].to_dict().items()
                   if not (isinstance(v, float) and pd.isna(v))}
        status, body = post_json("/api/providers/score", {
            "provider_npi": str(first_p["provider_npi"]), "enrich_from_dataset": False,
            "features": first_p,
        })
        check("single provider still scores (200 + risk)", status == 200 and 0 <= body.get("risk_score", -1) <= 100, str(status))

        status, body = post_json("/api/claims/explain", {"analysis": claim_resp["results"][0]})
        check("explain without key -> 503 configuration error",
              status == 503 and "FWA_LLM_API_KEY" in body.get("detail", ""))
        status, body = post_json("/api/review/accept",
                                 {"case_id": str(claim_resp["results"][0]["claim_id"]), "entity_type": "claim"})
        check("human review accept still works", status == 200 and body.get("review_status") == "Accepted")
        status, body = request("GET", "/api/llm/status", None, {})
        check("LLM status reports configured=false", status == 200 and body.get("configured") is False)

        # Review must not alter ML output: re-score the same single claim.
        status, body = post_json("/api/claims/score", {
            "claim_id": str(first["Claim_ID"]), "enrich_from_dataset": False,
            "features": first,
        })
        check("review/explain never altered ML output",
              status == 200 and abs(body.get("risk_score") - single_claim.get("risk_score")) < 1e-9)
        api_ok = True

    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()

    if not api_ok:
        print(f"\nRESULT: {PASS} passed, {FAIL} failed (offline sections skipped)")
        return 1

    # ===================== 3. AE INTEGRITY + SCORE IMMUTABILITY (offline)
    print("\n[3] Analytical Engine integrity + score immutability (offline ML comparison)")
    from app.main import registry  # loads the locked artifacts in-process, server is down
    from app import uploads as uploads_mod
    from analytical_engine.inference import predict_claims, predict_providers

    claim_records = uploads_mod.parse_claim_upload(raw, "test_uploaded_claims.csv", "text/csv", registry.claim)
    check("parser returns the actual uploaded rows", len(claim_records) == 25)
    ml_frame, ml_results = predict_claims(registry.claim, claim_records)
    offline_risk = {str(claim_records[i]["Claim_ID"]): ml_results[i].risk_score for i in range(25)}
    api_risk = {str(r["claim_id"]): r["risk_score"] for r in results}
    queue_risk = {str(q["claim_id"]): q["risk_score"] for q in queue}
    check("ML risk == API risk for every uploaded claim",
          all(abs(offline_risk[k] - api_risk[k]) < 1e-9 for k in offline_risk))
    check("ML risk == queue risk for every uploaded claim",
          all(abs(offline_risk[k] - queue_risk[k]) < 1e-9 for k in offline_risk))

    # Signals must reflect the UPLOADED values. Calculators round display
    # values to 4 decimals, so the tolerance covers that rounding exactly.
    by_id = {str(claim_records[i]["Claim_ID"]): claim_records[i] for i in range(25)}
    matched = mismatches = 0
    for r in results:
        src = by_id[str(r["claim_id"])]
        for sig in r["signals"]:
            for attr, val in (sig.get("attribute_values") or {}).items():
                if attr in src and isinstance(src[attr], (int, float)):
                    tol = 5e-5 + 1e-9 * abs(float(src[attr]))
                    if abs(float(src[attr]) - float(val)) <= tol:
                        matched += 1
                    else:
                        mismatches += 1
    check("signal attribute values match uploaded values", matched > 0 and mismatches == 0,
          f"matched={matched} mismatched={mismatches}")

    # ============================ 4. VECTORISATION (1+1 calls per batch)
    print("\n[4] Vectorised batch inference — 1 XGBoost + 1 LightGBM call per batch")
    registry.claim.reset_call_counters()
    predict_claims(registry.claim, claim_records)
    check("claims: XGBoost predict_proba called exactly once", registry.claim.xgb_calls == 1, str(registry.claim.xgb_calls))
    check("claims: LightGBM predict_proba called exactly once", registry.claim.lgb_calls == 1, str(registry.claim.lgb_calls))

    prov_records = uploads_mod.parse_provider_upload(raw_p, "test_uploaded_providers.csv", "text/csv", registry.provider)
    registry.provider.reset_call_counters()
    _, prov_ml = predict_providers(registry.provider, prov_records)
    check("providers: XGBoost predict_proba called exactly once", registry.provider.xgb_calls == 1, str(registry.provider.xgb_calls))
    check("providers: LightGBM predict_proba called exactly once", registry.provider.lgb_calls == 1, str(registry.provider.lgb_calls))
    prov_risk = {str(prov_records[i]["provider_npi"]): prov_ml[i].risk_score for i in range(len(prov_ml))}
    check("ML risk == API risk for every uploaded provider",
          all(abs(prov_risk[str(r['provider_npi'])] - r["risk_score"]) < 1e-9 for r in p_results))

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
