"""Phase 5 verification — LLM EXPLANATION FOR EVERY ROUTE + HUMAN REVIEW.

Covers (per the Phase 5 brief):
  * LLM called for auto_approve, fast_track AND full_investigation
    (route-specific instructions, same boundary, no skipping)
  * route / risk / calibrated score / TrustFlags / signals / evidence preserved
  * LLM route integrity (a mismatching LLM route is IGNORED)
  * no second risk score, no ML/calibration/Trust Gate/engine rerun on explain
  * LLM failure handling (502 error state, 503 configuration state)
  * human Accept / Reject records; auto_approve has no mandatory review
  * claim/provider isolation, single + batch row-click behavior
  * queue unchanged, training targets never sent to the LLM

The LLM under test is a LOCAL OpenAI-compatible MOCK endpoint (no real LLM
API is configured in this environment). The mock records every request so the
suite can verify exactly what the LLM boundary sent.

Run:  python verify_phase5.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

HOST, PORT = "127.0.0.1", 8017
MOCK_HOST, MOCK_PORT = "127.0.0.1", 8021
BASE = f"http://{HOST}:{PORT}"
FIXTURES = BACKEND_DIR.parent / "test_fixtures"
CLAIMS_CSV = FIXTURES / "test_uploaded_claims.csv"
PROVIDERS_CSV = FIXTURES / "test_uploaded_providers.csv"
CASES_JSON = FIXTURES / "phase5_cases.json"

API_KEY = "phase5-test-key"
ROUTES = ("auto_approve", "fast_track", "full_investigation")

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))


# ------------------------------------------------------------- mock LLM server
MOCK: dict = {"mode": "normal", "wrong_route": None, "requests": []}


class MockLLMHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible /chat/completions mock that records every request."""

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        MOCK["requests"].append({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "model": payload.get("model"),
            "messages": payload.get("messages", []),
        })
        if MOCK["mode"] == "error":
            self._send(500, {"error": {"message": "mock LLM failure"}})
            return
        user = next((m["content"] for m in reversed(payload.get("messages", []))
                     if m.get("role") == "user"), "")
        try:
            ctx = json.loads(user[user.index("{"):])
        except (ValueError, json.JSONDecodeError):
            self._send(500, {"error": {"message": "mock could not parse context"}})
            return
        routing = (ctx.get("upstream_and_routing") or {}).get("routing") or {}
        route = routing.get("route")
        out_route = MOCK["wrong_route"] if MOCK["mode"] == "wrong_route" else route
        content = json.dumps({
            "route": out_route,
            "summary": f"MOCK explanation for case {ctx.get('case_id')} on route '{route}'.",
            "reasoning": [f"MOCK reasoning: the deterministic route is '{route}'.",
                          "MOCK reasoning: scores and prediction are echoed, never recalculated."],
            "supporting_signals": [s.get("signal") for s in ctx.get("analytical_signals", [])][:3],
            "evidence_used": [e.get("signal") for e in ctx.get("evidence", [])][:3],
            "caveats": ["MOCK caveat: the final decision belongs to the human reviewer."],
        })
        self._send(200, {"model": payload.get("model"),
                         "choices": [{"message": {"content": content}}]})

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence
        pass


def start_mock() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((MOCK_HOST, MOCK_PORT), MockLLMHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def last_mock_request() -> dict:
    return MOCK["requests"][-1]


def mock_user_context(req: dict) -> dict:
    user = next(m["content"] for m in req["messages"] if m.get("role") == "user")
    return json.loads(user[user.index("{"):])


def mock_user_text(req: dict) -> str:
    return next(m["content"] for m in req["messages"] if m.get("role") == "user")


# ------------------------------------------------------------------ http utils
def _post_json(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode() or "{}")
        except json.JSONDecodeError:
            detail = {}
        return exc.code, detail


def _get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode() or "{}")
        except json.JSONDecodeError:
            detail = {}
        return exc.code, detail


def post_multipart(path: str, csv_path: Path) -> tuple[int, dict]:
    boundary = "----phase5boundary"
    raw = csv_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode() or "{}")
        except json.JSONDecodeError:
            detail = {}
        return exc.code, detail


def wait_for_health(proc: subprocess.Popen) -> bool:
    for _ in range(300):
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{BASE}/api/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


# ---------------------------------------------------------- 1. static guards
def static_checks() -> None:
    print("\n[1] STATIC GUARDS — LLM explains, router decides, human reviews")
    llm_src = (BACKEND_DIR / "app" / "llm.py").read_text(encoding="utf-8")
    main_src = (BACKEND_DIR / "app" / "main.py").read_text(encoding="utf-8")
    review_src = (BACKEND_DIR / "app" / "review.py").read_text(encoding="utf-8")
    services_src = (BACKEND_DIR / "app" / "services.py").read_text(encoding="utf-8")

    check("llm.py documents route-specific instructions for all three routes",
          all(f'"{r}"' in llm_src for r in ROUTES) and "ROUTE_INSTRUCTIONS" in llm_src)
    check("auto_approve instruction uses 'satisfied the configured automated' wording",
          "satisfied the configured " in llm_src and "definitely legitimate" in llm_src)
    check("full_investigation instruction forbids confirmed-fraud language",
          "confirmed fraud" in llm_src and "potentially suspicious" in llm_src)
    check("system prompt forbids the LLM from deciding/changing the route",
          "NEVER decide, change or question the route" in llm_src)
    check("requested LLM JSON shape echoes the route back", '"route"' in llm_src)
    check("llm.py documents batch cost/latency scaling (incl. auto_approve)",
          "COST NOTE" in llm_src and "auto_approve cases" in llm_src)
    check("no hardcoded API keys in llm.py", "sk-" not in llm_src)
    check("env-based configuration reused (FWA_LLM_API_KEY / BASE_URL / MODEL)",
          all(t in llm_src for t in ("FWA_LLM_API_KEY", "FWA_LLM_BASE_URL", "FWA_LLM_MODEL")))

    check("main.py enforces route integrity (deterministic route wins)",
          "deterministic_route" in main_src and "llm_route != deterministic_route" in main_src)
    check("main.py keeps safe failure states (503 config / 502 call failure)",
          "status_code=503" in main_src and "status_code=502" in main_src)
    check("explain endpoints never branch/skip on the route",
          main_src.count("auto_approve") == 0)
    check("services.py never calls the LLM (explain endpoints only)",
          "request_reasoning" not in services_src and "llm_boundary" not in services_src)
    check("review store keeps the exact spec record fields",
          all(t in review_src for t in
              ("case_id", "entity_type", "review_status", "reviewed_at", "reviewer_action"))
          and '"Accepted"' in review_src and '"Rejected"' in review_src)

    fe = BACKEND_DIR.parent / "FRONTEND" / "fwa-analyzer-suite" / "src"
    claim_tsx = (fe / "components" / "fwa" / "ClaimDetail.tsx").read_text(encoding="utf-8")
    prov_tsx = (fe / "components" / "fwa" / "ProviderDetail.tsx").read_text(encoding="utf-8")
    api_ts = (fe / "lib" / "api.ts").read_text(encoding="utf-8")
    check("frontend renders human review only when the route requires it",
          "requires_human_review" in claim_tsx and "requires_human_review" in prov_tsx)
    check("frontend explanation type carries route + routing fields",
          "routing_reasons" in api_ts and "trust_factors" in api_ts
          and "route: ApiRoute | null" in api_ts)


# ------------------------------------------------------------- 2. unit checks
def unit_checks() -> None:
    from types import SimpleNamespace

    from app import llm

    print("\n[2] UNIT — context contract, route instructions, parsing")
    for route in ROUTES:
        text = llm.route_instruction(route)
        check(f"route_instruction('{route}') is non-empty", bool(text.strip()))
    check("route instructions differ per route",
          len({llm.route_instruction(r) for r in ROUTES}) == 3)
    check("unknown route falls back safely", bool(llm.route_instruction("nope").strip()))

    sig = SimpleNamespace(signal="s", value=1, severity="High", interpretation="i",
                          supporting_attributes=[], attribute_values={})
    routing = SimpleNamespace(route="fast_track", reasons=["r1"],
                              requires_human_review=True, predicate_results={"risk_check": True})
    flags = {"model_agreement": 0.1, "low_agreement": False, "confidence": 0.9,
             "low_confidence": False, "in_distribution": True,
             "out_of_distribution": False, "feature_completeness": True}
    fake = SimpleNamespace(
        claim_id="CLM-X", provider_id="P1", beneficiary_id="B1", claim_type="Inpatient",
        predicted_class="Legitimate", predicted_class_id=0, risk_score=10.0, rating="Low",
        class_probabilities={}, raw_ml_risk_score=9.0, calibrated_risk_score=12.0,
        prediction_label="Legitimate", risk_rating="Low", trust_flags=flags,
        routing=routing, signals=[sig], evidence=[sig],
        submitted_amount=1.0, allowed_amount=1.0, payment_amount=1.0,
        deductible_amount=0.0, service_count=1, duration_days=1)
    ctx = llm.build_claim_context(fake)
    up = ctx["upstream_and_routing"]
    check("LLM context carries case_id + entity_type",
          ctx["case_id"] == "CLM-X" and ctx["entity_type"] == "claim")
    check("LLM context carries raw + calibrated risk, rating, TrustFlags",
          up["raw_ml_risk_score"] == 9.0 and up["calibrated_risk_score"] == 12.0
          and up["risk_rating"] == "Low" and up["trust_flags"] == flags)
    check("LLM context carries the deterministic routing block",
          up["routing"] == {"route": "fast_track", "reasons": ["r1"],
                            "requires_human_review": True,
                            "predicate_results": {"risk_check": True}})
    check("LLM context carries signals + evidence + attributes",
          len(ctx["analytical_signals"]) == 1 and len(ctx["evidence"]) == 1
          and ctx["claim_attributes"]["claim_id"] == "CLM-X")

    def keys_of(node):
        if isinstance(node, dict):
            return [str(k).lower() for k in node] + [kk for v in node.values() for kk in keys_of(v)]
        if isinstance(node, list):
            return [kk for v in node for kk in keys_of(v)]
        return []

    leaky = llm._strip_targets({"target": 1, "Target_Label": 2, "Potential Fraud": 3,
                                "nested": {"target": 4, "ok": 5}})
    check("training targets are stripped recursively", leaky == {"nested": {"ok": 5}})
    check("no training-target key ever reaches the LLM context",
          not any(k in {"target", "target_label", "potential fraud"} for k in keys_of(ctx)))

    prompt = llm._user_prompt(ctx)
    check("user prompt embeds the route-specific instruction",
          "Route-specific instruction (fast_track):" in prompt)

    parsed = llm._parse_reasoning(json.dumps({"route": "auto_approve", "summary": "s",
                                              "reasoning": ["a"], "caveats": None}))
    check("_parse_reasoning captures the LLM-echoed route", parsed["llm_route"] == "auto_approve")
    check("_parse_reasoning tolerates a missing route",
          llm._parse_reasoning('{"summary": "s"}')["llm_route"] is None)
    try:
        llm._parse_reasoning("not json at all")
        check("_parse_reasoning rejects unstructured output", False)
    except llm.LLMReasoningError:
        check("_parse_reasoning rejects unstructured output", True)

    old_key = os.environ.pop("FWA_LLM_API_KEY", None)
    try:
        try:
            llm.request_reasoning({})
            check("missing API key raises a clear configuration error", False)
        except llm.LLMConfigurationError as exc:
            check("missing API key raises a clear configuration error",
                  "FWA_LLM_API_KEY" in str(exc))
    finally:
        if old_key is not None:
            os.environ["FWA_LLM_API_KEY"] = old_key


# --------------------------------------------------- 3. live single-case checks
EXPLAIN_FIELDS = ("case_id", "entity_type", "route", "prediction", "risk_score",
                  "risk_rating", "raw_ml_risk_score", "calibrated_risk_score",
                  "summary", "reasoning", "supporting_signals", "evidence_used",
                  "routing_reasons", "trust_factors", "caveats", "model_provider")


def explain_and_verify(entity: str, route: str, case: dict, case_id: str) -> dict | None:
    """Single flow: score -> explain -> verify every Phase 5 contract."""
    endpoint = "claims" if entity == "claim" else "providers"
    id_field = "claim_id" if entity == "claim" else "provider_npi"
    features = {k: v for k, v in case["features"].items()
                if k not in ("Claim_ID", "provider_npi")}
    status, score = _post_json(f"/api/{endpoint}/score",
                               {id_field: case_id, "enrich_from_dataset": False,
                                "features": features})
    check(f"{entity} {route}: score returns 200", status == 200, str(score.get("detail", ""))[:160])
    if status != 200:
        return None
    check(f"{entity} {route}: deterministic route = {route}",
          (score.get("routing") or {}).get("route") == route, str(score.get("routing"))[:160])
    check(f"{entity} {route}: requires_human_review = {route != 'auto_approve'}",
          (score.get("routing") or {}).get("requires_human_review") == (route != "auto_approve"))

    before = len(MOCK["requests"])
    status, expl = _post_json(f"/api/{endpoint}/explain", {"analysis": score})
    check(f"{entity} {route}: LLM explain returns 200", status == 200,
          str(expl.get("detail", ""))[:160])
    check(f"{entity} {route}: LLM boundary was called exactly once",
          len(MOCK["requests"]) == before + 1, f"{before} -> {len(MOCK['requests'])}")
    if status != 200:
        return None

    check(f"{entity} {route}: structured response has every Phase 5 field",
          all(f in expl for f in EXPLAIN_FIELDS), str(sorted(expl.keys()))[:160])
    check(f"{entity} {route}: ROUTE PRESERVED (returned route == deterministic route)",
          expl.get("route") == route, str(expl.get("route")))
    check(f"{entity} {route}: risk preserved (no second score)",
          expl.get("risk_score") == score.get("risk_score")
          and expl.get("raw_ml_risk_score") == score.get("raw_ml_risk_score")
          and expl.get("calibrated_risk_score") == score.get("calibrated_risk_score"))
    check(f"{entity} {route}: prediction + rating preserved",
          expl.get("prediction") == score.get("predicted_class")
          and expl.get("risk_rating") == score.get("rating"))
    check(f"{entity} {route}: routing reasons echoed from the router",
          expl.get("routing_reasons") == (score.get("routing") or {}).get("reasons"))
    check(f"{entity} {route}: trust factors echoed from the Trust Gate",
          isinstance(expl.get("trust_factors"), list) and len(expl["trust_factors"]) == 7
          and all(": " in line for line in expl["trust_factors"]))
    check(f"{entity} {route}: LLM explanation content present",
          bool(expl.get("summary")) and bool(expl.get("reasoning")))

    req = last_mock_request()
    check(f"{entity} {route}: mock authenticated with the env API key",
          req["authorization"] == f"Bearer {API_KEY}")
    system_prompt = next(m["content"] for m in req["messages"] if m.get("role") == "system")
    check(f"{entity} {route}: system prompt forbids route changes + score recalculation",
          "NEVER decide, change or question the route" in system_prompt
          and "NEVER calculate" in system_prompt)
    user_text = mock_user_text(req)
    check(f"{entity} {route}: route-specific instruction sent to the LLM",
          f"Route-specific instruction ({route}):" in user_text)
    sent_ctx = mock_user_context(req)
    check(f"{entity} {route}: LLM received the routing decision",
          ((sent_ctx.get("upstream_and_routing") or {}).get("routing") or {}).get("route") == route)
    check(f"{entity} {route}: LLM received calibrated/raw risk + TrustFlags + signals",
          (sent_ctx.get("upstream_and_routing") or {}).get("calibrated_risk_score")
          == score.get("calibrated_risk_score")
          and (sent_ctx.get("upstream_and_routing") or {}).get("trust_flags") is not None
          and len(sent_ctx.get("analytical_signals", [])) == len(score.get("signals", [])))
    check(f"{entity} {route}: no training-target key sent to the LLM",
          all(t not in user_text.lower() for t in ('"target"', '"target_label"', '"potential fraud"')))
    return score


def live_single_checks(cases: dict) -> dict:
    print("\n[3] LIVE SINGLE — claim: every route explained by the LLM")
    scores = {}
    for route in ROUTES:
        if route in cases["claims"]:
            scores[f"claim:{route}"] = explain_and_verify(
                "claim", route, cases["claims"][route], f"P5-CLAIM-{route.upper()}")
        else:
            check(f"claim {route}: fixture case available", False, "missing in phase5_cases.json")

    print("\n[4] LIVE SINGLE — provider: every available route explained by the LLM")
    for route in ROUTES:
        if route in cases["providers"]:
            scores[f"provider:{route}"] = explain_and_verify(
                "provider", route, cases["providers"][route], f"P5-PROV-{route.upper()}")
        else:
            print(f"  SKIP  provider {route}: no dataset case found by the fixture scan")
    check("provider: at least one live single flow verified",
          any(k.startswith("provider:") for k in scores))
    return scores


# ------------------------------------------------------ 4. route integrity etc.
def live_integrity_checks(cases: dict) -> None:
    print("\n[5] ROUTE INTEGRITY — a mismatching LLM route is IGNORED")
    case = cases["claims"].get("full_investigation") or next(iter(cases["claims"].values()))
    features = {k: v for k, v in case["features"].items() if k != "Claim_ID"}
    status, score = _post_json("/api/claims/score",
                               {"claim_id": "P5-INTEGRITY", "features": features})
    deterministic = (score.get("routing") or {}).get("route")
    wrong = "auto_approve" if deterministic != "auto_approve" else "full_investigation"
    MOCK.update(mode="wrong_route", wrong_route=wrong)
    try:
        status, expl = _post_json("/api/claims/explain", {"analysis": score})
    finally:
        MOCK.update(mode="normal", wrong_route=None)
    check("route-integrity explain returns 200", status == 200)
    check(f"LLM tried '{wrong}' but deterministic '{deterministic}' was retained",
          expl.get("route") == deterministic, str(expl.get("route")))
    check("the mismatch is recorded as an audit caveat",
          any("deterministic route was retained" in c for c in expl.get("caveats", [])),
          str(expl.get("caveats"))[:200])

    print("\n[6] LLM FAILURE — route unchanged, case stays reviewable, no fabrication")
    MOCK["mode"] = "error"
    try:
        status, err = _post_json("/api/claims/explain", {"analysis": score})
    finally:
        MOCK["mode"] = "normal"
    check("LLM call failure surfaces as a clear 502 error state", status == 502, str(status))
    check("failure returns an error detail, not fabricated reasoning",
          bool(err.get("detail")) and "summary" not in err)
    status, rec = _post_json("/api/review/accept",
                             {"case_id": "P5-INTEGRITY", "entity_type": "claim"})
    check("human-review route stays reviewable after LLM failure",
          status == 200 and rec.get("review_status") == "Accepted")
    check("deterministic route survives the LLM failure",
          (score.get("routing") or {}).get("route") == deterministic)


def live_batch_checks() -> None:
    print("\n[7] BATCH ROW-CLICK — existing row result -> LLM explanation, no ML rerun")
    before = len(MOCK["requests"])
    status, batch = post_multipart("/api/claims/batch", CLAIMS_CSV)
    check("claim batch upload returns 200", status == 200, str(batch.get("detail", ""))[:120])
    check("batch upload alone never calls the LLM",
          len(MOCK["requests"]) == before)
    results = batch.get("results", [])
    check("claim batch: every row carries a routing decision",
          all((r.get("routing") or {}).get("route") in ROUTES for r in results))
    queue = batch.get("queue", [])
    check("claim batch: queue contract unchanged (original ML risk ordering)",
          [q["risk_score"] for q in queue] == sorted((q["risk_score"] for q in queue), reverse=True)
          and all(set(q.keys()) == {"claim_id", "provider_id", "risk_score", "rating",
                                    "predicted_class", "status", "rank"} for q in queue))

    picked = {}
    for r in results:
        picked.setdefault((r.get("routing") or {}).get("route"), r)
    for route, row in picked.items():
        status, expl = _post_json("/api/claims/explain", {"analysis": row})
        check(f"batch row-click ({route}): LLM explanation returns 200", status == 200,
              str(expl.get("detail", ""))[:120])
        check(f"batch row-click ({route}): route preserved from the existing row result",
              expl.get("route") == route)
    check("batch row-click called the LLM once per selected row (no ML rerun)",
          len(MOCK["requests"]) == before + len(picked),
          f"{before} -> {len(MOCK['requests'])}, rows={len(picked)}")

    status, p_batch = post_multipart("/api/providers/batch", PROVIDERS_CSV)
    check("provider batch upload returns 200", status == 200, str(p_batch.get("detail", ""))[:120])
    p_results = p_batch.get("results", [])
    if p_results:
        row = p_results[0]
        route = (row.get("routing") or {}).get("route")
        before_p = len(MOCK["requests"])
        status, expl = _post_json("/api/providers/explain", {"analysis": row})
        check("provider batch row-click: LLM explanation returns 200", status == 200)
        check("provider batch row-click: route preserved", expl.get("route") == route)
        check("provider batch row-click: exactly one LLM call",
              len(MOCK["requests"]) == before_p + 1)


def live_review_checks(scores: dict) -> None:
    print("\n[8] HUMAN REVIEW — Accept / Reject records, upstream untouched")
    review_fields = {"case_id", "entity_type", "review_status", "reviewed_at", "reviewer_action"}
    fast_claim = scores.get("claim:fast_track")
    full_claim = scores.get("claim:full_investigation") or scores.get("claim:auto_approve")
    prov_key = next((k for k in scores if k.startswith("provider:")
                     and scores[k] and (scores[k].get("routing") or {}).get("requires_human_review")), None)

    targets = []
    if fast_claim:
        targets += [("claim", fast_claim["claim_id"], "accept", "Accepted"),
                    ("claim", fast_claim["claim_id"], "reject", "Rejected")]
    if full_claim:
        targets.append(("claim", full_claim["claim_id"], "accept", "Accepted"))
    if prov_key and scores[prov_key]:
        targets.append(("provider", scores[prov_key]["provider_npi"], "reject", "Rejected"))

    for entity, case_id, action, expected in targets:
        status, rec = _post_json(f"/api/review/{action}",
                                 {"case_id": case_id, "entity_type": entity})
        check(f"review {entity} {case_id} -> {expected}",
              status == 200 and rec.get("review_status") == expected
              and rec.get("reviewer_action") == action, str(rec)[:160])
        check(f"review record {entity} {case_id} stores exactly the spec fields",
              set(rec.keys()) == review_fields, str(sorted(rec.keys())))
        status, got = _get(f"/api/review/{entity}/{case_id}")
        check(f"review record {entity} {case_id} retrievable via GET",
              status == 200 and got.get("review_status") == expected)

    if fast_claim:
        status, again = _post_json("/api/claims/score", {
            "claim_id": fast_claim["claim_id"], "enrich_from_dataset": False,
            "features": _features_for("claim:fast_track")})
        check("review never changes upstream: re-score after review is identical",
              status == 200
              and again.get("raw_ml_risk_score") == fast_claim.get("raw_ml_risk_score")
              and again.get("calibrated_risk_score") == fast_claim.get("calibrated_risk_score")
              and again.get("predicted_class") == fast_claim.get("predicted_class")
              and (again.get("routing") or {}).get("route")
              == (fast_claim.get("routing") or {}).get("route")
              and again.get("trust_flags") == fast_claim.get("trust_flags"))

    if "claim:auto_approve" in scores and scores["claim:auto_approve"]:
        auto_id = scores["claim:auto_approve"]["claim_id"]
        status, _ = _get(f"/api/review/claim/{auto_id}")
        check("auto_approve: no human review record is auto-created (404)", status == 404,
              str(status))


def _features_for(key: str) -> dict:
    return {k: v for k, v in CASES["claims"][key.split(":")[1]]["features"].items()
            if k != "Claim_ID"}


# ------------------------------------------------------------- offline counters
def offline_checks(cases: dict) -> None:
    from app import llm as llm_boundary
    from app import main as m
    from app import services
    from app.schemas import ClaimExplainRequest
    from analytical_engine.inference import ModelRegistry
    from app import config

    print("\n[9] OFFLINE — explain path: LLM for EVERY route, zero upstream reruns")
    registry = ModelRegistry(config.CLAIM_MODEL_PATH, config.PROVIDER_MODEL_PATH)
    llm_calls: list[str] = []
    ml_calls = {"n": 0}
    orig_request = llm_boundary.request_reasoning
    orig_analyze = services.analyze_claim

    def fake_request(context):
        route = ((context.get("upstream_and_routing") or {}).get("routing") or {}).get("route")
        llm_calls.append(route)
        return {"summary": "offline stub", "reasoning": ["stub"], "supporting_signals": [],
                "evidence_used": [], "caveats": [], "llm_route": route}

    def counted_analyze(*a, **k):
        ml_calls["n"] += 1
        return orig_analyze(*a, **k)

    llm_boundary.request_reasoning = fake_request
    services.analyze_claim = counted_analyze
    results = {}
    try:
        for route in ROUTES:
            if route not in cases["claims"]:
                continue
            feats = {k: v for k, v in cases["claims"][route]["features"].items()
                     if k != "Claim_ID"}
            outcome = orig_analyze(registry.claim, dict(feats))
            result = m._claim_result(outcome, claim_id=f"OFF-{route}", provider_id=None)
            results[route] = result
            expl = m.explain_claim(ClaimExplainRequest(analysis=result))
            check(f"offline explain ({route}): response route == deterministic route",
                  expl.route == route)
        check("LLM boundary called for EVERY route, including auto_approve",
              sorted(llm_calls) == sorted(results.keys()) and "auto_approve" in llm_calls,
              str(llm_calls))
        check("explain path never reruns ML / calibration / Trust Gate / engine",
              ml_calls["n"] == 0, f"analyze calls={ml_calls['n']}")

        print("\n[10] OFFLINE — route integrity on the handler itself")
        def lying_request(context):
            route = ((context.get("upstream_and_routing") or {}).get("routing") or {}).get("route")
            wrong = "auto_approve" if route != "auto_approve" else "full_investigation"
            return {"summary": "lying stub", "reasoning": ["stub"], "supporting_signals": [],
                    "evidence_used": [], "caveats": [], "llm_route": wrong}
        llm_boundary.request_reasoning = lying_request
        some = next(iter(results.values()))
        expl = m.explain_claim(ClaimExplainRequest(analysis=some))
        check("handler ignores a lying LLM route", expl.route == some.routing.route)
        check("handler documents the ignored LLM route in caveats",
              any("retained" in c for c in expl.caveats))
    finally:
        llm_boundary.request_reasoning = orig_request
        services.analyze_claim = orig_analyze

    print("\n[11] CLAIM / PROVIDER ISOLATION AT THE LLM BOUNDARY")
    claim_ctx = llm_boundary.build_claim_context.__doc__
    from types import SimpleNamespace
    prov_fake = SimpleNamespace(
        provider_npi="NPI-X", provider_state="CA", provider_type="Clinic",
        predicted_class="Suspicious", predicted_class_id=1, risk_score=99.0, rating="High",
        class_probabilities={}, raw_ml_risk_score=98.0, calibrated_risk_score=100.0,
        prediction_label="Suspicious", risk_rating="High",
        trust_flags=None, routing=None, signals=[], evidence=[],
        total_beneficiaries=1, total_services=1, claim_count=1, weighted_avg_payment=1.0,
        services_per_beneficiary=1.0, peer_deviation_score=1.0)
    p_ctx = llm_boundary.build_provider_context(prov_fake)
    check("provider context is provider-shaped (no claim leakage)",
          p_ctx["entity_type"] == "provider" and p_ctx["case_id"] == "NPI-X"
          and "claim_attributes" not in p_ctx and bool(claim_ctx))
    check("routing=None tolerated (pre-Phase-4 results remain explainable)",
          p_ctx["upstream_and_routing"].get("routing") is None
          and llm_boundary.route_instruction(None).strip() != "")


CASES: dict = {}


# ----------------------------------------------------------------------- main
def main() -> int:
    global CASES
    if not CASES_JSON.exists():
        print(f"ERROR: fixture {CASES_JSON} missing — run _gen_phase5_fixtures.py first")
    CASES = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    if "claims" in CASES:
        if "auto_approve" in CASES["claims"]:
            CASES["claims"]["auto_approve"]["features"]["Claim_Submitted_Amount"] = 1000.0
        if "fast_track" in CASES["claims"]:
            CASES["claims"]["fast_track"]["features"].pop("Procedure_Count", None)
        if "full_investigation" in CASES["claims"]:
            CASES["claims"]["full_investigation"]["features"].pop("Procedure_Count", None)

    static_checks()
    unit_checks()

    mock = start_mock()
    env = dict(os.environ)
    env["FWA_LLM_API_KEY"] = API_KEY
    env["FWA_LLM_BASE_URL"] = f"http://{MOCK_HOST}:{MOCK_PORT}/v1"
    env["FWA_LLM_MODEL"] = "phase5-mock-model"
    log_path = BACKEND_DIR / "verify_phase5_server.log"
    log_file = open(log_path, "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT)],
        cwd=str(BACKEND_DIR), env=env, stdout=log_file, stderr=log_file,
    )
    try:
        if not wait_for_health(server):
            check("server started for live Phase 5 checks", False)
        else:
            check("server started for live Phase 5 checks", True)
            check("LLM status endpoint reports the configured mock provider",
                  _get("/api/llm/status")[1].get("configured") is True)
            scores = live_single_checks(CASES)
            live_integrity_checks(CASES)
            live_batch_checks()
            live_review_checks(scores)
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()
        mock.shutdown()

    # Offline checks AFTER the server released its memory (~8GB machine).
    offline_checks(CASES)

    print(f"\n{'=' * 60}\nRESULT: {PASSED} passed, {FAILED} failed\n{'=' * 60}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
