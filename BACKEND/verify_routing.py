"""Phase 4 verification — DETERMINISTIC ROUTING.

Covers (per the Phase 4 brief):
  * auto_approve / fast_track / full_investigation cases
  * every predicate independently, threshold boundaries, route precedence
  * missing/weak trust, high/low calibrated risk, low agreement,
    low confidence, out-of-distribution, incomplete features
  * claim routing, provider routing, batch routing
  * routing does NOT change raw ML risk / calibrated risk / prediction /
    TrustFlags / signals / evidence
  * routing does NOT rerun ML / calibration / Trust Gate / Analytical Engine
    / LLM after those results already exist
  * live API tests on all four scoring endpoints with real uploaded rows

Run:  python verify_routing.py
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

HOST, PORT = "127.0.0.1", 8016
BASE = f"http://{HOST}:{PORT}"
FIXTURES = BACKEND_DIR.parent / "test_fixtures"
CLAIMS_CSV = FIXTURES / "test_uploaded_claims.csv"
PROVIDERS_CSV = FIXTURES / "test_uploaded_providers.csv"

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


# --------------------------------------------------------------------- helpers
def clean_flags(**over) -> dict:
    """A fully-trustworthy TrustFlags dict; individual fields overridable."""
    flags = {
        "model_agreement": 0.02,
        "low_agreement": False,
        "confidence": 0.55,
        "low_confidence": False,
        "in_distribution": True,
        "out_of_distribution": False,
        "feature_completeness": True,
    }
    flags.update(over)
    return flags


def ctx(calibrated: float, prediction: str = "Legitimate", flags: dict | None = None) -> dict:
    return {
        "raw_ml_risk_score": round(calibrated * 0.9, 6),
        "calibrated_risk_score": calibrated,
        "prediction": prediction,
        "trust_flags": flags if flags is not None else clean_flags(),
    }


def signals(count: int, severity: str = "High") -> list:
    return [SimpleNamespace(severity=severity) for _ in range(count)]


def override_cfg(**claims_over) -> dict:
    from app import routing

    cfg = routing.load_config()
    cfg = copy.deepcopy(cfg)
    cfg["claims"].update(claims_over)
    return cfg


def _post_json(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def post_multipart(path: str, csv_path: Path) -> tuple[int, dict]:
    boundary = "----phase4boundary"
    raw = csv_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
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
            with urllib.request.urlopen(f"{BASE}/api/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


PREDICATES = (
    "risk_check", "agreement_check", "confidence_check", "distribution_check",
    "completeness_check", "prediction_check", "signal_check",
)
ROUTES = {"auto_approve", "fast_track", "full_investigation"}


def routing_ok(decision: dict | None) -> bool:
    """Full routing structure: 4 keys, valid route, 7 audited predicates,
    human-review flag consistent with the route."""
    if not isinstance(decision, dict):
        return False
    if set(decision.keys()) != {"route", "reasons", "requires_human_review", "predicate_results"}:
        return False
    if decision["route"] not in ROUTES:
        return False
    if not isinstance(decision["reasons"], list) or not decision["reasons"]:
        return False
    if list(decision["predicate_results"].keys()) != list(PREDICATES):
        return False
    return decision["requires_human_review"] == (decision["route"] != "auto_approve")


# =========================================================== 1. static guards
def static_checks() -> None:
    print("\n[1] STATIC GUARDS — routing is rule-based and never the LLM")
    src = (BACKEND_DIR / "app" / "routing.py").read_text(encoding="utf-8")
    check("routing.py imports no LLM client", not any(t in src for t in ("openai", "httpx", "requests.", "FWA_LLM_API_KEY")))
    check("routing.py imports no ML/calibration/Trust Gate/engine module",
          not any(t in src for t in ("analytical_engine", "from .calibration", "trust_gate", "inference")))
    check("routing.py never mentions an orchestrator/agent loop",
          not any(t in src for t in ("orchestrator", "agent_loop", "random.")))
    from app import routing
    check("documented precedence = full -> fast -> auto",
          routing.ROUTE_PRECEDENCE == ("full_investigation", "fast_track", "auto_approve"))
    cfg = routing.load_config()
    check("config precedence matches code precedence",
          tuple(cfg.get("route_precedence", [])) == routing.ROUTE_PRECEDENCE)
    check("claim and provider sections are separate",
          set(cfg["claims"].keys()) & {"threshold_provenance"} != set() and "claims" in cfg and "providers" in cfg
          and cfg["claims"]["full_investigation_risk_threshold"] != cfg["providers"]["full_investigation_risk_threshold"])
    check("config documents the future-LLM architectural rule",
          "routing layer decides" in cfg.get("architecture_rule_for_future_llm_phase", ""))
    services_src = (BACKEND_DIR / "app" / "services.py").read_text(encoding="utf-8")
    check("services.py wires routing after the Analytical Engine",
          "rt.route_claim" in services_src and "rt.route_provider" in services_src)


# ============================================================ 2. unit policy
def unit_policy_checks() -> None:
    from app import routing

    print("\n[2] THE THREE ROUTES — canonical cases")
    auto = routing.route_claim(ctx(5.0), [])
    check("auto_approve case: clean + low calibrated risk + no signals",
          auto["route"] == "auto_approve", str(auto))
    check("auto_approve: requires_human_review is False", auto["requires_human_review"] is False)
    check("auto_approve: all 7 predicates pass", all(auto["predicate_results"].values()))

    fast = routing.route_claim(ctx(30.0), [])  # between 25.11 and 58.76
    check("fast_track case: moderate calibrated risk, every predicate passing",
          fast["route"] == "fast_track", str(fast))
    check("fast_track: requires_human_review is True", fast["requires_human_review"] is True)

    fast_sig = routing.route_claim(ctx(5.0), signals(8))
    check("fast_track case: moderate supporting-signal count (8 == p90 anchor)",
          fast_sig["route"] == "fast_track", str(fast_sig))

    full = routing.route_claim(ctx(70.0), [])
    check("full_investigation case: high calibrated risk (70 > 58.76)",
          full["route"] == "full_investigation", str(full))
    check("full_investigation: requires_human_review is True", full["requires_human_review"] is True)

    print("\n[3] EVERY PREDICATE INDEPENDENTLY")
    cases = [
        ("agreement_check", ctx(5.0, flags=clean_flags(low_agreement=True, model_agreement=0.3))),
        ("confidence_check", ctx(5.0, flags=clean_flags(low_confidence=True, confidence=0.05))),
        ("distribution_check", ctx(5.0, flags=clean_flags(in_distribution=False, out_of_distribution=True))),
        ("completeness_check", ctx(5.0, flags=clean_flags(feature_completeness=False))),
        ("prediction_check", ctx(5.0, prediction="Fraud")),
        ("signal_check_none", ctx(5.0)),  # control
    ]
    for name, c in cases[:-1]:
        decision = routing.route_claim(c, signals(0))
        check(f"failing {name} escalates to full_investigation",
              decision["route"] == "full_investigation" and decision["predicate_results"][name] is False,
              str(decision["route"]))
    check("control case stays auto_approve", routing.route_claim(cases[-1][1], [])["route"] == "auto_approve")

    full_sig = routing.route_claim(ctx(5.0), signals(9))
    check("signal_check fails at 9 supporting signals (>= full anchor)",
          full_sig["route"] == "full_investigation" and full_sig["predicate_results"]["signal_check"] is False)

    provider_susp = routing.route_provider(ctx(5.0, prediction="Suspicious"), [])
    check("provider: predicted 'Suspicious' fails prediction_check -> full_investigation",
          provider_susp["route"] == "full_investigation")

    print("\n[4] THRESHOLD BOUNDARIES")
    cfg = routing.load_config()
    c_full = cfg["claims"]["full_investigation_risk_threshold"]
    c_fast = cfg["claims"]["fast_track_risk_threshold"]
    check("calibrated == full threshold escalates (>= semantics)",
          routing.route_claim(ctx(c_full), [])["route"] == "full_investigation")
    check("calibrated infinitesimally below full threshold does not escalate on risk",
          routing.route_claim(ctx(c_full - 0.0001), [])["route"] == "fast_track")
    check("calibrated == fast threshold routes fast_track (>= semantics)",
          routing.route_claim(ctx(c_fast), [])["route"] == "fast_track")
    check("calibrated just below fast threshold stays auto_approve",
          routing.route_claim(ctx(c_fast - 0.0001), [])["route"] == "auto_approve")
    custom = override_cfg(full_investigation_risk_threshold=42.5)
    check("thresholds are configurable (custom full threshold 42.5)",
          routing.route_claim(ctx(43.0), [], config=custom)["route"] == "full_investigation"
          and routing.route_claim(ctx(42.0), [], config=custom)["route"] != "full_investigation")
    check("supporting signals == fast anchor - 1 stay auto_approve",
          routing.route_claim(ctx(5.0), signals(7))["route"] == "auto_approve")
    check("Low-severity signals never count as supporting",
          routing.route_claim(ctx(5.0), signals(11, severity="Low"))["route"] == "auto_approve")

    print("\n[5] ROUTE PRECEDENCE")
    both = routing.route_claim(ctx(70.0, flags=clean_flags(low_agreement=True)), signals(9))
    check("full beats fast when both could apply (high risk + weak trust + signals)",
          both["route"] == "full_investigation")
    mid = routing.route_claim(ctx(30.0), signals(8))
    check("fast beats auto when moderate risk AND moderate signals apply",
          mid["route"] == "fast_track")
    check("precedence never depends on dict ordering (fixed audit order)",
          list(full["predicate_results"].keys()) == list(PREDICATES))

    print("\n[6] MISSING / WEAK TRUST CONDITIONS (fail-safe)")
    check("no upstream context at all -> full_investigation",
          routing.route_claim(None, [])["route"] == "full_investigation")
    check("missing calibrated risk -> full_investigation",
          routing.route_claim({"prediction": "Legitimate", "trust_flags": clean_flags()}, [])["route"] == "full_investigation")
    check("missing trust_flags -> full_investigation",
          routing.route_claim({"calibrated_risk_score": 1.0, "prediction": "Legitimate"}, [])["route"] == "full_investigation")
    no_trust = routing.route_claim({"calibrated_risk_score": 1.0, "prediction": "Legitimate"}, [])
    check("missing trust_flags marks every trust predicate failed",
          all(no_trust["predicate_results"][k] is False for k in
              ("agreement_check", "confidence_check", "distribution_check", "completeness_check")))

    print("\n[7] CLAIM / PROVIDER ISOLATION")
    same = ctx(60.0)  # above claim full threshold 58.76, inside provider fast band [53.16, 100)
    check("same case routes differently per entity config (claim full vs provider fast)",
          routing.route_claim(same, [])["route"] == "full_investigation"
          and routing.route_provider(same, [])["route"] == "fast_track")
    check("provider full threshold is entity-specific (100.0 anchor)",
          routing.route_provider(ctx(100.0), [])["route"] == "full_investigation"
          and routing.route_provider(ctx(99.99), [])["route"] == "fast_track")

    print("\n[8] DETERMINISM + AUDITABILITY")
    one = routing.route_claim(ctx(30.0), signals(3))
    two = routing.route_claim(ctx(30.0), signals(3))
    check("identical inputs -> identical decisions (no randomness)", one == two)
    check("every decision carries human-readable reasons",
          all(routing.route_claim(c, s)["reasons"] for c, s in ((ctx(5.0), []), (ctx(70.0), []), (ctx(30.0), []))))


# ====================================================== 3. live server checks
def live_checks() -> dict:
    print("\n[9] LIVE API — GET /api/health")
    with urllib.request.urlopen(f"{BASE}/api/health", timeout=30) as resp:
        health = json.loads(resp.read().decode())
    check("health ok", health.get("status") == "ok")

    claim_payload = {"claim_id": None, "enrich_from_dataset": False, "features": {}}
    status, claim_single = _post_json("/api/claims/score", claim_payload)
    check("POST /api/claims/score returns 200", status == 200, str(claim_single.get("detail", ""))[:120])
    check("claim single: routing present with full structure", routing_ok(claim_single.get("routing")),
          str(claim_single.get("routing"))[:160])
    check("claim single: upstream fields intact alongside routing",
          claim_single.get("raw_ml_risk_score") is not None
          and claim_single.get("calibrated_risk_score") is not None
          and claim_single.get("prediction") == claim_single.get("predicted_class")
          and isinstance(claim_single.get("trust_flags"), dict)
          and len(claim_single.get("signals", [])) > 0)
    print(f"      claim single route = {claim_single.get('routing', {}).get('route')}")

    status, prov_single = _post_json("/api/providers/score", {"provider_npi": None, "enrich_from_dataset": False, "features": {}})
    check("POST /api/providers/score returns 200", status == 200, str(prov_single.get("detail", ""))[:120])
    check("provider single: routing present with full structure", routing_ok(prov_single.get("routing")))
    print(f"      provider single route = {prov_single.get('routing', {}).get('route')}")

    print("\n[10] LIVE API — batch endpoints with REAL uploaded CSVs")
    status, claim_batch = post_multipart("/api/claims/batch", CLAIMS_CSV)
    check("POST /api/claims/batch returns 200", status == 200, str(claim_batch.get("detail", ""))[:120])
    results = claim_batch.get("results", [])
    check("claim batch: 25 results for 25 uploaded rows", claim_batch.get("total") == 25 and len(results) == 25)
    check("claim batch: EVERY row has exactly one routing decision",
          all(routing_ok(r.get("routing")) for r in results))
    check("claim batch: one route per row (no shared batch-level route)",
          len([r for r in results if r.get("routing")]) == 25)
    check("claim batch: existing fields preserved (raw/calibrated/trust_flags/signals/evidence)",
          all(r.get("raw_ml_risk_score") is not None and r.get("calibrated_risk_score") is not None
              and isinstance(r.get("trust_flags"), dict) and isinstance(r.get("signals"), list)
              and isinstance(r.get("evidence"), list) for r in results))
    queue = claim_batch.get("queue", [])
    check("claim batch: queue unchanged contract (original ML risk, exact keys)",
          len(queue) == 25
          and all(set(q.keys()) == {"claim_id", "provider_id", "risk_score", "rating",
                                    "predicted_class", "status", "rank"} for q in queue)
          and all("routing" not in q and "calibrated_risk_score" not in q for q in queue))
    check("claim batch: queue sorted by original ML risk desc, ranks 1..N",
          [q["risk_score"] for q in queue] == sorted((q["risk_score"] for q in queue), reverse=True)
          and [q["rank"] for q in queue] == list(range(1, 26)))
    rc = claim_batch.get("summary", {}).get("routing_counts", {})
    check("claim batch: summary routing_counts present and consistent",
          set(rc.keys()) == ROUTES and sum(rc.values()) == 25)
    print(f"      claim routing_counts = {rc}")

    status, prov_batch = post_multipart("/api/providers/batch", PROVIDERS_CSV)
    check("POST /api/providers/batch returns 200", status == 200, str(prov_batch.get("detail", ""))[:120])
    p_results = prov_batch.get("results", [])
    check("provider batch: 7 results for 7 uploaded rows", prov_batch.get("total") == 7 and len(p_results) == 7)
    check("provider batch: EVERY row has exactly one routing decision",
          all(routing_ok(r.get("routing")) for r in p_results))
    prc = prov_batch.get("summary", {}).get("routing_counts", {})
    check("provider batch: summary routing_counts present and consistent",
          set(prc.keys()) == ROUTES and sum(prc.values()) == 7)
    print(f"      provider routing_counts = {prc}")
    return {"claim_single": claim_single, "prov_single": prov_single,
            "claim_batch": claim_batch, "prov_batch": prov_batch}


# ==================================================== 4. offline ML integrity
def offline_checks() -> None:
    import numpy as np
    import pandas as pd

    from analytical_engine import engine as ae
    from analytical_engine.inference import ModelRegistry
    from app import config, services
    from app import routing as rt
      from trust_engine import trust_gate as tg

    registry = ModelRegistry(config.CLAIM_MODEL_PATH, config.PROVIDER_MODEL_PATH)
    claims = pd.read_csv(config.CLAIM_DATA_PATH, low_memory=False).head(10).to_dict("records")
    providers = pd.read_csv(config.PROVIDER_DATA_PATH, low_memory=False).head(6).to_dict("records")

    print("\n[11] ROUTING RUNS EXACTLY ONCE PER CASE AND NEVER RERUNS UPSTREAM STAGES")
    counters = {"routing": 0, "engine": 0, "ml": 0, "calibrate": 0, "gate": 0}
    orig_route, orig_engine = rt.route_claim, services.run_claim_engine
    orig_ml = services.predict_claims
    orig_cal = services._calibrator("claim").calibrate_risk_scores
    orig_gate = tg.claim_trust_flags

    def counted_route(c, s, config=None):
        counters["routing"] += 1
        return orig_route(c, s, config=config)

    def counted_engine(model, ml, row, upstream_context=None):
        counters["engine"] += 1
        return orig_engine(model, ml, row, upstream_context=upstream_context)

    def counted_ml(model, records, return_blends=False):
        counters["ml"] += 1
        return orig_ml(model, records, return_blends=return_blends)

    def counted_cal(raw):
        counters["calibrate"] += 1
        return orig_cal(raw)

    def counted_gate(*a, **k):
        counters["gate"] += 1
        return orig_gate(*a, **k)

    rt.route_claim = counted_route
    services.run_claim_engine = counted_engine
    services.predict_claims = counted_ml
    services._calibrator("claim").calibrate_risk_scores = counted_cal
    tg.claim_trust_flags = counted_gate
    try:
        batch = services.analyze_claims_batch(registry.claim, claims)
    finally:
        rt.route_claim = orig_route
        services.run_claim_engine = orig_engine
        services.predict_claims = orig_ml
        services._calibrator("claim").calibrate_risk_scores = orig_cal
        tg.claim_trust_flags = orig_gate

    n = len(claims)
    check(f"batch of {n}: vectorized ML ran exactly once", counters["ml"] == 1, str(counters))
    check("batch: calibration ran exactly once", counters["calibrate"] == 1)
    check("batch: Trust Gate ran exactly once", counters["gate"] == 1)
    check(f"batch: Analytical Engine ran {n} times (per row)", counters["engine"] == n)
    check(f"batch: routing ran exactly {n} times (once per case)", counters["routing"] == n)
    check("batch: every outcome carries a routing decision",
          all(routing_ok(o.upstream_context.get("routing")) for o in batch["outcomes"]))

    print("\n[12] ROUTING NEVER MODIFIES UPSTREAM VALUES, SIGNALS OR EVIDENCE")
    outcome = services.analyze_claim(registry.claim, claims[0])
    before_ctx = {k: copy.deepcopy(v) for k, v in outcome.upstream_context.items() if k != "routing"}
    # Re-route the already-complete outcome and prove nothing upstream moves.
    fresh = rt.route_claim(outcome.upstream_context, outcome.signals)
    after_ctx = {k: v for k, v in outcome.upstream_context.items() if k != "routing"}
    check("rerunning the router on an existing result changes nothing upstream", before_ctx == after_ctx)
    check("routing decision is reproducible for the same result",
          fresh == outcome.upstream_context["routing"])
    sig_before = copy.deepcopy([s.__dict__ for s in outcome.signals])
    ev_before = copy.deepcopy([e.__dict__ for e in outcome.evidence])
    rt.route_claim(outcome.upstream_context, outcome.signals)
    check("signals are byte-identical after routing", sig_before == [s.__dict__ for s in outcome.signals])
    check("evidence is byte-identical after routing", ev_before == [e.__dict__ for e in outcome.evidence])
    check("raw ML risk and calibrated risk are untouched by routing",
          outcome.upstream_context["raw_ml_risk_score"] == before_ctx["raw_ml_risk_score"]
          and outcome.upstream_context["calibrated_risk_score"] == before_ctx["calibrated_risk_score"]
          and outcome.upstream_context["trust_flags"] == before_ctx["trust_flags"])

    print("\n[13] ENGINE OUTPUT IDENTICAL WITH AND WITHOUT ROUTING")
    direct = ae.run_claim_engine(
        registry.claim,
        services._claim_stage(registry.claim, [claims[1]])[1][0],
        services._merged_rows([claims[1]], services._claim_stage(registry.claim, [claims[1]])[0])[0],
    )
    routed = services.analyze_claim(registry.claim, claims[1])
    check("signals identical: engine-only vs engine+routing",
          [s.__dict__ for s in direct.signals] == [s.__dict__ for s in routed.signals])
    check("evidence identical: engine-only vs engine+routing",
          [e.__dict__ for e in direct.evidence] == [e.__dict__ for e in routed.evidence])
    check("ML output identical: risk score / prediction unchanged",
          direct.ml.risk_score == routed.ml.risk_score
          and direct.ml.predicted_class == routed.ml.predicted_class)

    print("\n[14] PROVIDER ROUTING + QUEUE REGRESSION + ISOLATION")
    services._CALIBRATORS.clear()
    p_batch = services.analyze_providers_batch(registry.provider, providers)
    check("provider batch: every row routed",
          all(routing_ok(o.upstream_context.get("routing")) for o in p_batch["outcomes"]))
    check("provider summary routing_counts consistent",
          set(p_batch["summary"]["routing_counts"].keys()) == ROUTES
          and sum(p_batch["summary"]["routing_counts"].values()) == len(providers))
    check("provider queue contract unchanged (original ML risk, exact keys)",
          all(set(q.keys()) == {"provider_npi", "provider_type", "risk_score", "rating",
                                "predicted_class", "status", "rank"} for q in p_batch["queue"])
          and [q["risk_score"] for q in p_batch["queue"]] ==
          sorted((q["risk_score"] for q in p_batch["queue"]), reverse=True))
    check("isolation: only the provider calibrator was loaded on the provider flow",
          set(services._CALIBRATORS.keys()) == {"provider"})
    services._CALIBRATORS.clear()
    services.analyze_claim(registry.claim, claims[2])
    check("isolation: only the claim calibrator was loaded on the claim flow",
          set(services._CALIBRATORS.keys()) == {"claim"})

    print("\n[15] REAL-DATA ROUTE COVERAGE EXAMPLES")
    for o in batch["outcomes"][:3]:
        c = o.upstream_context
        r = c["routing"]
        print(f"      claim calibrated={c['calibrated_risk_score']:.2f} "
              f"prediction={c['prediction']} -> {r['route']} | {r['reasons'][0]}")
    for o in p_batch["outcomes"][:3]:
        c = o.upstream_context
        r = c["routing"]
        print(f"      provider calibrated={c['calibrated_risk_score']:.2f} "
              f"prediction={c['prediction']} -> {r['route']} | {r['reasons'][0]}")
    check("at least one non-auto route observed on real data (sanity printout above)",
          any(o.upstream_context["routing"]["route"] != "auto_approve" for o in batch["outcomes"])
          or any(o.upstream_context["routing"]["route"] != "auto_approve" for o in p_batch["outcomes"]))


# ----------------------------------------------------------------------- main
def main() -> int:
    static_checks()
    unit_policy_checks()

    env = {k: v for k, v in os.environ.items() if k != "FWA_LLM_API_KEY"}
    log_path = BACKEND_DIR / "verify_routing_server.log"
    log_file = open(log_path, "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT)],
        cwd=str(BACKEND_DIR), env=env, stdout=log_file, stderr=log_file,
    )
    try:
        if not wait_for_health(server):
            check("server started for live API checks", False)
        else:
            check("server started for live API checks", True)
            live_checks()
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()

    # Offline ML checks AFTER the server released its memory (~8GB machine).
    offline_checks()

    print(f"\n{'=' * 60}\nRESULT: {PASSED} passed, {FAILED} failed\n{'=' * 60}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
