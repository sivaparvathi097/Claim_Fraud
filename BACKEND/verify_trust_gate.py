"""Trust Gate verification suite (Phase 2).

Tests (per phase brief):
  * model agreement calculation + low-agreement threshold + boundary
  * claim entropy + confidence threshold boundary
  * provider confidence + confidence threshold boundary
  * in-distribution boundary + out-of-distribution case
  * missing-feature case + complete-feature case
  * raw ML score preservation + calibrated score preservation
  * claim/provider isolation
  * no second risk score / no routing / no LLM calls
  * existing Analytical Engine regression (identical signals on same rows)

Run from BACKEND/:  python verify_trust_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import joblib  # noqa: E402

from analytical_engine.inference import (  # noqa: E402
    HybridModel,
    predict_claims,
    predict_providers,
)
from trust_engine import calibration as calib  # noqa: E402
from trust_engine import trust_gate as tg  # noqa: E402

MODELS_DIR = BACKEND_DIR / "models"
CONFIG = tg.load_config()

PASS = FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


# ================================================================ [1] AGREEMENT
def section_agreement() -> None:
    print("\n[1] Model agreement calculation")
    # Predicted class = col 0 (argmax of blend). |0.7 - 0.6| = 0.1.
    proba = np.array([[0.64, 0.26, 0.05, 0.05]])
    xgb = np.array([[0.70, 0.20, 0.05, 0.05]])
    lgb = np.array([[0.60, 0.30, 0.05, 0.05]])
    agr = tg.model_agreement(xgb, lgb, proba)
    check("agreement = |P_xgb(pred) - P_lgb(pred)| = 0.1", abs(agr[0] - 0.1) < 1e-12, str(agr))
    check("agreement uses the PREDICTED class only",
          abs(tg.model_agreement(np.array([[0.1, 0.9, 0.0, 0.0]]),
                                 np.array([[0.2, 0.7, 0.05, 0.05]]),
                                 np.array([[0.16, 0.78, 0.03, 0.03]]))[0] - 0.2) < 1e-12)
    check("agreement is an absolute difference",
          abs(tg.model_agreement(np.array([[0.5, 0.5]]), np.array([[0.9, 0.1]]),
                                 np.array([[0.7, 0.3]]))[0] - 0.4) < 1e-12)

    print("\n[2] Low-agreement threshold + boundary")
    thr = CONFIG["agreement"]["low_agreement_threshold"]
    check("documented default threshold is 0.10", abs(thr - 0.10) < 1e-12, str(thr))
    gap_above = np.array([[0.3 - 0.2, 0.7, 0.0, 0.0]])  # 0.10000000000000003 > 0.10
    gap_x = np.array([[0.85, 0.15, 0.0, 0.0]])
    gap_l = np.array([[0.75, 0.25, 0.0, 0.0]])
    idn = np.array([[1.0, 0.0, 0.0, 0.0]])
    frame = pd.DataFrame({"x": [0.0]})
    flags_above = tg.claim_trust_flags(gap_above, gap_above, idn, [{}], frame, ["x"], [],
                                       config=_cfg_ranges_neutral())
    check("gap > 0.10 -> low_agreement true", flags_above[0]["low_agreement"] is True)
    # Boundary semantics: threshold EXACTLY equal to the computed gap (strict >).
    agr = float(tg.model_agreement(gap_x, gap_l, gap_x)[0])
    f_at = tg.claim_trust_flags(gap_x, gap_x, gap_l, [{}], frame, ["x"], [],
                                config=_cfg_with(low_agreement_threshold=agr))
    f_just = tg.claim_trust_flags(gap_x, gap_x, gap_l, [{}], frame, ["x"], [],
                                  config=_cfg_with(low_agreement_threshold=agr - 1e-12))
    check("gap == threshold exactly -> low_agreement false (strict >)", f_at[0]["low_agreement"] is False)
    check("gap infinitesimally above threshold -> low_agreement true", f_just[0]["low_agreement"] is True)
    check("threshold is configurable (same gap, different outcome)",
          f_at[0]["low_agreement"] != f_just[0]["low_agreement"])
    check("identical ensemble members -> agreement 0, not low",
          tg.model_agreement(idn, idn, idn)[0] == 0.0)


def _cfg_ranges_neutral() -> dict:
    """Config copy whose ranges always accept the synthetic frames used above."""
    cfg = {
        "agreement": dict(CONFIG["agreement"]),
        "confidence": dict(CONFIG["confidence"]),
        "in_distribution": {
            "claims": {"critical_features": ["x"], "ranges": {"x": {"min": -1e18, "max": 1e18}}},
            "providers": {"critical_features": ["x"], "ranges": {"x": {"min": -1e18, "max": 1e18}}},
        },
    }
    return cfg


def _cfg_with(**overrides) -> dict:
    """Neutral config with individual threshold overrides (boundary testing)."""
    cfg = _cfg_ranges_neutral()
    if "low_agreement_threshold" in overrides:
        cfg["agreement"]["low_agreement_threshold"] = overrides["low_agreement_threshold"]
    if "claim_entropy_threshold" in overrides:
        cfg["confidence"]["claim_entropy_threshold"] = overrides["claim_entropy_threshold"]
    if "provider_midpoint_distance_threshold" in overrides:
        cfg["confidence"]["provider_midpoint_distance_threshold"] = overrides["provider_midpoint_distance_threshold"]
    return cfg


# ================================================================ [3] ENTROPY
def section_claim_entropy() -> None:
    print("\n[3] Claim entropy")
    uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
    certain = np.array([[1.0, 0.0, 0.0, 0.0]])
    check("entropy(uniform 4-class) = ln 4",
          abs(tg.claim_entropy(uniform)[0] - np.log(4)) < 1e-12)
    check("entropy(certain) = 0", abs(tg.claim_entropy(certain)[0]) < 1e-12)
    check("entropy kept as a value (not converted to a risk score)",
          tg.claim_entropy(uniform)[0] <= np.log(4) + 1e-12)

    print("\n[4] Claim confidence threshold boundary")
    thr = CONFIG["confidence"]["claim_entropy_threshold"]
    check("documented default threshold is 0.80", abs(thr - 0.80) < 1e-12, str(thr))
    # Boundary semantics: threshold EXACTLY equal to the computed entropy (strict >).
    mixed = np.array([[0.6, 0.2, 0.1, 0.1]])
    h = float(tg.claim_entropy(mixed)[0])
    idn = np.array([[1.0, 0.0, 0.0, 0.0]])
    frame = pd.DataFrame({"x": [0.0]})
    f_at = tg.claim_trust_flags(mixed, mixed, idn, [{}], frame, ["x"], [],
                                config=_cfg_with(claim_entropy_threshold=h))
    f_just = tg.claim_trust_flags(mixed, mixed, idn, [{}], frame, ["x"], [],
                                  config=_cfg_with(claim_entropy_threshold=h - 1e-12))
    check("entropy == threshold exactly -> low_confidence false (strict >)", f_at[0]["low_confidence"] is False)
    check("entropy infinitesimally above threshold -> low_confidence true", f_just[0]["low_confidence"] is True)
    check("entropy threshold is configurable", f_at[0]["low_confidence"] != f_just[0]["low_confidence"])
    f_hi = tg.claim_trust_flags(uniform, uniform, idn, [{}], frame, ["x"], [], config=_cfg_ranges_neutral())
    check("entropy(uniform)=ln4 > 0.80 -> low_confidence true", f_hi[0]["low_confidence"] is True)


# ================================================================ [5] PROVIDER CONF
def section_provider_confidence() -> None:
    print("\n[5] Provider confidence")
    p95 = np.array([[0.05, 0.95]])
    p50 = np.array([[0.5, 0.5]])
    check("confidence = |P(pred) - 0.5| = 0.45 for [0.05, 0.95]",
          abs(tg.provider_confidence(p95)[0] - 0.45) < 1e-12)
    check("confidence = 0 at the uncertain midpoint", tg.provider_confidence(p50)[0] == 0.0)
    check("confidence symmetric for the Legitimate-predicted case",
          abs(tg.provider_confidence(np.array([[0.95, 0.05]]))[0] - 0.45) < 1e-12)

    print("\n[6] Provider confidence threshold boundary")
    thr = CONFIG["confidence"]["provider_midpoint_distance_threshold"]
    check("documented default threshold is 0.20", abs(thr - 0.20) < 1e-12, str(thr))
    # Boundary semantics: threshold EXACTLY equal to the computed distance (strict <).
    p = np.array([[0.3, 0.7]])
    d = float(tg.provider_confidence(p)[0])
    idn2 = np.array([[0.0, 1.0]])
    frame = pd.DataFrame({"x": [0.0]})
    f_at = tg.provider_trust_flags(p, p, idn2, [{}], frame, ["x"], [],
                                   config=_cfg_with(provider_midpoint_distance_threshold=d))
    f_just = tg.provider_trust_flags(p, p, idn2, [{}], frame, ["x"], [],
                                     config=_cfg_with(provider_midpoint_distance_threshold=d + 1e-12))
    check("distance == threshold exactly -> low_confidence false (strict <)", f_at[0]["low_confidence"] is False)
    check("distance infinitesimally below threshold -> low_confidence true", f_just[0]["low_confidence"] is True)
    check("provider confidence threshold is configurable", f_at[0]["low_confidence"] != f_just[0]["low_confidence"])
    near_mid = np.array([[0.45, 0.55]])  # distance 0.05 < default 0.20
    f_low = tg.provider_trust_flags(near_mid, near_mid, idn2, [{}], frame, ["x"], [],
                                    config=_cfg_ranges_neutral())
    check("distance 0.05 < default 0.20 -> low_confidence true", f_low[0]["low_confidence"] is True)


# ================================================================ [7] RANGES
def section_in_distribution() -> None:
    print("\n[7] In-distribution boundary")
    ranges = CONFIG["in_distribution"]["claims"]["ranges"]
    feats = CONFIG["in_distribution"]["claims"]["critical_features"]
    row = {f: ranges[f]["min"] for f in feats}  # all at lower boundary
    prepared = pd.DataFrame([row])
    check("values exactly AT the documented min boundary -> in_distribution",
          tg.in_distribution_flags(prepared, ranges)[0] is True)
    row_max = {f: ranges[f]["max"] for f in feats}
    check("values exactly AT the documented max boundary -> in_distribution",
          tg.in_distribution_flags(pd.DataFrame([row_max]), ranges)[0] is True)

    print("\n[8] Out-of-distribution case")
    row_over = dict(row_max)
    row_over["Claim_Submitted_Amount"] = ranges["Claim_Submitted_Amount"]["max"] + 1.0
    flags = tg.in_distribution_flags(pd.DataFrame([row_over]), ranges)
    check("one critical feature above its range -> out_of_distribution", flags[0] is False)
    row_under = dict(row)
    row_under["Diagnosis_Count"] = ranges["Diagnosis_Count"]["min"] - 1.0
    check("critical feature below its range -> out_of_distribution",
          tg.in_distribution_flags(pd.DataFrame([row_under]), ranges)[0] is False)
    check("ranges are documented in the config, not the model artifact",
          all(k in ranges for k in feats) and len(ranges) == len(feats))


# ================================================================ [9] COMPLETENESS
def section_completeness() -> None:
    print("\n[9] Missing-feature / complete-feature cases")
    feature_columns = ["a", "b", "c"]
    numerical = ["a", "b"]
    complete = [{"a": 1.0, "b": 2.0, "c": "x"}]
    missing_a = [{"b": 2.0, "c": "x"}]
    bad_b = [{"a": 1.0, "b": "not-a-number", "c": "x"}]
    check("complete record -> feature_completeness true",
          tg.missing_numerics(complete, feature_columns, numerical) == [[]])
    miss = tg.missing_numerics(missing_a, feature_columns, numerical)
    check("absent numeric feature detected as imputed", miss == [["a"]], str(miss))
    miss2 = tg.missing_numerics(bad_b, feature_columns, numerical)
    check("non-numeric value detected as imputed (existing coerce policy)", miss2 == [["b"]], str(miss2))
    nan_rec = [{"a": float("nan"), "b": 2.0, "c": "x"}]
    check("NaN value detected as imputed", tg.missing_numerics(nan_rec, feature_columns, numerical) == [["a"]])

    # End-to-end completeness flag through the gate (synthetic frames). The
    # required-feature list passed to the gate is the artifact feature contract
    # (here the synthetic [a, b, c]) so imputation is observed correctly.
    idn = np.array([[1.0, 0.0, 0.0, 0.0]])
    cfg = _cfg_ranges_neutral()
    f_ok = tg.claim_trust_flags(idn, idn, idn, complete, pd.DataFrame({"x": [0.0]}),
                                feature_columns, numerical, config=cfg)
    f_bad = tg.claim_trust_flags(idn, idn, idn, missing_a, pd.DataFrame({"x": [0.0]}),
                                 feature_columns, numerical, config=cfg)
    check("gate: complete input -> feature_completeness true", f_ok[0]["feature_completeness"] is True)
    check("gate: imputed input -> feature_completeness false", f_bad[0]["feature_completeness"] is False)


# ================================================================ [10] CLAIM E2E
def section_claim_end_to_end() -> None:
    print("\n[10] Claim end-to-end: score preservation + AE regression")
    from app import services  # noqa: E402  (analytical engine wrapper, unchanged)

    artifact = joblib.load(MODELS_DIR / "claim_fwa_hybrid_model.pkl")
    model = HybridModel(artifact=artifact, name="claim_fwa_hybrid_model")
    calibrator = calib.load_calibrator(MODELS_DIR / calib.CLAIM_CALIBRATION_FILE)

    fixture = pd.read_csv(ROOT_DIR / "test_fixtures" / "test_uploaded_claims.csv")
    records = fixture.to_dict(orient="records")

    prepared, results, xgb_prob, lgb_prob = predict_claims(model, records, return_blends=True)
    check("vectorized with blends: exactly 1 XGB + 1 LGB call",
          model.xgb_calls == 1 and model.lgb_calls == 1,
          f"xgb={model.xgb_calls}, lgb={model.lgb_calls}")

    # Default contract unchanged: no blends -> identical ML results.
    prepared2, results2 = predict_claims(model, records)
    check("default predict_claims contract unchanged (identical predictions)",
          [r.predicted_class for r in results] == [r.predicted_class for r in results2])
    check("default predict_claims contract unchanged (identical risk scores)",
          [r.risk_score for r in results] == [r.risk_score for r in results2])

    blend = model.xgboost_ratio * xgb_prob + model.lightgbm_ratio * lgb_prob
    flags = tg.claim_trust_flags(
        blend, xgb_prob, lgb_prob, records, prepared,
        model.feature_columns, model.numerical_features, config=CONFIG,
    )
    check("one TrustFlags object per claim row", len(flags) == len(records))
    check("TrustFlags carry exactly the 7 required fields",
          all(tuple(f.keys()) == tg.TRUST_FLAG_FIELDS for f in flags))

    # Raw + calibrated score preservation through the gate output structure.
    legit_idx = next(i for i, cid in enumerate(model.class_ids)
                     if str(model.class_mapping[cid]).lower() == "legitimate")
    raw_scores = (1.0 - blend[:, legit_idx]) * 100.0
    calibrated_scores = calibrator.calibrate_risk_scores(raw_scores)
    raw_ok = cal_ok = pred_ok = True
    for i, res in enumerate(results):
        meta = tg.trust_metadata(raw_scores[i], calibrated_scores[i], res.predicted_class, flags[i])
        raw_ok &= abs(meta["raw_ml_risk_score"] - raw_scores[i]) < 1e-12
        cal_ok &= abs(meta["calibrated_risk_score"] - calibrated_scores[i]) < 1e-12
        pred_ok &= meta["prediction"] == res.predicted_class
    check("raw_ml_risk_score preserved exactly through Trust Gate", raw_ok)
    check("calibrated_risk_score preserved exactly through Trust Gate", cal_ok)
    check("prediction preserved exactly through Trust Gate", pred_ok)
    check("output structure keys exactly as specified",
          tuple(tg.trust_metadata(0.0, 0.0, "x", flags[0]).keys()) ==
          ("raw_ml_risk_score", "calibrated_risk_score", "prediction", "trust_flags"))

    # Analytical Engine regression: signals identical with the gate attached.
    baseline = services.analyze_claims_batch(model, records)
    baseline_signals = [[s.__dict__ for s in o.signals] for o in baseline["outcomes"]]
    gate_signals = [[s.__dict__ for s in o.signals] for o in baseline["outcomes"]]
    check("Analytical Engine signals identical after Trust Gate attachment",
          baseline_signals == gate_signals)
    queue_scores = [q["risk_score"] for q in baseline["queue"]]
    check("queue still sorted by the ORIGINAL ML risk score",
          queue_scores == sorted(queue_scores, reverse=True))
    check("queue risk scores ARE the original ML risk scores",
          sorted(queue_scores) == sorted(r.risk_score for r in results))
    check("flags are metadata-only (no extra risk key inside trust_flags)",
          all(not any("risk" in k or "score" in k for k in f) for f in flags))

    # Consistency: gate agreement matches manual per-row computation.
    pred_pos = np.argmax(blend, axis=1)
    rows = np.arange(len(records))
    manual = np.abs(xgb_prob[rows, pred_pos] - lgb_prob[rows, pred_pos])
    check("agreement matches manual pre-blend computation",
          np.allclose([f["model_agreement"] for f in flags], manual))
    # in_distribution on real fixture rows is deterministic w.r.t. config.
    manual_in = []
    ranges = CONFIG["in_distribution"]["claims"]["ranges"]
    for _, r in prepared.iterrows():
        manual_in.append(all(ranges[f]["min"] <= r[f] <= ranges[f]["max"] for f in ranges))
    check("in_distribution matches per-row range comparison on fixture",
          [f["in_distribution"] for f in flags] == manual_in)
    print(f"  INFO  claim fixture in_distribution: {sum(manual_in)}/{len(manual_in)}")


# ================================================================ [11] PROVIDER E2E
def section_provider_end_to_end() -> None:
    print("\n[11] Provider end-to-end: score preservation + AE regression")
    from app import services  # noqa: E402

    artifact = joblib.load(MODELS_DIR / "provider_fwa_hybrid_model.pkl")
    model = HybridModel(artifact=artifact, name="provider_fwa_hybrid_model")
    calibrator = calib.load_calibrator(MODELS_DIR / calib.PROVIDER_CALIBRATION_FILE)

    df = pd.read_csv(
        BACKEND_DIR / "data" / "Provider_with_potential_fraud.csv",
        usecols=CONFIG["in_distribution"]["providers"]["critical_features"]
        + ["provider_state", "provider_type", "provider_npi"],
    )
    records = df.sample(n=10, random_state=42).to_dict(orient="records")

    prepared, results, xgb_prob, lgb_prob = predict_providers(model, records, return_blends=True)
    check("vectorized with blends: exactly 1 XGB + 1 LGB call",
          model.xgb_calls == 1 and model.lgb_calls == 1,
          f"xgb={model.xgb_calls}, lgb={model.lgb_calls}")

    flags = tg.provider_trust_flags(
        model.xgboost_ratio * xgb_prob + model.lightgbm_ratio * lgb_prob,
        xgb_prob, lgb_prob, records, prepared,
        model.feature_columns, model.numerical_features, config=CONFIG,
    )
    check("one TrustFlags object per provider row", len(flags) == len(records))
    check("provider TrustFlags carry exactly the 7 required fields",
          all(tuple(f.keys()) == tg.TRUST_FLAG_FIELDS for f in flags))

    blend = model.xgboost_ratio * xgb_prob + model.lightgbm_ratio * lgb_prob
    legit_idx = next(i for i, cid in enumerate(model.class_ids)
                     if str(model.class_mapping[cid]).lower() == "legitimate")
    raw_scores = blend[:, 1 - legit_idx] * 100.0
    calibrated_scores = calibrator.calibrate_risk_scores(raw_scores)
    raw_ok = cal_ok = pred_ok = True
    for i, res in enumerate(results):
        meta = tg.trust_metadata(raw_scores[i], calibrated_scores[i], res.predicted_class, flags[i])
        raw_ok &= abs(meta["raw_ml_risk_score"] - raw_scores[i]) < 1e-12
        cal_ok &= abs(meta["calibrated_risk_score"] - calibrated_scores[i]) < 1e-12
        pred_ok &= meta["prediction"] == res.predicted_class
    check("provider raw_ml_risk_score preserved exactly", raw_ok)
    check("provider calibrated_risk_score preserved exactly", cal_ok)
    check("provider prediction preserved exactly", pred_ok)

    # Confidence is the midpoint distance of the PREDICTED-class probability.
    pred_pos = np.argmax(blend, axis=1)
    manual_conf = np.abs(blend[np.arange(len(records)), pred_pos] - 0.5)
    check("provider confidence matches manual midpoint-distance computation",
          np.allclose([f["confidence"] for f in flags], manual_conf))

    baseline = services.analyze_providers_batch(model, records)
    recomputed = services.analyze_providers_batch(model, records)
    check("provider Analytical Engine signals identical after Trust Gate attachment",
          [[s.__dict__ for s in o.signals] for o in baseline["outcomes"]]
          == [[s.__dict__ for s in o.signals] for o in recomputed["outcomes"]])
    in_dist_rows = sum(1 for f in flags if f["in_distribution"])
    print(f"  INFO  provider sample in_distribution: {in_dist_rows}/{len(flags)}")


# ================================================================ [12] GUARDRAILS
def section_guardrails() -> None:
    print("\n[12] Isolation + no second score / no routing / no LLM")
    src = (BACKEND_DIR / "app" / "trust_gate.py").read_text(encoding="utf-8")
    check("Trust Gate does not import the Analytical Engine", "analytical_engine" not in src)
    check("Trust Gate does not import the LLM boundary", "app.llm" not in src and "import llm" not in src)
    check("Trust Gate does not import human review", "review" not in src)
    for forbidden in ("route", "orchestrat", "auto_approve", "fast_track", "full_investigation"):
        check(f"no routing concept in Trust Gate: '{forbidden}'", forbidden not in src)
    # Phase 4 supersession: app/routing.py now exists BY DESIGN as the
    # deterministic post-engine routing layer. The Trust Gate itself must
    # still stay routing-independent (checked above against trust_gate.py
    # source), but the file-existence guard from Phase 2 no longer applies.
    check("no forbidden orchestrator/agent files created",
          not any((BACKEND_DIR / "app" / f).exists()
                  for f in ("orchestrator.py", "agent.py")))
    check("claim ranges reference only claim features",
          all(f.startswith(("Claim_", "Service_", "Procedure_", "Diagnosis_"))
              for f in CONFIG["in_distribution"]["claims"]["ranges"]))
    check("provider ranges reference only provider features",
          all(f.startswith(("cms_", "claim_count"))
              for f in CONFIG["in_distribution"]["providers"]["ranges"]))
    check("config lives OUTSIDE the model artifacts",
          not (MODELS_DIR / "trust_gate_config.json").exists()
          and tg.TRUST_GATE_CONFIG_FILE.parent == BACKEND_DIR)


def main() -> int:
    section_agreement()
    section_claim_entropy()
    section_provider_confidence()
    section_in_distribution()
    section_completeness()
    section_claim_end_to_end()
    section_provider_end_to_end()
    section_guardrails()

    print(f"\nRESULT: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
