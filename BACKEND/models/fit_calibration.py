"""Fit the claim and provider probability calibrators on their HELD-OUT test splits.

Held-out calibration data (identified from the training scripts + artifacts):

Claim model  (train_claim_hybrid.py):
    Temporal split: TRAIN 2015-2021, VALIDATION 2022, TEST 2023.
    SMOTENC was applied ONLY to training rows; the 2023 test rows were never
    resampled and never used for ensemble-ratio selection.
    -> calibration split = Claim_Year == 2023 rows of claimsfinal_with_target.csv
       (local label column is named 'target'; training script called it
       'Target_Label' — same integer labels 0..3).

Provider model (train_provider_hybrid.py):
    Stratified 70/15/15 with random_state=42; TEST untouched until final
    evaluation. The split is exactly reproducible with the same
    train_test_split chain, so the calibration split is reconstructed by
    replaying it on row indices (never retraining anything).
    -> calibration split = the reproduced 15% test rows.

Both calibrators are fitted STRICTLY on these held-out rows. Targets used:

    claim    : y = (Target_Label != 0)   [P(non-legitimate) = 1 - P(Legitimate)]
    provider : y = (Potential Fraud == 1) [P(Suspicious)]

Method selection: each held-out split is divided 60% fit / 40% selection
(both parts held out from model training); isotonic and Platt are fitted on
the fit part, the method with lower Brier score on the selection part wins,
and the winning method is refit on the FULL held-out split for the shipped
versioned artifact.

Outputs:
    models/claim_calibration_v1.pkl
    models/provider_calibration_v1.pkl
    models/reports/*.{csv,json}

Run from BACKEND/:  python models/fit_calibration.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from analytical_engine.inference import ModelRegistry, _legit_index  # noqa: E402
from trust_engine.calibration import (  # noqa: E402
    CLAIM_CALIBRATION_FILE,
    PROVIDER_CALIBRATION_FILE,
    fit_calibrator,
    save_calibrator,
    _apply,
)

DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = MODELS_DIR / "reports"

CLAIMS_CSV = DATA_DIR / "claimsfinal_with_target.csv"
PROVIDERS_CSV = DATA_DIR / "Provider_with_potential_fraud.csv"

RANDOM_STATE = 42          # identical to both training scripts
CLAIM_TEST_YEAR = 2023     # temporal TEST year from train_claim_hybrid.py
PROVIDER_TARGET = "Potential Fraud"
CLAIM_TARGET = "target"    # local dataset label column ('Target_Label' in training script)
SCORE_CHUNK = 8000         # rows per inference chunk (memory guard, same model calls)


# ------------------------------------------------------------------ metrics
def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width ECE: sum over bins of (n_bin / n) * |mean_prob - observed_freq|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (prob > lo) & (prob <= hi) if i > 0 else (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def bucket_table(y_true: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency by decile bucket."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (raw > lo) & (raw <= hi) if i > 0 else (raw >= lo) & (raw <= hi)
        n = int(mask.sum())
        rows.append({
            "bucket": f"({lo:.1f}, {hi:.1f}]" if i > 0 else f"[{lo:.1f}, {hi:.1f}]",
            "n": n,
            "raw_prob_mean": float(raw[mask].mean()) if n else None,
            "observed_frequency": float(y_true[mask].mean()) if n else None,
            "calibrated_prob_mean": float(calibrated[mask].mean()) if n else None,
        })
    return pd.DataFrame(rows)


def metrics_block(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict:
    return {
        "raw_brier": float(brier_score_loss(y, raw)),
        "calibrated_brier": float(brier_score_loss(y, calibrated)),
        "raw_ece": expected_calibration_error(y, raw),
        "calibrated_ece": expected_calibration_error(y, calibrated),
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
    }


def raw_vs_calibrated_examples(raw: np.ndarray, calibrate) -> list[dict]:
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    out = []
    for q in qs:
        r = float(np.quantile(raw, q))
        c = float(calibrate(np.array([r]))[0])
        out.append({"quantile": q, "raw_probability": round(r, 6),
                    "calibrated_probability": round(c, 6),
                    "raw_risk_score": round(r * 100, 2),
                    "calibrated_risk_score": round(c * 100, 2)})
    return out


# ------------------------------------------------------------ scoring helper
def hybrid_probability(model, records: pd.DataFrame, positive: str) -> np.ndarray:
    """Blended probability of the positive class via the LOCKED artifact path.

    Uses the artifact's own prepare_frame + predict_proba (one XGB + one LGB
    call per chunk), chunked only to bound memory. No model parameter or
    preprocessing is altered.
    """
    legit = _legit_index(model)
    probs = []
    for start in range(0, len(records), SCORE_CHUNK):
        chunk = records.iloc[start:start + SCORE_CHUNK]
        prepared = model.prepare_frame(chunk)
        proba = model.predict_proba(prepared)
        if positive == "non_legitimate":
            probs.append(1.0 - proba[:, legit])
        else:  # 'suspicious' — binary artifact, Suspicious is the non-legit column
            probs.append(proba[:, 1 - legit])
    return np.concatenate(probs)


# ------------------------------------------------------------ method select
def select_and_fit(raw_p: np.ndarray, y: np.ndarray):
    """60% fit / 40% selection (both held out from training) -> winning method.

    Returns (selected_method, selection_metrics_for_both_methods).
    """
    idx_fit, idx_sel = train_test_split(
        np.arange(len(y)), test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    comparison = {}
    for method in ("isotonic", "sigmoid"):
        cal = fit_calibrator(method, raw_p[idx_fit], y[idx_fit])
        calib_sel = _apply(cal, method, raw_p[idx_sel])
        comparison[method] = {
            "selection_brier": float(brier_score_loss(y[idx_sel], calib_sel)),
            "selection_ece": expected_calibration_error(y[idx_sel], calib_sel),
        }
    selected = min(comparison, key=lambda m: (comparison[m]["selection_brier"], comparison[m]["selection_ece"]))
    return selected, comparison, idx_fit, idx_sel


# ================================================================== CLAIMS
def fit_claims(registry: ModelRegistry) -> None:
    print("\n" + "=" * 78)
    print("CLAIM CALIBRATION — held-out temporal TEST split (Claim_Year == 2023)")
    print("=" * 78)

    model = registry.claim
    cols = model.feature_columns + ["Claim_Year", CLAIM_TARGET, "Claim_ID"]
    df = pd.read_csv(CLAIMS_CSV, usecols=cols, low_memory=False)
    df["Claim_Year"] = pd.to_numeric(df["Claim_Year"], errors="coerce").astype("Int64")
    test_df = df[df["Claim_Year"] == CLAIM_TEST_YEAR].reset_index(drop=True)
    print(f"Held-out calibration rows (2023): {len(test_df):,}")

    y = (test_df[CLAIM_TARGET].astype(int) != 0).to_numpy(dtype=np.int8)
    features = test_df[model.feature_columns]
    raw_p = hybrid_probability(model, features, "non_legitimate")
    print(f"Raw P(non-legitimate): mean={raw_p.mean():.4f} max={raw_p.max():.4f}")

    selected, comparison, idx_fit, idx_sel = select_and_fit(raw_p, y)
    print(f"Method comparison on selection part: {json.dumps(comparison, indent=2)}")
    print(f"Selected method: {selected}")

    # Final artifact refit on the FULL held-out split.
    calibrator = fit_calibrator(selected, raw_p, y)
    calibrated = _apply(calibrator, selected, raw_p)
    whole_metrics = metrics_block(y, raw_p, calibrated)
    sel_metrics = metrics_block(
        y[idx_sel], raw_p[idx_sel],
        _apply(fit_calibrator(selected, raw_p[idx_fit], y[idx_fit]), selected, raw_p[idx_sel]),
    )

    class_counts = test_df[CLAIM_TARGET].astype(int).value_counts().sort_index().to_dict()
    bundle = {
        "calibrator": calibrator,
        "method": selected,
        "model_name": "claim_fwa_hybrid_model",
        "calibration_version": "claim-calibration-v1",
        "calibrated_quantity": "P(Fraud or Waste or Abuse) = 1 - P(Legitimate)",
        "raw_risk_definition": "(1 - P(Legitimate)) * 100 — UNCHANGED, preserved as raw_ml_risk_score",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "source_split": {
            "role": "test",
            "dataset": "data/claimsfinal_with_target.csv",
            "split_definition": f"temporal split Claim_Year == {CLAIM_TEST_YEAR} (train 2015-2021, validation 2022)",
            "held_out_from_training": True,
            "smote_applied_to_split": False,
            "used_for_ensemble_ratio_selection": False,
            "n_samples": int(len(test_df)),
            "class_counts_Target_Label": {str(k): int(v) for k, v in class_counts.items()},
            "positive_rate_non_legitimate": float(y.mean()),
            "label_column_in_local_dataset": CLAIM_TARGET,
        },
        "method_selection": {
            "fit_fraction": 0.60,
            "selection_fraction": 0.40,
            "candidates": comparison,
            "selected_by": "lowest Brier score on the untouched 40% selection part",
        },
        "fit_metrics_whole_heldout_split": whole_metrics,
        "selection_part_metrics": sel_metrics,
        "raw_vs_calibrated_examples": raw_vs_calibrated_examples(raw_p, lambda p: _apply(calibrator, selected, p)),
    }
    save_calibrator(bundle, MODELS_DIR / CLAIM_CALIBRATION_FILE)
    print(f"Saved {MODELS_DIR / CLAIM_CALIBRATION_FILE}")

    # ------------------------------------------------------------- reports
    table = bucket_table(y, raw_p, calibrated)
    table.to_csv(REPORTS_DIR / "claim_calibration_report.csv", index=False)
    print("\nClaim calibration report (deciles of raw probability):")
    print(table.to_string(index=False))

    with open(REPORTS_DIR / "claim_calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "calibrator"}, f, indent=2)

    # ---------------- Diagnostic one-vs-rest calibration (Fraud/Waste/Abuse)
    # Diagnostic only: production risk remains 1 - P(Legitimate).
    proba_full = []
    for start in range(0, len(features), SCORE_CHUNK):
        chunk = features.iloc[start:start + SCORE_CHUNK]
        proba_full.append(model.predict_proba(model.prepare_frame(chunk)))
    proba_full = np.concatenate(proba_full)
    mapping = model.class_mapping
    ids = model.class_ids
    labels = test_df[CLAIM_TARGET].astype(int).to_numpy()
    ovr = {}
    for cid, name in mapping.items():
        if str(name).lower() == "legitimate":
            continue
        col = ids.index(cid)
        raw_c = proba_full[:, col]
        y_c = (labels == cid).astype(np.int8)
        cal_c = fit_calibrator(selected, raw_c[idx_fit], y_c[idx_fit])
        calib_c = _apply(cal_c, selected, raw_c[idx_sel])
        ovr[name] = {
            "class_id": int(cid),
            "positive_rate": float(y_c.mean()),
            "raw_brier_selection": float(brier_score_loss(y_c[idx_sel], raw_c[idx_sel])),
            "calibrated_brier_selection": float(brier_score_loss(y_c[idx_sel], calib_c)),
            "raw_ece_selection": expected_calibration_error(y_c[idx_sel], raw_c[idx_sel]),
            "calibrated_ece_selection": expected_calibration_error(y_c[idx_sel], calib_c),
            "note": "diagnostic one-vs-rest only; NOT used by the production risk score",
        }
    with open(REPORTS_DIR / "claim_ovr_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(ovr, f, indent=2)
    print("\nClaim one-vs-rest diagnostics (selection part):")
    print(json.dumps(ovr, indent=2))


# ================================================================ PROVIDERS
def fit_providers(registry: ModelRegistry) -> None:
    print("\n" + "=" * 78)
    print("PROVIDER CALIBRATION — reproduced stratified 70/15/15 TEST split (rs=42)")
    print("=" * 78)

    # Reproduce the exact training-script split on row indices.
    y_all = pd.read_csv(PROVIDERS_CSV, usecols=[PROVIDER_TARGET])[PROVIDER_TARGET].astype(np.int8).to_numpy()
    n = len(y_all)
    idx_all = np.arange(n)
    _, idx_temp = train_test_split(idx_all, test_size=0.30, stratify=y_all, random_state=RANDOM_STATE)
    idx_train_unused, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=y_all[idx_temp], random_state=RANDOM_STATE
    )
    is_test = np.zeros(n, dtype=bool)
    is_test[idx_test] = True
    print(f"Held-out calibration rows (test): {int(is_test.sum()):,} of {n:,}")

    model = registry.provider
    cols = model.feature_columns + [PROVIDER_TARGET, "provider_npi"]
    frames = []
    offset = 0
    for chunk in pd.read_csv(PROVIDERS_CSV, usecols=cols, chunksize=100_000, low_memory=False):
        mask = is_test[offset:offset + len(chunk)]
        if mask.any():
            frames.append(chunk[mask])
        offset += len(chunk)
    test_df = pd.concat(frames, ignore_index=True)
    assert len(test_df) == int(is_test.sum()), "split reproduction mismatch"

    y = test_df[PROVIDER_TARGET].astype(np.int8).to_numpy()
    features = test_df[model.feature_columns]
    raw_p = hybrid_probability(model, features, "suspicious")
    print(f"Raw P(Suspicious): mean={raw_p.mean():.4f} max={raw_p.max():.4f}")

    selected, comparison, idx_fit, idx_sel = select_and_fit(raw_p, y)
    print(f"Method comparison on selection part: {json.dumps(comparison, indent=2)}")
    print(f"Selected method: {selected}")

    calibrator = fit_calibrator(selected, raw_p, y)
    calibrated = _apply(calibrator, selected, raw_p)
    whole_metrics = metrics_block(y, raw_p, calibrated)
    sel_metrics = metrics_block(
        y[idx_sel], raw_p[idx_sel],
        _apply(fit_calibrator(selected, raw_p[idx_fit], y[idx_fit]), selected, raw_p[idx_sel]),
    )

    bundle = {
        "calibrator": calibrator,
        "method": selected,
        "model_name": "provider_fwa_hybrid_model",
        "calibration_version": "provider-calibration-v1",
        "calibrated_quantity": "P(Suspicious)",
        "raw_risk_definition": "P(Suspicious) * 100 — UNCHANGED, preserved as raw_ml_risk_score",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "source_split": {
            "role": "test",
            "dataset": "data/Provider_with_potential_fraud.csv",
            "split_definition": "stratified 70/15/15 (train_test_split test_size=0.30 then 0.50, random_state=42) — TEST part",
            "held_out_from_training": True,
            "used_for_ensemble_ratio_selection": False,
            "n_samples": int(len(test_df)),
            "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
            "positive_rate_suspicious": float(y.mean()),
        },
        "method_selection": {
            "fit_fraction": 0.60,
            "selection_fraction": 0.40,
            "candidates": comparison,
            "selected_by": "lowest Brier score on the untouched 40% selection part",
        },
        "fit_metrics_whole_heldout_split": whole_metrics,
        "selection_part_metrics": sel_metrics,
        "raw_vs_calibrated_examples": raw_vs_calibrated_examples(raw_p, lambda p: _apply(calibrator, selected, p)),
    }
    save_calibrator(bundle, MODELS_DIR / PROVIDER_CALIBRATION_FILE)
    print(f"Saved {MODELS_DIR / PROVIDER_CALIBRATION_FILE}")

    table = bucket_table(y, raw_p, calibrated)
    table.to_csv(REPORTS_DIR / "provider_calibration_report.csv", index=False)
    print("\nProvider calibration report (deciles of raw probability):")
    print(table.to_string(index=False))

    with open(REPORTS_DIR / "provider_calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "calibrator"}, f, indent=2)


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(
        MODELS_DIR / "claim_fwa_hybrid_model.pkl",
        MODELS_DIR / "provider_fwa_hybrid_model.pkl",
    )
    fit_claims(registry)
    fit_providers(registry)
    print("\nCalibration fitting complete. Raw ML models and risk scores were NOT modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
