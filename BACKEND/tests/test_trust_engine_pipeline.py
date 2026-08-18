from __future__ import annotations

from analytical_engine.inference import ModelRegistry, predict_claims
from app import config
from trust_engine.pipeline import evaluate


def test_trust_engine_pipeline_output_shape() -> None:
    registry = ModelRegistry(config.CLAIM_MODEL_PATH, config.PROVIDER_MODEL_PATH)
    prepared, results, xgb_prob, lgb_prob = predict_claims(registry.claim, [{}], return_blends=True)

    blend = registry.claim.xgboost_ratio * xgb_prob + registry.claim.lightgbm_ratio * lgb_prob
    legit_idx = next(
        i for i, cid in enumerate(registry.claim.class_ids)
        if registry.claim.class_mapping[cid].lower() == "legitimate"
    )
    raw_risk = float((1.0 - blend[0, legit_idx]) * 100.0)

    out = evaluate(
        {
            "entity_type": "claims",
            "prediction": results[0].predicted_class,
            "raw_risk_score": raw_risk,
            "raw_xgb_probabilities": xgb_prob[0].tolist(),
            "raw_lgb_probabilities": lgb_prob[0].tolist(),
            "blended_probabilities": blend[0].tolist(),
        },
        {
            **prepared.iloc[0].to_dict(),
            "imputed_features": [],
        },
    )

    assert set(out.keys()) == {"calibrated_score", "prediction", "trust_flags"}
    assert isinstance(out["calibrated_score"], float)
    assert isinstance(out["prediction"], str)

    flags = out["trust_flags"]
    assert set(flags.keys()) >= {
        "model_agreement",
        "low_agreement",
        "confidence_entropy",
        "low_confidence",
        "in_distribution",
        "feature_completeness",
    }
