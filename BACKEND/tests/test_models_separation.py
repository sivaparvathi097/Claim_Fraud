from __future__ import annotations

from analytical_engine.inference import ModelRegistry, predict_claims, predict_providers
from app import config


def test_models_output_has_no_trust_engine_fields() -> None:
    registry = ModelRegistry(config.CLAIM_MODEL_PATH, config.PROVIDER_MODEL_PATH)

    _, claim_results = predict_claims(registry.claim, [{}])
    claim = claim_results[0]
    assert not hasattr(claim, "calibrated_score")
    assert not hasattr(claim, "trust_flags")

    _, provider_results = predict_providers(registry.provider, [{}])
    provider = provider_results[0]
    assert not hasattr(provider, "calibrated_score")
    assert not hasattr(provider, "trust_flags")
