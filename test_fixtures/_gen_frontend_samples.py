"""One-off generator for frontend batch-test sample CSVs.

Trims the already-verified upload fixtures down small, human-readable samples
that still cover all three deterministic routes, then validates each sample
through the SAME parser the upload endpoint uses (app.uploads) so the files
are guaranteed uploadable in the frontend.

Outputs:
  test_fixtures/sample_claims_frontend.csv
  test_fixtures/sample_providers_frontend.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent / "BACKEND"
sys.path.insert(0, str(BACKEND))

from analytical_engine.inference import ModelRegistry  # noqa: E402
from app import config, services, uploads  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "test_fixtures"


def select_covering(frame: pd.DataFrame, analyze, model, max_rows: int) -> pd.DataFrame:
    """Pick up to max_rows rows that cover as many routes as possible."""
    scored = []
    for idx, row in frame.iterrows():
        rec = row.to_dict()
        result = analyze(model, rec)
        scored.append((result.upstream_context["routing"]["route"], idx))
    # one row per route first, then fill remaining slots
    picked: list[int] = []
    seen: set[str] = set()
    for route, idx in scored:
        if route not in seen:
            picked.append(idx)
            seen.add(route)
    for route, idx in scored:
        if len(picked) >= max_rows:
            break
        if idx not in picked:
            picked.append(idx)
    return frame.loc[picked[:max_rows]].reset_index(drop=True)


def main() -> None:
    registry = ModelRegistry(config.CLAIM_MODEL_PATH, config.PROVIDER_MODEL_PATH)
    claim_model = registry.claim
    provider_model = registry.provider

    claims = pd.read_csv(FIXTURES / "test_uploaded_claims.csv")
    providers = pd.read_csv(FIXTURES / "test_uploaded_providers.csv")

    claim_sample = select_covering(claims, services.analyze_claim, claim_model, 8)
    provider_sample = select_covering(providers, services.analyze_provider, provider_model, 4)

    # The 25-row claim fixture has no fast_track case; splice in the verified
    # Phase 5 fast-track row so the frontend sample covers all three routes.
    covered = {
        services.analyze_claim(claim_model, rec).upstream_context["routing"]["route"]
        for rec in claim_sample.to_dict(orient="records")
    }
    if "fast_track" not in covered:
        phase5 = json.loads((FIXTURES / "phase5_cases.json").read_text())
        fast_row = pd.DataFrame([phase5["claims"]["fast_track"]["features"]])
        claim_sample = pd.concat([claim_sample, fast_row], ignore_index=True)

    claim_path = FIXTURES / "sample_claims_frontend.csv"
    provider_path = FIXTURES / "sample_providers_frontend.csv"
    claim_sample.to_csv(claim_path, index=False)
    provider_sample.to_csv(provider_path, index=False)

    # Validate through the EXACT upload parser the endpoints use.
    parsed_claims = uploads.parse_claim_upload(
        claim_path.read_bytes(), claim_path.name, "text/csv", claim_model
    )
    parsed_providers = uploads.parse_provider_upload(
        provider_path.read_bytes(), provider_path.name, "text/csv", provider_model
    )
    assert len(parsed_claims) == len(claim_sample)
    assert len(parsed_providers) == len(provider_sample)

    print(f"claims sample: {len(claim_sample)} rows -> {claim_path.name}")
    for rec in parsed_claims:
        route = services.analyze_claim(claim_model, rec).upstream_context["routing"]["route"]
        print(f"  Claim_ID={rec['Claim_ID']}  route={route}")
    print(f"providers sample: {len(provider_sample)} rows -> {provider_path.name}")
    for rec in parsed_providers:
        route = services.analyze_provider(provider_model, rec).upstream_context["routing"]["route"]
        print(f"  provider_npi={rec['provider_npi']}  route={route}")
    print("Both samples parse cleanly through app.uploads (upload-endpoint contract).")


if __name__ == "__main__":
    main()
