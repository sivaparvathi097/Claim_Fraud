"""THROWAWAY: generate Phase 5 LLM-suite fixtures (feature payloads per route).

Writes test_fixtures/phase5_cases.json with one case per route per entity.
Features are derived from REAL dataset rows (training targets excluded) and
each expected route is computed offline via the SAME single-score pipeline the
API uses, so the live suite can POST these features without dataset lookups.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd  # noqa: E402

from analytical_engine.inference import ModelRegistry  # noqa: E402
from app import config, services  # noqa: E402

OUT = BACKEND_DIR.parent / "test_fixtures" / "phase5_cases.json"
TARGET_COLS = {"target", "target_label", "potential fraud"}


def features_of(row: pd.Series) -> dict:
    out = {}
    for col, val in row.to_dict().items():
        if str(col).strip().lower() in TARGET_COLS:
            continue  # training labels never travel as inference input
        if pd.isna(val):
            continue
        out[str(col)] = val.item() if hasattr(val, "item") else val
    return out


def case(kind: str, row: pd.Series, route: str) -> dict:
    feats = features_of(row)
    return {"entity": kind, "expected_route": route, "features": feats}


registry = ModelRegistry(config.CLAIM_MODEL_PATH, config.PROVIDER_MODEL_PATH)
result: dict = {"claims": {}, "providers": {}}

print("reading claims csv...", flush=True)
claims = pd.read_csv(config.CLAIM_DATA_PATH, low_memory=False)
print(f"claims loaded: {claims.shape}", flush=True)
for i in range(min(len(claims), 1500)):
    if len(result["claims"]) == 3:
        break
    rec = claims.iloc[i].to_dict()
    outcome = services.analyze_claim(registry.claim, rec)
    route = outcome.upstream_context["routing"]["route"]
    if route in result["claims"]:
        continue
    entry = case("claim", claims.iloc[i], route)
    # Verify the FEATURES-ONLY payload reproduces the same route (the live
    # suite posts exactly these features).
    check = services.analyze_claim(registry.claim, dict(entry["features"]))
    assert check.upstream_context["routing"]["route"] == route, (i, route)
    entry["calibrated_risk_score"] = outcome.upstream_context["calibrated_risk_score"]
    entry["prediction"] = outcome.upstream_context["prediction"]
    result["claims"][route] = entry
    print(f"claim {route}: row {i} calibrated={entry['calibrated_risk_score']:.2f} "
          f"prediction={entry['prediction']} features={len(entry['features'])}", flush=True)

services._CALIBRATORS.clear()
print("reading provider csv...", flush=True)
providers = pd.read_csv(config.PROVIDER_DATA_PATH, low_memory=False)
print(f"providers loaded: {providers.shape}", flush=True)
for i in range(min(len(providers), 20000)):
    if len(result["providers"]) == 3:
        break
    rec = providers.iloc[i].to_dict()
    try:
        outcome = services.analyze_provider(registry.provider, rec)
    except Exception:  # noqa: BLE001
        continue
    route = outcome.upstream_context["routing"]["route"]
    if route in result["providers"]:
        continue
    entry = case("provider", providers.iloc[i], route)
    check = services.analyze_provider(registry.provider, dict(entry["features"]))
    assert check.upstream_context["routing"]["route"] == route, (i, route)
    entry["calibrated_risk_score"] = outcome.upstream_context["calibrated_risk_score"]
    entry["prediction"] = outcome.upstream_context["prediction"]
    result["providers"][route] = entry
    print(f"provider {route}: row {i} calibrated={entry['calibrated_risk_score']:.2f} "
          f"prediction={entry['prediction']} features={len(entry['features'])}", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
print("written:", OUT, flush=True)
print("claims routes:", sorted(result["claims"].keys()), flush=True)
print("provider routes:", sorted(result["providers"].keys()), flush=True)
