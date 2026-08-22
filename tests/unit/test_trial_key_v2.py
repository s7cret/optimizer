from dataclasses import replace
from decimal import Decimal

import pytest
from openpine_contracts import (
    CanonicalizationError,
    canonical_dumps,
    verify_content_hash,
)

from optimizer.core.trial_key import (
    TRIAL_SCHEMA_ID,
    TrialIdentity,
    contract_schema_hashes,
    normalize_parameters,
)


def _identity() -> TrialIdentity:
    return TrialIdentity(
        generated_artifact_hash="sha256:artifact",
        data_snapshot_series_hash="sha256:series",
        parameters={
            "length": 10,
            "threshold": 0.125,
            "nested": [Decimal("1.500"), True],
        },
        engine_build_hash="sha256:engine-build",
        engine_config_hash="sha256:engine-config",
        semantic_profile="strict_5x",
        finality_policy={"bars": "FINAL"},
        warmup_policy={"mode": "CALC_ONLY", "bars": 20},
        score_policy={"window": "closed_range", "epsilon": 1e-12},
        end_policy={"mode": "liquidate"},
        contract_schema_hashes={
            "openpine.generated_artifact.v2": "sha256:generated-schema",
            "openpine.trial.v2": "sha256:trial-schema",
        },
        stack_manifest_hash="sha256:stack",
        deterministic_seed=42,
        fold_identity={"fold": 2, "range": [100, 200]},
        walk_forward_identity={"window": 3, "phase": "test"},
        objective_version="net-profit.v2",
        constraints_version="risk-limits.v3",
    )


def _contains_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_float(item) for item in value)
    return False


def test_trial_key_v2_seals_full_canonical_identity_without_float_boundary() -> None:
    key = _identity().seal()

    assert key.trial_key.startswith("sha256:")
    assert key.payload["schema_id"] == TRIAL_SCHEMA_ID
    assert key.payload["content_hash"] == key.trial_key
    assert verify_content_hash(key.payload)
    assert not _contains_float(key.payload)
    assert key.payload["normalized_parameters"] == {
        "length": 10,
        "nested": ["1.5", True],
        "threshold": "0.125",
    }
    assert canonical_dumps(key.payload) == key.canonical_json

    reordered = replace(
        _identity(),
        parameters={
            "nested": [Decimal("1.5000"), True],
            "threshold": 0.125,
            "length": 10,
        },
        contract_schema_hashes={
            "openpine.trial.v2": "sha256:trial-schema",
            "openpine.generated_artifact.v2": "sha256:generated-schema",
        },
    ).seal()
    assert reordered.trial_key == key.trial_key


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generated_artifact_hash", "sha256:other-artifact"),
        ("data_snapshot_series_hash", "sha256:other-series"),
        ("parameters", {"length": 11}),
        ("engine_build_hash", "sha256:other-build"),
        ("engine_config_hash", "sha256:other-config"),
        ("semantic_profile", "legacy_4x"),
        ("finality_policy", {"bars": "OPEN"}),
        ("warmup_policy", {"mode": "CALC_THEN_RESET_BROKER"}),
        ("score_policy", {"window": "all"}),
        ("end_policy", {"mode": "keep_open"}),
        ("contract_schema_hashes", {"openpine.trial.v2": "sha256:other-schema"}),
        ("stack_manifest_hash", "sha256:other-stack"),
        ("deterministic_seed", 7),
        ("fold_identity", {"fold": 9}),
        ("walk_forward_identity", {"window": 9}),
        ("objective_version", "objective.v9"),
        ("constraints_version", "constraints.v9"),
    ],
)
def test_trial_key_v2_changes_when_any_identity_component_changes(
    field: str, value: object
) -> None:
    original = _identity().seal().trial_key
    changed = replace(_identity(), **{field: value}).seal().trial_key
    assert changed != original


def test_parameter_normalization_rejects_unsafe_contract_values() -> None:
    assert normalize_parameters({"zero": -0.0, "tuple": (1, "x"), "none": None}) == {
        "none": None,
        "tuple": [1, "x"],
        "zero": "0",
    }
    with pytest.raises(CanonicalizationError, match="finite"):
        normalize_parameters({"bad": float("nan")})
    with pytest.raises(CanonicalizationError, match="finite"):
        normalize_parameters({"bad": Decimal("NaN")})
    with pytest.raises(CanonicalizationError, match="collide"):
        normalize_parameters({"e\u0301": 1, "é": 2})
    with pytest.raises(CanonicalizationError, match="map keys"):
        normalize_parameters({1: "bad"})  # type: ignore[dict-item]
    with pytest.raises(CanonicalizationError, match="unsupported"):
        normalize_parameters({"bad": object()})


def test_contract_schema_hashes_are_loaded_from_installed_contract_wheel() -> None:
    hashes = contract_schema_hashes()
    assert "openpine.trial.v2" in hashes
    assert all(value.startswith("sha256:") for value in hashes.values())
