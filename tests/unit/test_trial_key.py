from __future__ import annotations

import pytest

from optimizer.core.trial_key import (
    SCHEMA_ID,
    TrialKeyError,
    identity_payload,
    make_trial_key,
    make_trial_record,
    normalize_parameters,
)


def _identity(**overrides):
    payload = {
        "strategy_artifact_hash": "sha256:" + ("a" * 64),
        "snapshot_hash": "sha256:" + ("b" * 64),
        "parameters": {"len": "14", "mult": "2"},
        "engine_version": "5.0.0rc1",
        "runtime_version": "5.0.0rc1",
        "contracts_version": "1.0.0rc1",
        "semantic_profile": "strict_5x",
        "numeric_policy": "decimal_string",
        "fill_policy": "engine_only",
        "warmup_mode": "calc_then_reset_broker",
        "fold_window": {"start": 0, "end": 100},
        "seed": 7,
        "stack_id": "stack-candidate-5.0.0-rc.1",
    }
    payload.update(overrides)
    return payload


def test_trial_key_is_stable_and_order_independent() -> None:
    left = make_trial_key(**_identity(parameters={"mult": "2.0", "len": "14"}))
    right = make_trial_key(**_identity(parameters={"len": "14.0", "mult": "2"}))
    assert left == right
    assert left.startswith("sha256:")


def test_trial_key_changes_when_stack_or_profile_changes() -> None:
    base = make_trial_key(**_identity())
    other_stack = make_trial_key(**_identity(stack_id="other-stack"))
    other_profile = make_trial_key(**_identity(semantic_profile="legacy_4x"))
    assert base != other_stack
    assert base != other_profile


def test_normalize_parameters_rejects_float_and_bool() -> None:
    with pytest.raises(TrialKeyError, match="must not be float"):
        normalize_parameters({"len": 1.5})
    with pytest.raises(TrialKeyError, match="decimal string or int"):
        normalize_parameters({"flag": True})
    with pytest.raises(TrialKeyError, match="unsupported type"):
        normalize_parameters({"bad": ["x"]})
    with pytest.raises(TrialKeyError, match="non-empty strings"):
        normalize_parameters({"": "1"})
    with pytest.raises(TrialKeyError, match="must be a mapping"):
        normalize_parameters(["len"])  # type: ignore[arg-type]


def test_identity_payload_validates_seed_and_window() -> None:
    with pytest.raises(TrialKeyError, match="seed"):
        identity_payload(**_identity(seed=True))
    with pytest.raises(TrialKeyError, match="fold_window"):
        identity_payload(**_identity(fold_window="full"))
    with pytest.raises(TrialKeyError, match="start/end"):
        identity_payload(**_identity(fold_window={"start": "0", "end": "1"}))
    with pytest.raises(TrialKeyError, match="greater than start"):
        identity_payload(**_identity(fold_window={"start": 10, "end": 10}))
    with pytest.raises(TrialKeyError, match="non-empty string"):
        identity_payload(**_identity(stack_id=""))


def test_make_trial_key_incomplete_kwargs() -> None:
    payload = _identity()
    payload.pop("stack_id")
    with pytest.raises(TrialKeyError, match="incomplete"):
        make_trial_key(**payload)


def test_make_trial_record_validates_and_hashes() -> None:
    record = make_trial_record(
        producer_version="5.0.0rc1",
        producer_commit="c" * 40,
        created_at_utc_ms=1,
        lifecycle="completed",
        trial_identity=_identity(),
        metrics={"net_profit": "1.25"},
    )
    assert record["schema_id"] == SCHEMA_ID
    assert record["trial_key"].startswith("sha256:")
    assert record["content_hash"].startswith("sha256:")
    assert record["parameters"] == {"len": "14", "mult": "2"}
    same = make_trial_record(
        producer_version="5.0.0rc1",
        producer_commit="c" * 40,
        created_at_utc_ms=1,
        lifecycle="completed",
        trial_identity=_identity(),
        metrics={"net_profit": "1.25"},
    )
    assert record["content_hash"] == same["content_hash"]


def test_make_trial_record_rejects_bad_envelope_fields() -> None:
    identity = _identity()
    with pytest.raises(TrialKeyError, match="created_at_utc_ms"):
        make_trial_record(
            producer_version="5.0.0rc1",
            producer_commit="c" * 40,
            created_at_utc_ms=-1,
            lifecycle="queued",
            trial_identity=identity,
        )
    with pytest.raises(TrialKeyError, match="retry_count"):
        make_trial_record(
            producer_version="5.0.0rc1",
            producer_commit="c" * 40,
            created_at_utc_ms=0,
            lifecycle="queued",
            trial_identity=identity,
            retry_count=-1,
        )
