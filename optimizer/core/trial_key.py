"""Deterministic openpine.trial.v2 identity. Core stays on RunnerProtocol."""

from __future__ import annotations

from typing import Any, Mapping

from openpine_contracts import (
    SERIALIZER_ID,
    CanonicalizationError,
    content_hash,
    decimal_string,
    validate_payload,
)

SCHEMA_ID = "openpine.trial.v2"
SCHEMA_VERSION = "2"
PRODUCER = "optimizer"

REQUIRED_IDENTITY_FIELDS = (
    "strategy_artifact_hash",
    "snapshot_hash",
    "parameters",
    "engine_version",
    "runtime_version",
    "contracts_version",
    "semantic_profile",
    "numeric_policy",
    "fill_policy",
    "warmup_mode",
    "fold_window",
    "seed",
    "stack_id",
)


class TrialKeyError(ValueError):
    """Fail-closed TrialKey construction."""


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TrialKeyError(f"{name} must be a non-empty string")
    return value


def normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(parameters, Mapping):
        raise TrialKeyError("parameters must be a mapping")
    out: dict[str, str] = {}
    for raw_key, raw_value in parameters.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise TrialKeyError("parameter names must be non-empty strings")
        if isinstance(raw_value, bool) or raw_value is None:
            raise TrialKeyError(f"parameter {raw_key} must be a decimal string or int")
        if isinstance(raw_value, float):
            raise TrialKeyError(f"parameter {raw_key} must not be float")
        if isinstance(raw_value, int):
            out[raw_key] = decimal_string(raw_value)
            continue
        if isinstance(raw_value, str):
            out[raw_key] = decimal_string(raw_value)
            continue
        raise TrialKeyError(f"parameter {raw_key} has unsupported type")
    return {key: out[key] for key in sorted(out)}


def _normalize_fold_window(fold_window: Any) -> dict[str, str] | None:
    if fold_window is None:
        return None
    if not isinstance(fold_window, Mapping):
        raise TrialKeyError("fold_window must be a mapping or null")
    start = fold_window.get("start")
    end = fold_window.get("end")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or isinstance(start, bool)
        or isinstance(end, bool)
    ):
        raise TrialKeyError("fold_window start/end must be integers")
    if end <= start:
        raise TrialKeyError("fold_window end must be greater than start")
    return {"start": decimal_string(start), "end": decimal_string(end)}


def identity_payload(
    *,
    strategy_artifact_hash: str,
    snapshot_hash: str,
    parameters: Mapping[str, Any],
    engine_version: str,
    runtime_version: str,
    contracts_version: str,
    semantic_profile: str,
    numeric_policy: str,
    fill_policy: str,
    warmup_mode: str,
    fold_window: Mapping[str, Any] | None,
    seed: int | None,
    stack_id: str,
) -> dict[str, Any]:
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TrialKeyError("seed must be an integer or null")
    payload = {
        "strategy_artifact_hash": _require_text(
            "strategy_artifact_hash", strategy_artifact_hash
        ),
        "snapshot_hash": _require_text("snapshot_hash", snapshot_hash),
        "parameters": normalize_parameters(parameters),
        "engine_version": _require_text("engine_version", engine_version),
        "runtime_version": _require_text("runtime_version", runtime_version),
        "contracts_version": _require_text("contracts_version", contracts_version),
        "semantic_profile": _require_text("semantic_profile", semantic_profile),
        "numeric_policy": _require_text("numeric_policy", numeric_policy),
        "fill_policy": _require_text("fill_policy", fill_policy),
        "warmup_mode": _require_text("warmup_mode", warmup_mode),
        "fold_window": _normalize_fold_window(fold_window),
        "seed": None if seed is None else seed,
        "stack_id": _require_text("stack_id", stack_id),
    }
    missing = [name for name in REQUIRED_IDENTITY_FIELDS if name not in payload]
    if missing:
        raise TrialKeyError(f"missing identity fields: {missing}")
    return payload


def make_trial_key(**identity: Any) -> str:
    try:
        payload = identity_payload(**identity)
        return content_hash(payload, schema_id=SCHEMA_ID)
    except TypeError as exc:
        raise TrialKeyError("trial identity is incomplete") from exc
    except CanonicalizationError as exc:
        raise TrialKeyError(str(exc)) from exc


def make_trial_record(
    *,
    producer_version: str,
    producer_commit: str,
    created_at_utc_ms: int,
    lifecycle: str,
    trial_identity: Mapping[str, Any],
    metrics: Mapping[str, str] | None = None,
    retry_count: int = 0,
) -> dict[str, Any]:
    if (
        not isinstance(created_at_utc_ms, int)
        or isinstance(created_at_utc_ms, bool)
        or created_at_utc_ms < 0
    ):
        raise TrialKeyError("created_at_utc_ms must be a non-negative integer")
    if retry_count < 0:
        raise TrialKeyError("retry_count must be >= 0")
    trial_key = make_trial_key(**dict(trial_identity))
    stack_id = _require_text("stack_id", trial_identity["stack_id"])
    record = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "producer_version": _require_text("producer_version", producer_version),
        "producer_commit": _require_text("producer_commit", producer_commit),
        "stack_id": stack_id,
        "created_at_utc_ms": created_at_utc_ms,
        "serializer_id": SERIALIZER_ID,
        "content_hash_alg": "sha256",
        "content_hash": "",
        "trial_key": trial_key,
        "parameters": normalize_parameters(trial_identity["parameters"]),
        "fold_window": _normalize_fold_window(trial_identity.get("fold_window")),
        "lifecycle": _require_text("lifecycle", lifecycle),
        "retry_count": retry_count,
        "metrics": {} if metrics is None else dict(metrics),
    }
    record["content_hash"] = content_hash(record, schema_id=SCHEMA_ID)
    validate_payload(SCHEMA_ID, record)
    return record
