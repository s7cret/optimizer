"""Deterministic identity sealing for ``openpine.trial.identity.v1`` trials."""

from __future__ import annotations

import unicodedata
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any

from openpine_contracts import (
    CanonicalizationError,
    canonical_dumps,
    content_hash,
    decimal_string,
    list_schema_ids,
    schema_hash,
    seal_content_hash,
    validate_payload,
    verify_content_hash,
)

TRIAL_SCHEMA_ID = "openpine.trial.identity.v1"
_SHA256 = re.compile(r"^sha256:(?!0{64}$)[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _normalize_boundary(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    "map keys must be strings",
                    details={"key_type": type(key).__name__},
                )
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise CanonicalizationError(
                    "map keys collide after NFC normalization",
                    details={"normalized_key": normalized_key},
                )
            normalized[normalized_key] = _normalize_boundary(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_normalize_boundary(item) for item in value]
    if isinstance(value, float):
        decimal = Decimal(str(value))
        if not decimal.is_finite():
            raise CanonicalizationError(
                "decimal parameter must be finite", details={"value": str(value)}
            )
        return decimal_string(decimal)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CanonicalizationError(
                "decimal parameter must be finite", details={"value": str(value)}
            )
        return decimal_string(value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if (
        value is None
        or isinstance(value, bool)
        or (isinstance(value, int) and not isinstance(value, bool))
    ):
        return value
    raise CanonicalizationError(
        "unsupported type on trial identity boundary",
        details={"type": type(value).__name__},
    )


def normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize parameters to canonical JSON values without float boundaries."""

    normalized = _normalize_boundary(parameters)
    return dict(normalized)


@lru_cache(maxsize=1)
def _installed_schema_hashes() -> tuple[tuple[str, str], ...]:
    return tuple(
        (schema_id, schema_hash(schema_id))
        for schema_id in list_schema_ids(include_aliases=False)
    )


def contract_schema_hashes() -> dict[str, str]:
    """Return hashes from the installed contracts wheel's official catalog."""

    return dict(_installed_schema_hashes())


def _policy_value(value: Any, *keys: str) -> str:
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    if isinstance(value, str):
        return value
    return ""


def validate_trial_identity_payload(trial_key: str, payload: Mapping[str, Any]) -> None:
    """Validate schema, root seal, and key binding for a persisted identity."""

    if payload.get("content_hash") != trial_key:
        raise ValueError("trial identity payload does not match trial_key")
    validate_payload(TRIAL_SCHEMA_ID, payload)
    if not verify_content_hash(payload, schema_id=TRIAL_SCHEMA_ID):
        raise ValueError("trial identity payload content hash is invalid")


def validate_critical_identity_hashes(config: Any) -> bool:
    """Return whether durable identity is enabled; reject every partial identity."""

    fields = (
        "generated_artifact_hash",
        "data_snapshot_series_hash",
        "engine_build_hash",
        "engine_config_hash",
        "stack_manifest_hash",
    )
    values = {field: getattr(config, field, None) for field in fields}
    strict_trigger_fields = (
        "generated_artifact_hash",
        "data_snapshot_series_hash",
        "engine_build_hash",
        "stack_manifest_hash",
    )
    if all(values[field] is None for field in strict_trigger_fields):
        return False
    invalid = [
        field
        for field, value in values.items()
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None
    ]
    if invalid:
        from optimizer.errors import ParameterValidationError

        raise ParameterValidationError(
            "durable trial identity requires nonzero sha256 hashes: "
            + ", ".join(invalid)
        )
    commit = getattr(config, "optimizer_commit", None)
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        from optimizer.errors import ParameterValidationError

        raise ParameterValidationError(
            "durable trial identity requires exact optimizer_commit (40 lowercase hex)"
        )
    return True


@dataclass(frozen=True, slots=True)
class TrialKey:
    """A sealed TrialKey and the canonical payload from which it was derived."""

    trial_key: str
    payload: dict[str, Any]

    @property
    def canonical_json(self) -> str:
        return canonical_dumps(self.payload)


@dataclass(frozen=True, slots=True)
class TrialIdentity:
    """All execution inputs that can affect a deterministic optimizer trial."""

    generated_artifact_hash: str
    data_snapshot_series_hash: str
    parameters: Mapping[str, Any]
    engine_build_hash: str
    engine_config_hash: str
    semantic_profile: str
    finality_policy: Any
    warmup_policy: Any
    score_policy: Any
    end_policy: Any
    contract_schema_hashes: Mapping[str, str]
    stack_manifest_hash: str
    deterministic_seed: int | None
    fold_identity: Any
    walk_forward_identity: Any
    producer_commit: str
    optimizer_id: str
    strategy_id: str
    source_hash: str
    emitted_module_hash: str
    numeric_policy: str
    fill_policy: str
    objective_version: str
    constraints_version: str
    objective_metric: str | None = None
    objective_direction: str = "MAXIMIZE"
    objective_aggregation: str = "single-run"
    constraints: tuple[Mapping[str, Any], ...] = ()
    created_at_utc_ms: int = 0

    def canonical_payload(self) -> dict[str, Any]:
        finality = _policy_value(self.finality_policy, "bars", "mode")
        finality = {
            "FINAL": "CLOSED_BAR_ONLY",
            "OPEN": "ALLOW_OPEN",
        }.get(finality, finality)
        warmup = _policy_value(self.warmup_policy, "mode")
        score = _policy_value(self.score_policy, "window", "mode")
        score = "ALL_BARS" if score in {"all", "ALL_BARS"} else "AFTER_WARMUP"
        end = _policy_value(self.end_policy, "mode")
        end = (
            "LIQUIDATE_ON_LAST_BAR"
            if end in {"liquidate", "LIQUIDATE_ON_LAST_BAR"}
            else "PRESERVE_OPEN_POSITIONS"
        )
        runner_fingerprint = content_hash(
            {
                "engine_build_hash": self.engine_build_hash,
                "engine_config_hash": self.engine_config_hash,
            },
            schema_major=1,
        )
        payload = {
            "schema_id": TRIAL_SCHEMA_ID,
            "schema_version": "1.0.0",
            "producer": "optimizer",
            "producer_version": "5.0.0-rc.4",
            "producer_commit": self.producer_commit,
            "stack_id": self.stack_manifest_hash,
            "created_at_utc_ms": self.created_at_utc_ms,
            "serializer_id": "openpine.canonical.json.v1",
            "content_hash_alg": "sha256",
            "optimizer_id": self.optimizer_id,
            "strategy_id": self.strategy_id,
            "generated_artifact_hash": self.generated_artifact_hash,
            "source_hash": self.source_hash or self.generated_artifact_hash,
            "emitted_module_hash": self.emitted_module_hash
            or self.generated_artifact_hash,
            "data_snapshot_hash": self.data_snapshot_series_hash,
            "stack_manifest_hash": self.stack_manifest_hash,
            "runner_fingerprint": runner_fingerprint,
            "parameters": normalize_parameters(self.parameters),
            "policies": {
                "semantic_profile": self.semantic_profile,
                "finality_policy": finality,
                "warmup_policy": warmup,
                "score_policy": score,
                "end_policy": end,
                "numeric_policy": self.numeric_policy,
                "fill_policy": self.fill_policy,
            },
            "schema_set": [
                {"schema_id": schema_id, "schema_hash": schema_hash}
                for schema_id, schema_hash in sorted(
                    self.contract_schema_hashes.items()
                )
            ],
            "seed": self.deterministic_seed,
            "fold": self.fold_identity,
            "walk_forward": self.walk_forward_identity,
            "objective": {
                "metric": self.objective_metric or self.objective_version,
                "direction": self.objective_direction,
                "aggregation": f"{self.objective_aggregation}:{self.constraints_version}",
            },
            "constraints": list(self.constraints),
        }
        return dict(_normalize_boundary(payload))

    def seal(self) -> TrialKey:
        sealed = seal_content_hash(self.canonical_payload(), schema_id=TRIAL_SCHEMA_ID)
        trial_key = str(sealed["content_hash"])
        validate_trial_identity_payload(trial_key, sealed)
        return TrialKey(trial_key, sealed)


__all__ = [
    "TRIAL_SCHEMA_ID",
    "TrialIdentity",
    "TrialKey",
    "contract_schema_hashes",
    "normalize_parameters",
    "validate_critical_identity_hashes",
    "validate_trial_identity_payload",
]
