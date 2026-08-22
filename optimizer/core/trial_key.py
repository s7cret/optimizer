"""Deterministic identity sealing for ``openpine.trial.v2`` trials."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any

from openpine_contracts import (
    CanonicalizationError,
    canonical_dumps,
    decimal_string,
    list_schema_ids,
    schema_hash,
    seal_content_hash,
)

TRIAL_SCHEMA_ID = "openpine.trial.v2"


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
    objective_version: str
    constraints_version: str

    def canonical_payload(self) -> dict[str, Any]:
        payload = {
            "schema_id": TRIAL_SCHEMA_ID,
            "generated_artifact_hash": self.generated_artifact_hash,
            "data_snapshot_series_hash": self.data_snapshot_series_hash,
            "normalized_parameters": normalize_parameters(self.parameters),
            "engine_build_hash": self.engine_build_hash,
            "engine_config_hash": self.engine_config_hash,
            "semantic_profile": self.semantic_profile,
            "finality_policy": self.finality_policy,
            "warmup_policy": self.warmup_policy,
            "score_policy": self.score_policy,
            "end_policy": self.end_policy,
            "contract_schema_hashes": self.contract_schema_hashes,
            "stack_manifest_hash": self.stack_manifest_hash,
            "deterministic_seed": self.deterministic_seed,
            "fold_identity": self.fold_identity,
            "walk_forward_identity": self.walk_forward_identity,
            "objective_version": self.objective_version,
            "constraints_version": self.constraints_version,
        }
        return dict(_normalize_boundary(payload))

    def seal(self) -> TrialKey:
        sealed = seal_content_hash(self.canonical_payload(), schema_id=TRIAL_SCHEMA_ID)
        return TrialKey(str(sealed["content_hash"]), sealed)


__all__ = [
    "TRIAL_SCHEMA_ID",
    "TrialIdentity",
    "TrialKey",
    "contract_schema_hashes",
    "normalize_parameters",
]
