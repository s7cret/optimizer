"""Normalized optimizer runner responses shared by trial execution paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimizer.core.diagnostic import Diagnostic


@dataclass(frozen=True)
class NormalizedRunnerResponse:
    metrics_source: Any
    hashes: dict[str, str]
    diagnostics: list[Diagnostic]
    is_contract_response: bool = False
    trades_available: bool = False
    equity_available: bool = False

    @property
    def summary_metrics_available(self) -> bool:
        return bool(self.metrics_source)

    def hash(self, name: str, raw: Any) -> str | None:
        if name in self.hashes:
            return self.hashes[name]
        value = getattr(raw, name, None) if raw is not None else None
        return None if value is None else str(value)


def _response_field(raw, name, default=None):
    if isinstance(raw, dict):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _response_diagnostics(raw, trial_id, params_hash):
    out = []
    for item in _response_field(raw, "diagnostics", []) or []:
        if isinstance(item, Diagnostic):
            out.append(item)
            continue
        if isinstance(item, dict):
            out.append(
                Diagnostic(
                    str(item.get("code", "RUNNER_DIAGNOSTIC")),
                    str(item.get("message", "runner diagnostic")),
                    str(item.get("severity", "warning")),
                    trial_id,
                    params_hash,
                    context=item.get("context", {}),
                )
            )
    return out


__all__ = ["NormalizedRunnerResponse"]
