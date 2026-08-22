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


__all__ = ["NormalizedRunnerResponse"]
