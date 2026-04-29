from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
import math
from typing import Any

from optimizer.protocols import RunnerCapabilities, RunnerRequest, RunnerResponse


def _metric_dict(result: Any, required: set[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in required:
        value = getattr(result, name, None)
        if value is None and isinstance(result, dict):
            value = result.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            metric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(metric):
            continue
        metrics[name] = metric
    return metrics


def _stable_hash(value: Any) -> str:
    try:
        from backtest_engine.core.deterministic_hash import sha256_obj
    except Exception:  # pragma: no cover - optional dependency import guard
        from optimizer.core.expression import stable_hash

        return stable_hash(value)
    return sha256_obj(value)


def _fingerprint_result(result: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    content_hash = getattr(result, "content_hash", None)
    if callable(content_hash):
        hashes["content_hash"] = str(content_hash())
    elif getattr(result, "content_hash_value", None):
        hashes["content_hash"] = str(result.content_hash_value)
    for attr, key in (
        ("data_fingerprint", "data_fingerprint"),
        ("strategy_fingerprint", "runner_fingerprint"),
        ("runtime_fingerprint", "runtime_fingerprint"),
    ):
        value = getattr(result, attr, None)
        if value:
            hashes[key] = str(value)
    config_snapshot = getattr(result, "config_snapshot", None)
    if config_snapshot:
        hashes["engine_config_hash"] = _stable_hash(config_snapshot)
    return hashes


def _diagnostics(result: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for severity, items in (
        ("warning", getattr(result, "warnings", []) or []),
        ("error", getattr(result, "errors", []) or []),
    ):
        for item in items:
            if is_dataclass(item):
                payload = asdict(item)
            elif isinstance(item, dict):
                payload = dict(item)
            else:
                payload = {"message": str(item)}
            payload.setdefault("code", "BACKTEST_ENGINE_DIAGNOSTIC")
            payload.setdefault("severity", severity)
            payload.setdefault("message", payload.get("message", str(item)))
            out.append(payload)
    return out


class BacktestEngineRunnerAdapter:
    """Optimizer RunnerRequest adapter around a local BacktestEngine strategy.

    The adapter is intentionally thin and deterministic: it forwards optimizer
    parameters to ``BacktestEngine.run(...)``, returns only requested scalar
    metrics, and propagates result/config/data hashes for trial lineage.
    """

    capabilities = RunnerCapabilities(
        supports_runner_request=True,
        supports_required_outputs=True,
        supported_outputs={"summary_metrics", "closed_trades", "equity_curve"},
        supports_content_hash=True,
        supports_data_fingerprint=True,
        supports_engine_config_hash=True,
    )

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any],
        strategy: Any,
        bars: Sequence[Any],
        static_params: dict[str, Any] | None = None,
    ) -> None:
        self.engine_factory = engine_factory
        self.strategy = strategy
        self.bars = list(bars)
        self.static_params = dict(static_params or {})

    def __call__(self, request: RunnerRequest) -> RunnerResponse:
        if request.contract != "pain.optimizer_runner.v1":
            return RunnerResponse(
                metrics={},
                hashes=dict(request.fingerprints),
                diagnostics=[
                    {
                        "code": "RUNNER_REQUEST_CONTRACT_MISMATCH",
                        "message": f"unsupported runner request contract: {request.contract}",
                        "severity": "error",
                    }
                ],
            )
        params = {**self.static_params, **request.params}
        engine = self.engine_factory()
        result = engine.run(self.strategy, bars=self.bars, params=params)
        hashes = {**request.fingerprints, **_fingerprint_result(result)}
        diagnostics = _diagnostics(result)
        metrics = _metric_dict(result, set(request.required_metrics))
        for name in set(request.required_metrics) - set(metrics):
            value = getattr(result, name, None)
            if value is None and isinstance(result, dict):
                value = result.get(name)
            if value is not None:
                diagnostics.append(
                    {
                        "code": "BACKTEST_ENGINE_BAD_METRIC_VALUE",
                        "message": f"metric {name!r} is not a finite numeric value",
                        "severity": "error",
                        "context": {"metric": name, "value_type": type(value).__name__},
                    }
                )
        if getattr(result, "status", "completed") != "completed":
            diagnostics.append(
                {
                    "code": "BACKTEST_ENGINE_RUN_NOT_COMPLETED",
                    "message": (
                        f"BacktestEngine returned status={getattr(result, 'status', None)!r}"
                    ),
                    "severity": "error",
                }
            )
        return RunnerResponse(
            metrics=metrics,
            raw_result=result,
            hashes=hashes,
            trades_available=getattr(result, "closed_trades", None) is not None,
            equity_available=getattr(result, "equity_curve", None) is not None,
            diagnostics=diagnostics,
        )


__all__ = ["BacktestEngineRunnerAdapter"]
