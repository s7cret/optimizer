from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
import inspect
import math
from typing import Any, cast

from optimizer.core.contracts import ACCEPTED_RUNNER_CONTRACTS
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
    except Exception:
        from optimizer.core.expression import stable_hash

        return stable_hash(value)
    return sha256_obj(value)


def _identity_value(value: Any) -> Any:
    if is_dataclass(value) and not inspect.isclass(value):
        return asdict(cast(Any, value))
    if isinstance(value, dict):
        return {str(key): _identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if inspect.isclass(value):
        return {
            "module": value.__module__,
            "qualname": value.__qualname__,
            "source": _source_or_none(value),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _source_or_none(value: Any) -> str | None:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return None


def _callable_identity(value: Any) -> dict[str, Any]:
    code = getattr(value, "__code__", None)
    closure = getattr(value, "__closure__", None) or ()
    return {
        "module": getattr(value, "__module__", type(value).__module__),
        "qualname": getattr(value, "__qualname__", type(value).__qualname__),
        "source": _source_or_none(value),
        "bytecode": None if code is None else code.co_code.hex(),
        "constants": None if code is None else repr(code.co_consts),
        "defaults": _identity_value(getattr(value, "__defaults__", None)),
        "closure": [_identity_value(cell.cell_contents) for cell in closure],
    }


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
                payload = asdict(cast(Any, item))
            elif isinstance(item, dict):
                payload = dict(item)
            else:
                payload = {"message": str(item)}
            payload.setdefault("code", "BACKTEST_ENGINE_DIAGNOSTIC")
            payload.setdefault("severity", severity)
            payload.setdefault("message", payload.get("message", str(item)))
            out.append(payload)
    return out


def _run_engine(
    engine: Any,
    strategy: Any,
    bars: Sequence[Any],
    params: dict[str, Any],
    effective_pre_bars: Any | None,
) -> Any:
    if effective_pre_bars is None:
        return engine.run(strategy, bars=list(bars), params=params)
    try:
        import inspect

        sig = inspect.signature(engine.run)
    except (TypeError, ValueError):
        return engine.run(strategy, bars=list(bars), params=params)
    if "effective_pre_bars" not in sig.parameters:
        return engine.run(strategy, bars=list(bars), params=params)
    return engine.run(
        strategy,
        bars=list(bars),
        params=params,
        effective_pre_bars=effective_pre_bars,
    )


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
        runner_fingerprint: str | None = None,
        data_fingerprint: str | None = None,
        engine_config_hash: str | None = None,
    ) -> None:
        self.engine_factory = engine_factory
        self.strategy = strategy
        self.bars = list(bars)
        self.static_params = dict(static_params or {})
        factory_identity = _callable_identity(engine_factory)
        self.data_fingerprint = data_fingerprint or _stable_hash(self.bars)
        self.engine_config_hash = engine_config_hash or _stable_hash(
            {"engine_factory": factory_identity, "static_params": self.static_params}
        )
        self.runner_fingerprint = runner_fingerprint or _stable_hash(
            {
                "adapter": f"{type(self).__module__}.{type(self).__qualname__}",
                "engine_factory": factory_identity,
                "strategy": _identity_value(strategy),
            }
        )

    def fingerprint(self) -> str:
        return self.runner_fingerprint

    def __call__(self, request: RunnerRequest) -> RunnerResponse:
        if request.contract not in ACCEPTED_RUNNER_CONTRACTS:
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
        result = _run_engine(
            engine,
            self.strategy,
            self.bars,
            params,
            self.static_params.get("_effective_pre_bars"),
        )
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
