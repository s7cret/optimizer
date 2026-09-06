from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from copy import deepcopy
import inspect
import math
from types import CodeType
from typing import Any, cast

from optimizer.core.contracts import ACCEPTED_RUNNER_CONTRACTS
from optimizer.protocols import RunnerCapabilities, RunnerRequest, RunnerResponse


def _field(result: Any, name: str, default: Any = None) -> Any:
    return result.get(name, default) if isinstance(result, dict) else getattr(result, name, default)


def _metric_dict(result: Any, required: set[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in required:
        value = _field(result, name, None)
        if value is None and isinstance(result, dict):
            value = result.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            metric = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(metric):
            continue
        metrics[name] = metric
    return metrics


def _stable_hash(value: Any) -> str:
    value = _identity_value(value)
    try:
        from backtest_engine.core.deterministic_hash import sha256_obj
    except ImportError:
        from optimizer.core.expression import stable_hash

        return stable_hash(value)
    return sha256_obj(value)


def _identity_value(value: Any, _depth: int = 0) -> Any:
    """Development fingerprint only; durable runs require explicit source hashes.

    Never identify opaque state solely by its Python type. Captured plain state
    is included, while cyclic/resource-backed state requires explicit hashes.
    """
    if _depth > 32:
        raise TypeError("recursive runner state requires explicit fingerprints")

    def visit(item):
        return _identity_value(item, _depth + 1)

    if is_dataclass(value) and not inspect.isclass(value):
        from dataclasses import fields

        return {field.name: visit(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError("runner identity mappings require string keys")
        return {key: visit(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [visit(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, (set, frozenset)):
        import json

        items = [visit(item) for item in value]
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, CodeType):
        return {
            "bytecode": value.co_code.hex(),
            "constants": visit(value.co_consts),
            "names": visit(value.co_names),
            "variables": visit(value.co_varnames),
            "freevars": visit(value.co_freevars),
            "cellvars": visit(value.co_cellvars),
            "args": [value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount],
            "flags": value.co_flags,
        }
    if inspect.isclass(value):
        return {
            "module": value.__module__,
            "qualname": value.__qualname__,
            "source": _source_or_none(value),
        }
    if inspect.isfunction(value):
        return {
            "code": visit(value.__code__),
            "defaults": visit(value.__defaults__),
            "kwdefaults": visit(value.__kwdefaults__),
            "closure": [visit(cell.cell_contents) for cell in (value.__closure__ or ())],
        }
    state = getattr(value, "__dict__", None)
    if state:
        return {"type": visit(type(value)), "state": visit(state)}
    raise TypeError(f"opaque runner state {type(value).__name__} requires explicit fingerprints")


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
        "constants": None if code is None else _identity_value(code.co_consts),
        "keyword_defaults": _identity_value(getattr(value, "__kwdefaults__", None)),
        "bound_state": _identity_value(value.__self__) if inspect.ismethod(value) else None,
        "callable_state": None
        if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value)
        else _identity_value(value),
        "defaults": _identity_value(getattr(value, "__defaults__", None)),
        "closure": [_identity_value(cell.cell_contents) for cell in closure],
    }


def _fingerprint_result(result: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    content_hash = _field(result, "content_hash", None)
    if callable(content_hash):
        hashes["content_hash"] = str(content_hash())
    elif isinstance(content_hash, str) and content_hash:
        hashes["content_hash"] = content_hash
    elif _field(result, "content_hash_value", None):
        hashes["content_hash"] = str(_field(result, "content_hash_value"))
    for attr, key in (
        ("data_fingerprint", "data_fingerprint"),
        ("strategy_fingerprint", "runner_fingerprint"),
        ("runtime_fingerprint", "runtime_fingerprint"),
    ):
        value = _field(result, attr, None)
        if value:
            hashes[key] = str(value)
    config_snapshot = _field(result, "config_snapshot", None)
    if config_snapshot:
        hashes["engine_config_hash"] = _stable_hash(config_snapshot)
    return hashes


def _diagnostics(result: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for severity, items in (
        ("warning", _field(result, "warnings", []) or []),
        ("error", _field(result, "errors", []) or []),
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
    except (TypeError, ValueError) as exc:
        raise ValueError("cannot verify engine warmup support") from exc
    parameter = sig.parameters.get("effective_pre_bars")
    accepts_keyword = parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
    if not accepts_keyword and not any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        raise ValueError("engine does not support the requested effective_pre_bars")
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
        strict_identity: bool = False,
    ) -> None:
        self.engine_factory = engine_factory
        self.strategy = strategy
        self.bars = deepcopy(list(bars))
        self.static_params = deepcopy(dict(static_params or {}))
        if type(strict_identity) is not bool:
            raise TypeError("strict_identity must be a bool")
        if strict_identity:
            import re

            for name, value in (
                ("runner_fingerprint", runner_fingerprint),
                ("data_fingerprint", data_fingerprint),
                ("engine_config_hash", engine_config_hash),
            ):
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"(?:sha256:)?(?!0{64}$)[0-9a-f]{64}", value) is None
                ):
                    raise ValueError(f"strict runner identity requires an explicit SHA256 {name}")
        # Explicit durable identities never introspect opaque runner state.
        factory_identity = (
            _callable_identity(engine_factory)
            if not runner_fingerprint or not engine_config_hash
            else None
        )
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
        for name in ("range", "seed"):
            if getattr(request, name, None) is not None:
                raise ValueError(f"BacktestEngine adapter does not support {name}")
        if request.early_stop_conditions:
            raise ValueError("BacktestEngine adapter does not support early_stop_conditions")
        if set(request.required_outputs) - self.capabilities.supported_outputs:
            raise ValueError("unsupported required outputs")
        params = deepcopy({**self.static_params, **request.params})
        pre_bars = params.pop("_effective_pre_bars", None)
        if pre_bars is not None and (type(pre_bars) is not int or pre_bars < 0):
            raise ValueError("effective_pre_bars must be a nonnegative integer")
        engine = self.engine_factory()
        result = _run_engine(
            engine,
            self.strategy if inspect.isclass(self.strategy) else deepcopy(self.strategy),
            deepcopy(self.bars),
            params,
            pre_bars,
        )
        hashes = {**request.fingerprints, **_fingerprint_result(result)}
        diagnostics = _diagnostics(result)
        metrics = _metric_dict(result, set(request.required_metrics))
        for name in set(request.required_metrics) - set(metrics):
            value = _field(result, name, None)
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
        if _field(result, "status", "completed") != "completed":
            diagnostics.append(
                {
                    "code": "BACKTEST_ENGINE_RUN_NOT_COMPLETED",
                    "message": (
                        f"BacktestEngine returned status={_field(result, 'status', None)!r}"
                    ),
                    "severity": "error",
                }
            )
        return RunnerResponse(
            metrics=metrics,
            raw_result=result,
            hashes=hashes,
            trades_available=_field(result, "closed_trades", None) is not None,
            equity_available=_field(result, "equity_curve", None) is not None,
            diagnostics=diagnostics,
        )


__all__ = ["BacktestEngineRunnerAdapter"]
