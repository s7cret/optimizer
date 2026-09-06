from dataclasses import asdict, is_dataclass
from typing import Callable
import math


class MetricExtractor:
    def __init__(
        self,
        custom_extractors: dict[str, Callable[[object], float]] | None = None,
        custom_conflict_policy: str = "override",
    ) -> None:
        self._custom_extractors = custom_extractors or {}
        self.custom_conflict_policy = custom_conflict_policy

    def _to_dict(self, obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {}

    def extract(self, backtest_result: object) -> dict[str, float]:
        raw = self._to_dict(backtest_result)
        metrics = {}

        def flatten(prefix, d):
            for k, v in d.items():
                name = f"{prefix}_{k}" if prefix else str(k)
                if isinstance(v, dict):
                    flatten(name, v)
                else:
                    try:
                        if v is not None and not isinstance(v, bool):
                            number = float(v)
                            if math.isfinite(number):
                                metrics[name] = number
                    except (TypeError, ValueError, OverflowError):
                        pass

        flatten("", raw)
        if "max_drawdown" in metrics and metrics["max_drawdown"] < 0:
            metrics["max_drawdown"] = abs(metrics["max_drawdown"])
        for name, fn in self._custom_extractors.items():
            value = fn(backtest_result)
            val = float(value)
            if isinstance(value, bool) or not math.isfinite(val):
                raise ValueError(f"custom metric {name!r} must be a finite number")
            if name in metrics and self.custom_conflict_policy == "error":
                raise ValueError(f"custom extractor conflict: {name}")
            if name in metrics and self.custom_conflict_policy == "keep_builtin":
                continue
            metrics[name] = val
        return metrics
