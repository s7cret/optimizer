import math

from optimizer.core.expression import safe_eval_numeric
from optimizer.core.metric_registry import MetricRegistry


def objective_direction(name, configured="auto"):
    if configured != "auto":
        return configured
    return "minimize" if MetricRegistry().direction(name) == "minimize" else "maximize"


def compute_objective(metrics, objective="net_profit", direction="auto", expression=None):
    if expression:
        return safe_eval_numeric(expression, metrics)
    v = metrics.get(objective)
    if v is None:
        raise KeyError(f"missing objective metric: {objective}")
    number = float(v)
    if isinstance(v, bool) or not math.isfinite(number):
        raise ValueError(f"objective metric {objective!r} must be finite and numeric")
    return number


def objective_sort_value(value, direction):
    if value is None:
        return float("-inf")
    return value if direction == "maximize" else -value


def objective_better(a, b, direction):
    if a is None:
        return False
    if b is None:
        return True
    return a > b if direction == "maximize" else a < b
