from statistics import mean
from typing import Any


def analyze(trials: list[Any]) -> dict[str, object]:
    completed = [t for t in trials if getattr(t, 'status', None) == 'completed' and t.objective_value is not None]
    if len(completed) < 2:
        return {'status': 'insufficient_data', 'importance': {}, 'diagnostics': [{'code': 'SENSITIVITY_REQUIRES_TRIALS', 'severity': 'warning', 'message': 'Need at least two completed trials'}]}
    names = sorted({k for t in completed for k in t.params})
    importance: dict[str, float | None] = {}
    for name in names:
        groups: dict[object, list[float]] = {}
        for t in completed:
            groups.setdefault(t.params.get(name), []).append(float(t.objective_value))
        if len(groups) < 2:
            importance[name] = 0.0
            continue
        means = [mean(v) for v in groups.values()]
        importance[name] = max(means) - min(means)
    total = sum(abs(v or 0.0) for v in importance.values())
    if total > 0:
        importance = {k: abs(v or 0.0) / total for k, v in importance.items()}
    return {'status': 'ok', 'importance': importance, 'diagnostics': []}
