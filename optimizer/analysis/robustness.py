import json
from statistics import mean
from typing import Any


def _key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def compute_neighborhood_robustness(
    trials: list[Any], space: Any | None = None, radius_steps: int = 1, min_neighbors: int = 1
) -> dict[int, float | None]:
    completed = [
        t
        for t in trials
        if getattr(t, "status", None) == "completed" and t.objective_value is not None
    ]
    by_key = {_key(t.params): t for t in completed}
    scores: dict[int, float | None] = {}
    for trial in completed:
        neighbors = []
        if space is not None:
            for params in space.neighbors(trial.params, radius_steps):
                other = by_key.get(_key(params))
                if other is not None and other.objective_value is not None:
                    neighbors.append(float(other.objective_value))
        if not neighbors:
            scores[trial.id] = trial.metrics.get("robustness_score")
            continue
        if len(neighbors) < min_neighbors:
            scores[trial.id] = None
            continue
        center = float(trial.objective_value)
        avg_drop = max(0.0, center - mean(neighbors))
        denom = abs(center) + 1e-12
        scores[trial.id] = max(0.0, min(1.0, 1.0 - avg_drop / denom))
    return scores


def analyze(
    trials: list[Any], space: Any | None = None, radius_steps: int = 1, min_neighbors: int = 1
) -> dict[str, object]:
    scores = compute_neighborhood_robustness(trials, space, radius_steps, min_neighbors)
    valid = {k: v for k, v in scores.items() if v is not None}
    return {
        "status": "ok" if valid else "insufficient_data",
        "scores_by_trial_id": scores,
        "average_score": mean(valid.values()) if valid else None,
        "diagnostics": []
        if valid
        else [
            {
                "code": "ROBUSTNESS_INSUFFICIENT_NEIGHBORS",
                "severity": "warning",
                "message": "No trials had enough evaluated neighbors",
            }
        ],
    }
