from statistics import mean
from typing import Any


def _metric(metrics: dict[str, Any], base: str, prefix: str) -> float | None:
    for key in (f'{prefix}_{base}', f'{base}_{prefix}', f'{prefix}.{base}'):
        val = metrics.get(key)
        if val is not None:
            return float(val)
    return None


def analyze(trials: list[Any], metric: str = 'net_profit') -> dict[str, object]:
    scores: dict[int, float | None] = {}
    gaps: dict[int, float | None] = {}
    for t in trials:
        if getattr(t, 'status', None) != 'completed':
            continue
        train = _metric(t.metrics, metric, 'train')
        test = _metric(t.metrics, metric, 'test') or _metric(t.metrics, metric, 'validation')
        if train is None or test is None:
            scores[t.id] = None
            gaps[t.id] = None
            continue
        gap = (train - test) / (abs(train) + 1e-12)
        gaps[t.id] = gap
        scores[t.id] = max(0.0, min(1.0, 1.0 - max(0.0, gap)))
    valid = [v for v in scores.values() if v is not None]
    return {
        'status': 'ok' if valid else 'insufficient_data',
        'metric': metric,
        'scores_by_trial_id': scores,
        'generalization_gap_by_trial_id': gaps,
        'average_score': mean(valid) if valid else None,
        'diagnostics': [] if valid else [{'code': 'OVERFITTING_REQUIRES_TRAIN_TEST_METRICS', 'severity': 'warning', 'message': 'Expected train_/test_ metric pairs'}],
    }
