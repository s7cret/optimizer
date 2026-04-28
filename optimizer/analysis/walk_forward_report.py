from statistics import mean
from typing import Any


def analyze(walk_forward_result: dict[str, Any]) -> dict[str, object]:
    windows = walk_forward_result.get('windows', []) if isinstance(walk_forward_result, dict) else []
    rows = []
    test_values = []
    for item in windows:
        trial = item.get('test_trial')
        value = getattr(trial, 'objective_value', None)
        if value is not None:
            test_values.append(float(value))
        rows.append({'window': item.get('window'), 'ranges': item.get('ranges'), 'test_objective': value, 'test_status': getattr(trial, 'status', None)})
    return {'status': 'ok' if rows else 'insufficient_data', 'windows': rows, 'average_test_objective': mean(test_values) if test_values else None}
