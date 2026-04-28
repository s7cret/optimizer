from optimizer.core.expression import safe_eval
from optimizer.core.metric_registry import MetricRegistry

def objective_direction(name, configured='auto'):
    if configured!='auto': return configured
    return 'minimize' if MetricRegistry().direction(name)=='minimize' else 'maximize'

def compute_objective(metrics, objective='net_profit', direction='auto', expression=None):
    if expression: return float(safe_eval(expression, metrics))
    v=metrics.get(objective)
    if v is None: raise KeyError(f'missing objective metric: {objective}')
    return float(v)

def objective_sort_value(value, direction):
    if value is None: return float('-inf')
    return value if direction=='maximize' else -value
