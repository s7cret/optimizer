# Optimizer

Independent Python package for strategy parameter optimization. The package does **not** import Pine/PineLib/AST2Python/MarketDataProvider/backtest-engine concrete types; the semantic boundary is a user supplied `BacktestRunner` callable or advanced `RunnerRequest` runner protocol.

## Quick start

```python
from optimizer import Parameter, OptimizerConfig, optimize

params = [Parameter('fast', 'int', 5, 2, 10, 1), Parameter('use_filter', 'bool', False)]

def runner(p):
    return {'net_profit': p['fast'] * 10, 'max_drawdown_percent': 20 - p['fast'], 'profit_factor': 1.2, 'sharpe_ratio': 0.8}

result = optimize(params, runner, OptimizerConfig(algorithm='grid', max_trials=20))
print(result.recommended_trial.params)
```

## Implemented MVP

- Parameter/ParameterSpace validation, grid/random/adaptive-grid generation, safe AST cross constraints.
- MetricExtractor and MetricRegistry with required metric/output awareness.
- Trial/OptimizerResult/Profile models, objective scoring, constraints, balanced ranking, leaderboard, Pareto front/knee.
- Runner boundary with basic and advanced `RunnerRequest` protocols and diagnostics when basic runners cannot receive hints.
- SQLite/JSON persistence, run fingerprints and force-resume guard.
- JSON/CSV/Markdown reports and CLI skeleton.

## Full-scope diagnostics/placeholders

Bayesian, genetic, walk-forward, advanced robustness/overfitting/profit-concentration/plot/diff commands are explicit placeholders that raise or report diagnostics rather than silently pretending completion.
