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

## Implemented

- Parameter/ParameterSpace validation, grid/random/adaptive-grid generation, safe AST cross constraints.
- MetricExtractor and MetricRegistry with required metric/output awareness.
- Trial/OptimizerResult/Profile models, objective scoring, constraints, balanced ranking, leaderboard, Pareto front/knee.
- Runner boundary with basic and advanced `RunnerRequest` protocols and diagnostics when basic runners cannot receive hints.
- SQLite/JSON persistence, run fingerprints and force-resume guard.
- JSON/CSV/Markdown reports, diff report, and plot export (HTML always; PNG/SVG when optional matplotlib is installed).
- Public `optimize()` routing for `grid`, `random`, `adaptive_grid`, native sequential `genetic`, and dependency-free surrogate `bayesian` optimizers.
- Range-aware walk-forward execution through `optimize(..., OptimizerConfig(algorithm='walk_forward'), start=..., end=...)` (or `optimizer.algorithms.walk_forward.run()`) for runners that support `RunnerRequest.range` or `with_range(start, end)`.
- Advanced analyses: neighborhood robustness, sensitivity/parameter importance, train/test overfitting gap, profit concentration, trade bootstrap Monte Carlo, heatmap and walk-forward summaries.

## Graceful degradation

Optional/data-dependent features return structured `status`/`diagnostics` instead of pretending success: image plots need matplotlib, trade analyses need saved trade lists, and walk-forward needs a range-aware runner. The package remains independent of concrete Pine/backtest engine implementations.
