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

- Parameter/ParameterSpace validation, grid/random/adaptive-grid generation, lazy grid iteration, safe AST cross constraints, and a separate dry-run validation path for invalid generated combinations.
- MetricExtractor and MetricRegistry with required metric/output/statistics awareness, objective-expression metric extraction, and profile metric requirements.
- Trial/OptimizerRunResult/Profile models, objective scoring with minimize/maximize direction support, hard/soft min/max/eq/neq constraints, auto profile constraints, balanced ranking, leaderboard, Pareto front/knee.
- Baseline trial execution/comparison with `RECOMMENDED_WORSE_THAN_BASELINE` diagnostics.
- Runner boundary with basic and advanced `RunnerRequest` protocols and diagnostics when basic runners cannot receive hints.
- SQLite/JSON persistence with params-hash resume, loaded prior trials in results, atomic JSON writes, SQLite WAL, fingerprints and force-resume guard.
- JSON/CSV/Markdown reports, diff report, and plot export (HTML always; PNG/SVG when optional matplotlib is installed).
- Public `optimize()` routing for `grid`, `random`, `adaptive_grid`, native sequential `genetic`, and dependency-free surrogate `bayesian` optimizers.
- Range-aware walk-forward execution through `optimize(..., OptimizerConfig(algorithm='walk_forward'), start=..., end=...)` (or `optimizer.algorithms.walk_forward.run()`) for runners that support `RunnerRequest.range` or `with_range(start, end)`.
- Local BacktestEngine adapter contract documented in `docs/BACKTEST_ENGINE_RUNNER.md`; it propagates hashes/diagnostics and fails trials on engine error diagnostics or missing required outputs.
- Advanced analyses: neighborhood robustness, sensitivity/parameter importance, train/test overfitting gap, profit concentration, trade bootstrap Monte Carlo, heatmap and walk-forward summaries.

## Graceful degradation

Optional/data-dependent features return structured `status`/`diagnostics` instead of pretending success: image plots need matplotlib, trade analyses need saved trade lists, and walk-forward needs a range-aware runner. The package remains independent of concrete Pine/backtest engine implementations.

## Installation, Docker, and Publication

```bash
./scripts/install.sh --dev
docker compose run --rm optimizer
```

For a public GitHub release checklist, see `docs/GITHUB_PUBLICATION.md`.
