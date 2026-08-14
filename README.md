# Optimizer 4.0.2

> Dependency-light parameter optimizer and runner contract layer for OpenPine strategy backtests.

[![Version](https://img.shields.io/badge/version-4.0.2-blue)](https://github.com/s7cret/optimizer) [![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://github.com/s7cret/optimizer) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/s7cret/optimizer)


**GitHub description:** Optimizer provides parameter-space validation, search algorithms, scoring, ranking, resume metadata, and reports for OpenPine/backtest runners through a clean runner protocol.

**Suggested topics:** `optimization`, `backtesting`, `algorithmic-trading`, `walk-forward`, `parameter-search`, `trading-strategies`, `python`, `openpine`.

## What Optimizer is

Optimizer is the parameter-search and result-ranking layer of the OpenPine stack. It builds candidate parameter sets, validates constraints, calls a supplied runner, scores completed trials, persists enough metadata for deterministic resume/audit, and exports reports.

```text
parameter space -> optimizer -> runner protocol -> backtest engine / OpenPine runner
                              -> ranked trials, profiles, reports
```

The optimizer core is deliberately separated from market data, Pine runtime, broker emulation, and strategy execution. Those systems connect through the runner protocol.

## Main capabilities

- Parameter definitions for `int`, `float`, `bool`, `string`, and `enum` values.
- Parameter-space validation for bounds, steps, types, defaults, and enabled flags.
- Search algorithms including grid, random, adaptive grid, Bayesian-style, genetic, and walk-forward surfaces where available in the package.
- Runner protocol with request/response metadata and lineage fingerprints.
- Objective scoring, constraints, ranking, and recommendation policies.
- Resume-safe trial persistence and result directories.
- Analysis helpers for robustness, sensitivity, overfitting, profit concentration, Monte Carlo, heatmaps, and walk-forward reports.
- JSON, CSV, Markdown, console, plot, and Telegram-summary reporting helpers.

## Boundaries

Optimizer does not fetch candles, parse Pine, lower AST, execute generated code, emulate fills, or place trades. It only coordinates parameter candidates and runner calls. The quality of optimizer output depends on the supplied runner, market data, scoring objective, constraints, and validation windows.

Process timeouts are lifecycle containment for trusted runner integrations, not a security sandbox. On Linux, Optimizer uses cgroup v2 when delegated and otherwise uses pidfd-pinned process-tree cleanup; other platforms use direct-process cleanup without unsafe numeric process-group signaling. Run untrusted runner code under a separate operating-system identity or external sandbox.

## Runner contract

Preferred contract id:

```text
pine.optimizer_runner.v1
```

The legacy typo `pain.optimizer_runner.v1` is accepted only for migration compatibility. New integrations should emit the `pine.*` contract.

Runner responses may include content, data, runner, runtime, and engine-config lineage hashes. Trial records preserve those fingerprints for audit, resume, and OpenPine reporting.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Install from GitHub tag:

```bash
python -m pip install 'git+https://github.com/s7cret/optimizer.git@v4.0.2'
```

## Python quick start

```python
from pathlib import Path

from optimizer import OptimizerConfig, Parameter, optimize

def runner(params: dict[str, float]) -> dict[str, float]:
    x = params["x"]
    y = params["y"]
    return {
        "net_profit": 100 - (x - 3) ** 2 * 10 - (y - 2) ** 2 * 5,
        "max_drawdown_percent": 5 + abs(x - y),
        "profit_factor": 1 + x / 10,
        "sharpe_ratio": y / 2,
    }

params = [
    Parameter("x", "int", 1, 1, 5, 1),
    Parameter("y", "int", 1, 1, 4, 1),
]

result = optimize(
    params,
    runner,
    OptimizerConfig(max_trials=20, output_dir=Path("optimizer_results")),
)

print(result.recommended_profile)
print(result.recommended_trial.params)
```

## CLI quick start

Create a parameter file and provide a `FILE:OBJECT` runner:

```bash
optimizer dry-run --params params.json --output-dir ./optimizer_results
optimizer run --params params.json --runner ./runner.py:runner --algorithm grid --objective net_profit --max-trials 100 --output-dir ./optimizer_results
optimizer analyze --result-dir ./optimizer_results
optimizer export --result-dir ./optimizer_results --format markdown
optimizer resume --params params.json --runner ./runner.py:runner --algorithm grid --objective net_profit --max-trials 100 --output-dir ./optimizer_results
```

The CLI is intentionally small and fail-closed. Higher-level dashboards, champion review, cross-repository comparisons, and trading workflows belong to OpenPine application surfaces.

## Repository layout

```text
optimizer/
  algorithms/             candidate generation algorithms
  core/                   parameter space, scoring, constraints, metrics, trial runner
  results/                trial/run/profile models and ranking data
  selection/              recommendation and ranking policies
  storage/                JSON/SQLite persistence surfaces
  reporting/              console, JSON, CSV, Markdown, plot, Telegram summaries
  analysis/               robustness, overfitting, sensitivity, Monte Carlo, heatmap helpers
  runners/                optional runner adapters
  cli/                    run/resume/dry-run/analyze/export commands
examples/                 standalone dummy runner example
```

## Release checks

```bash
python -m compileall -q optimizer tests examples
python -m ruff check .
BLACK_NUM_WORKERS=1 python -m black --check .
python -m mypy optimizer
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_cov tests --cov=optimizer --cov-report=term
python -m optimizer.quality duplicates optimizer
python -m optimizer.quality architecture optimizer --max-lines 700
python -m optimizer.distribution manifest --root .
python -m optimizer.release --root .
bash scripts/smoke_import_parse.sh
```

## Trading disclaimer

Optimization can easily overfit historical data. Always use out-of-sample validation, walk-forward checks, conservative constraints, realistic fees/slippage, and paper testing before considering live execution.

## Documentation

- `docs/ARCHITECTURE.md` — optimizer responsibilities, runner contract, CLI boundary, release hygiene, parameter validation.
- `docs/DEVELOPMENT.md` — local checks and maintenance workflow.
- `docs/RELEASE_4_0.md` — 4.0.0 release readiness checklist.

## License

MIT. See `LICENSE`.

## Support

OpenPine development is independent and MIT-licensed. Support is optional and does not change license terms, feature access, or project guarantees.

- Telegram: https://t.me/OpenPine
- TON: `UQAyIr2sQ4-_Q5L-4VINcU18khDas5GPbAlYEkQN6S_qzui2`
- SOL: `EbxMUK2W4RGeQZCTRFrdgpEJvnqtyczPZvBrQa1cYJnQ`