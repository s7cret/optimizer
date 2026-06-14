# Development

Use Python 3.11 or newer.

```bash
python -m pip install -e .[dev]
python -m compileall -q optimizer tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_cov --cov=optimizer --cov-report=term
python -m ruff check .
BLACK_NUM_WORKERS=1 python -m black --check .
python -m mypy optimizer
python -m optimizer.quality duplicates optimizer
python -m optimizer.quality architecture optimizer --max-lines 700
python -m optimizer.distribution manifest --root .
python -m optimizer.release --root .
bash scripts/smoke_import_parse.sh
bash scripts/wheel_smoke.sh
```

The package release gate is hermetic and enforces 100% package coverage. It also checks
that the public CLI contains only implemented standalone commands, distribution metadata
rejects release artifacts, and parameter validation rejects ambiguous bool/numeric edge
cases. BacktestEngine/PineLib/OpenPine integration should be validated in the full-stack
repository layout after the standalone library gate passes.
