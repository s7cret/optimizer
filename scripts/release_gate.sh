#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export BLACK_NUM_WORKERS="${BLACK_NUM_WORKERS:-1}"

run() {
  echo "+ $*"
  "$@"
}

run "$PYTHON" -m compileall -q optimizer tests
run "$PYTHON" -m ruff check .
run "$PYTHON" -m black --check .
run "$PYTHON" -m mypy optimizer
run "$PYTHON" -m pytest -q -p pytest_cov --cov=optimizer --cov-report=term
run "$PYTHON" -m optimizer.quality duplicates optimizer
run "$PYTHON" -m optimizer.quality architecture optimizer --max-lines 700
run "$PYTHON" -m optimizer.distribution manifest --root .
run "$PYTHON" -m optimizer.release --root .
run bash scripts/smoke_import_parse.sh
run bash scripts/wheel_smoke.sh
