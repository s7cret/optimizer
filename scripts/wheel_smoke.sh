#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
SRC="$TMPDIR/src"
mkdir -p "$SRC" "$TMPDIR/dist" "$TMPDIR/site"
"$PYTHON" - <<'PY' "$PWD" "$SRC"
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
ignored_names = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    "dist",
    "build",
    "optimizer.egg-info",
    "optimizer_results",
}


def ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in ignored_names or name.endswith(".zip")}

shutil.copytree(source, dest, dirs_exist_ok=True, ignore=ignore)
PY
(cd "$SRC" && "$PYTHON" -m pip wheel --no-index --no-deps --no-build-isolation -w "$TMPDIR/dist" .)
"$PYTHON" -m pip install --no-index --no-deps --target "$TMPDIR/site" "$TMPDIR"/dist/*.whl
OPTIMIZER_WHEEL_SMOKE_OUT="$TMPDIR/optimizer_results" PYTHONPATH="$TMPDIR/site" "$PYTHON" - <<'PY'
import os
from pathlib import Path
from optimizer import OptimizerConfig, Parameter, optimize


def runner(params):
    return {"net_profit": 1.0, "max_drawdown": 1.0}

result = optimize(
    [Parameter("x", "int", 1, 1, 1, 1)],
    runner,
    OptimizerConfig(
        max_trials=1,
        report_profiles=False,
        use_profile_auto_constraints=False,
        output_dir=Path(os.environ["OPTIMIZER_WHEEL_SMOKE_OUT"]),
    ),
)
assert result.status == "completed"
print("optimizer wheel smoke ok")
PY
