#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
SRC="$TMPDIR/src"
SITE_PACKAGES="$TMPDIR/site-packages"
mkdir -p "$SRC" "$TMPDIR/dist" "$SITE_PACKAGES"
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
"$PYTHON" -m build --wheel --outdir "$TMPDIR/dist" "$SRC"
"$PYTHON" -m pip install --no-index --no-deps --target "$SITE_PACKAGES" "$TMPDIR"/dist/*.whl
(
  cd "$TMPDIR"
  OPTIMIZER_WHEEL_SMOKE_OUT="$TMPDIR/optimizer_results" \
    "$PYTHON" -I - "$SITE_PACKAGES" <<'PY'
import os
import sys
from pathlib import Path

site_packages = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(site_packages))

import optimizer
from optimizer import OptimizerConfig, Parameter, optimize

optimizer_file = Path(optimizer.__file__).resolve()
assert optimizer_file.is_relative_to(site_packages), (
    f"optimizer imported from {optimizer_file}, outside {site_packages}"
)


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
print(f"optimizer wheel smoke ok: {optimizer_file}")
PY
)
