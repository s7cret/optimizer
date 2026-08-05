#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
"$PYTHON" - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory

from optimizer import OptimizerConfig, Parameter, optimize


def runner(params):
    return {"net_profit": float(params["x"]), "max_drawdown": 1.0}


with TemporaryDirectory(prefix="optimizer-smoke-") as output_dir:
    result = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        runner,
        OptimizerConfig(
            max_trials=1,
            report_profiles=False,
            use_profile_auto_constraints=False,
            output_dir=Path(output_dir),
        ),
    )
assert result.status == "completed", result.status
assert result.recommended_trial is not None
print("optimizer smoke ok")
PY
