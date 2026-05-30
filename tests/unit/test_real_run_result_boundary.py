import json
import subprocess
import sys
from pathlib import Path

from optimizer import (
    DryRunValidationResult,
    OptimizerConfig,
    OptimizerRunResult,
    Parameter,
    dry_run_validate,
    optimize,
)


def test_public_optimize_returns_run_result_not_dry_run_result(tmp_path):
    result = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        lambda p: {"net_profit": p["x"], "max_drawdown_percent": 1},
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            use_profile_auto_constraints=False,
        ),
    )

    assert isinstance(result, OptimizerRunResult)
    assert not isinstance(result, DryRunValidationResult)
    assert set(result.trials_count_by_status) == {"completed", "failed"}
    assert {t.status for t in result.all_trials} == {"completed"}


def test_dry_run_cli_writes_validation_result_without_production_result(tmp_path):
    params_file = tmp_path / "params.json"
    params_file.write_text(
        json.dumps(
            [
                {
                    "name": "x",
                    "param_type": "int",
                    "default": 1,
                    "min_val": 1,
                    "max_val": 1,
                    "step": 1,
                }
            ]
        )
    )
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimizer.cli.main",
            "dry-run",
            "--params",
            str(params_file),
            "--output-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "valid"
    assert (out_dir / "dry_run_validation.json").exists()
    assert not (out_dir / "result.json").exists()
    assert not (out_dir / "trials.jsonl").exists()


def test_production_package_has_no_legacy_status_literals():
    root = Path(__file__).resolve().parents[2] / "optimizer"
    text = "\n".join(p.read_text() for p in root.rglob("*.py"))

    assert '"skipped"' not in text
    assert 'status = "timeout"' not in text
    assert "OptimizerResult" not in text
