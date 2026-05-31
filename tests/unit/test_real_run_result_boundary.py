import json
import subprocess
import sys
from pathlib import Path

from optimizer import (
    DryRunValidationResult,
    ObjectiveSpec,
    OptimizerConfig,
    OptimizationConstraints,
    OptimizerRunRequest,
    OptimizerRunResult,
    Parameter,
    ParameterSpace,
    RunnerCapabilities,
    StrategyRef,
    optimize,
    optimize_request,
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
    assert result.status == "completed"
    assert result.best_params == {"x": 1}
    assert result.best_score == 1
    assert result.trials == tuple(result.all_trials)
    assert result.artifact_path is not None
    assert set(result.trials_count_by_status) == {"completed", "failed"}
    assert {t.status for t in result.all_trials} == {"completed"}


def test_optimizer_run_request_carries_explicit_production_contract(tmp_path):
    request = OptimizerRunRequest(
        run_id="run-1",
        strategy_ref=StrategyRef("strategy-1", version="v1"),
        parameter_space=ParameterSpace([Parameter("x", "int", 2, 1, 2, 1)]),
        data_query={"symbol": "BTCUSDT", "timeframe": "15"},
        objective=ObjectiveSpec("net_profit", "maximize"),
        constraints=OptimizationConstraints(cross_constraints=("x == 2",)),
    )

    result = optimize_request(
        request,
        lambda p: {"net_profit": p["x"], "max_drawdown_percent": 1},
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            use_profile_auto_constraints=False,
        ),
    )

    assert isinstance(result, OptimizerRunResult)
    assert result.run_id == "run-1"
    assert result.status == "completed"
    assert result.best_params == {"x": 2}
    assert result.best_score == 2
    assert result.data_query == {"symbol": "BTCUSDT", "timeframe": "15"}
    assert result.artifact_path == tmp_path / "trials.jsonl"


def test_unproven_realtime_data_query_is_rejected_before_runner_call(tmp_path):
    request = OptimizerRunRequest(
        run_id="run-risky",
        strategy_ref=None,
        parameter_space=ParameterSpace([Parameter("x", "int", 1, 1, 1, 1)]),
        data_query={"symbol": "BTCUSDT", "realtime": True, "duTickCompleteness": "blocked"},
    )

    def runner(_params):  # pragma: no cover - must not be called
        raise AssertionError("runner should not be called")

    result = optimize_request(
        request,
        runner,
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            use_profile_auto_constraints=False,
        ),
    )

    assert result.status == "failed"
    assert result.trials == ()
    assert result.data_query == request.data_query
    assert any(d.code == "UNPROVEN_REALTIME_INTRABAR_DATA_QUERY" for d in result.diagnostics)


def test_proven_realtime_data_query_is_allowed(tmp_path):
    request = OptimizerRunRequest(
        run_id="run-proven",
        strategy_ref=None,
        parameter_space=ParameterSpace([Parameter("x", "int", 1, 1, 1, 1)]),
        data_query={
            "symbol": "BTCUSDT",
            "realtime": True,
            "oracle_gates": {
                "tvRealtimeBoundary": "proven",
                "duTickCompleteness": "proven",
                "intrabarOrderFill": "proven",
            },
        },
    )

    result = optimize_request(
        request,
        lambda p: {"net_profit": p["x"], "max_drawdown_percent": 1},
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            use_profile_auto_constraints=False,
        ),
    )

    assert result.status == "completed"
    assert result.best_params == {"x": 1}


def test_objective_expression_does_not_require_default_objective_metric(tmp_path):
    class R:
        capabilities = RunnerCapabilities(
            supports_runner_request=True,
            supports_required_outputs=True,
            supported_outputs={"summary_metrics"},
        )

        def __call__(self, req):
            assert "net_profit" not in req.required_metrics
            return {
                "metrics": {"custom_alpha": 3.0},
                "trades_available": False,
                "equity_available": False,
            }

    result = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        R(),
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            use_profile_auto_constraints=False,
            objective_expression="custom_alpha * 2",
        ),
    )

    assert result.status == "completed"
    assert result.best_score == 6.0


def test_failed_only_run_has_failed_result_contract(tmp_path):
    def runner(_params):
        raise RuntimeError("boom")

    result = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        runner,
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            use_profile_auto_constraints=False,
        ),
    )

    assert result.status == "failed"
    assert result.best_params is None
    assert result.best_score is None
    assert result.trials_count_by_status == {"completed": 0, "failed": 1}
    assert all(t.status == "failed" for t in result.trials)


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


def test_production_package_has_no_notimplemented_runtime_contracts():
    root = Path(__file__).resolve().parents[2] / "optimizer"
    offenders = [
        str(p.relative_to(root.parent))
        for p in root.rglob("*.py")
        if "NotImplementedError" in p.read_text()
    ]

    assert offenders == []
