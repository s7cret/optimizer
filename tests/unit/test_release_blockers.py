import json
import time

import pytest

from optimizer import OptimizerConfig, Parameter, dry_run_validate, optimize
from optimizer.core.diagnostic import Diagnostic
from optimizer.core.expression import safe_eval_numeric
from optimizer.core.metric_registry import MetricRegistry
from optimizer.errors import ParameterValidationError, StorageError


def process_timeout_slow_runner(_params):
    time.sleep(1.0)
    return {"net_profit": 1}


def process_timeout_fast_runner(params):
    return {"net_profit": params["x"], "max_drawdown_percent": 1}


def test_diagnostic_public_signature_order():
    d = Diagnostic("C", "msg", "error", 7, "hash", "metric", {"x": 1})
    assert d.to_dict() == {
        "code": "C",
        "message": "msg",
        "severity": "error",
        "trial_id": 7,
        "params_hash": "hash",
        "metric": "metric",
        "context": {"x": 1},
    }


def test_minimize_objective_ranking_and_profiles(tmp_path):
    def runner(p):
        return {
            "max_drawdown_percent": p["x"],
            "net_profit": 10 - p["x"],
            "profit_factor": 1.1,
            "sharpe_ratio": 1,
        }

    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend="json",
        objective="max_drawdown_percent",
        objective_direction="auto",
        max_trials=3,
        use_profile_auto_constraints=False,
    )
    res = optimize([Parameter("x", "int", 1, 1, 3, 1)], runner, cfg)
    assert res.best_trial.params["x"] == 1
    assert [t.params["x"] for t in res.top_trials] == [1, 2, 3]
    assert res.profiles["best_drawdown"].trial.params["x"] == 1


def test_constraints_eq_neq_soft_penalty_and_no_pass_diagnostic(tmp_path):
    def runner(p):
        return {
            "net_profit": p["x"],
            "profit_factor": 1.0,
            "max_drawdown_percent": p["x"],
        }

    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=2,
        use_profile_auto_constraints=False,
        constraint_mode="both",
        constraints={
            "max_drawdown_percent": {"max": 0, "hard": True},
            "profit_factor": {"eq": 1.0, "hard": False},
            "net_profit": {"neq": 1, "hard": False, "penalty": 2},
        },
    )
    res = optimize([Parameter("x", "int", 1, 1, 2, 1)], runner, cfg)
    assert all(not t.passed_constraints for t in res.all_trials)
    assert any(d.code == "NO_TRIALS_PASSED_CONSTRAINTS" for d in res.diagnostics)
    assert any(
        d.code == "CONSTRAINT_VIOLATION" for t in res.all_trials for d in t.diagnostics
    )


def test_resume_uses_params_hash_and_loads_prior_trials(tmp_path):
    calls = []

    def runner(p):
        calls.append(p["x"])
        return {
            "net_profit": p["x"],
            "max_drawdown_percent": 1,
            "profit_factor": 1.1,
            "sharpe_ratio": 1,
        }

    params = [Parameter("x", "int", 1, 1, 2, 1)]
    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=2,
        use_profile_auto_constraints=False,
    )
    optimize(params, runner, cfg)
    second = optimize(params, runner, cfg)
    assert len(calls) == 2
    assert second.trials_count_by_status["completed"] == 2
    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert len({r["params_hash"] for r in rows}) == 2


def test_resume_with_metadata_but_missing_trials_fails_closed(tmp_path):
    def runner(p):
        return {
            "net_profit": p["x"],
            "max_drawdown_percent": 1,
            "profit_factor": 1.1,
            "sharpe_ratio": 1,
        }

    params = [Parameter("x", "int", 1, 1, 1, 1)]
    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=1,
        use_profile_auto_constraints=False,
    )
    optimize(params, runner, cfg)
    (tmp_path / "trials.jsonl").unlink()

    with pytest.raises(StorageError, match="no persisted optimizer trials"):
        optimize(params, runner, cfg)


def test_invalid_grid_combos_are_dry_run_validation_not_production_trials(tmp_path):
    res = optimize(
        [Parameter("x", "int", 1, 1, 2, 1), Parameter("y", "int", 1, 1, 2, 1)],
        lambda p: {
            "net_profit": p["x"] + p["y"],
            "max_drawdown_percent": 1,
            "profit_factor": 1.1,
            "sharpe_ratio": 1,
        },
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            max_trials=4,
            use_profile_auto_constraints=False,
        ),
        cross_constraints=["x == y"],
    )
    assert res.trials_count_by_status == {"completed": 2, "failed": 0}
    assert {t.status for t in res.all_trials} == {"completed"}
    assert len(res.all_trials) == 2

    dry = dry_run_validate(
        [Parameter("x", "int", 1, 1, 2, 1), Parameter("y", "int", 1, 1, 2, 1)],
        cross_constraints=["x == y"],
    )
    assert dry.status == "invalid"
    assert dry.valid_combinations == 2
    assert dry.invalid_combinations == 2
    assert any(d.code == "INVALID_PARAM_COMBINATIONS" for d in dry.diagnostics)


def test_invalid_optimizer_config_fails_before_production_run():
    with pytest.raises(ParameterValidationError, match="max_trials"):
        OptimizerConfig(max_trials=0)
    with pytest.raises(ParameterValidationError, match="walk_forward_windows"):
        OptimizerConfig(walk_forward_windows=0)
    with pytest.raises(ParameterValidationError, match="walk_forward_train_ratio"):
        OptimizerConfig(walk_forward_train_ratio=1.0)
    with pytest.raises(ParameterValidationError, match="walk_forward_pre_bars"):
        OptimizerConfig(walk_forward_pre_bars=-1)


def test_metric_registry_expression_and_profile_requirements():
    reg = MetricRegistry()
    assert reg.extract_expression_metrics("net_profit / max_drawdown_percent") == {
        "net_profit",
        "max_drawdown_percent",
    }
    assert {"net_profit", "max_drawdown_percent"} <= reg.profile_required_metrics(
        ["best_balanced"]
    )
    assert reg.get_required_statistics_profile(["net_profit"]) == "minimal"


def test_safe_numeric_expression_rejects_bad_math():
    with pytest.raises(Exception):
        safe_eval_numeric("net_profit / z", {"net_profit": 1, "z": 0})
    with pytest.raises(Exception):
        safe_eval_numeric("2 ** 99", {})


def test_timeout_returns_failed_status_without_waiting_for_runner_completion(tmp_path):
    def slow(_p):
        time.sleep(0.3)
        return {"net_profit": 1}

    t0 = time.perf_counter()
    res = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        slow,
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            timeout_per_trial_sec=0.05,
            use_profile_auto_constraints=False,
        ),
    )
    assert time.perf_counter() - t0 < 0.25
    assert res.trials_count_by_status["failed"] == 1
    assert any(d.code == "TRIAL_TIMEOUT" for t in res.all_trials for d in t.diagnostics)


def test_process_timeout_backend_terminates_picklable_runner(tmp_path):
    t0 = time.perf_counter()
    res = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        process_timeout_slow_runner,
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            timeout_per_trial_sec=0.05,
            timeout_backend="process",
            use_profile_auto_constraints=False,
        ),
    )
    assert time.perf_counter() - t0 < 0.5
    assert res.trials_count_by_status["failed"] == 1
    assert any(d.code == "TRIAL_TIMEOUT" for t in res.all_trials for d in t.diagnostics)


def test_auto_timeout_backend_falls_back_for_local_runner(tmp_path):
    def local_runner(params):
        return {"net_profit": params["x"], "max_drawdown_percent": 1}

    res = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        local_runner,
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            timeout_per_trial_sec=1.0,
            timeout_backend="auto",
            use_profile_auto_constraints=False,
        ),
    )
    assert res.trials_count_by_status["completed"] == 1
    assert any(
        d.code == "RUNNER_TIMEOUT_THREAD_FALLBACK"
        for trial in res.all_trials
        for d in trial.diagnostics
    )


def test_process_timeout_backend_runs_picklable_runner(tmp_path):
    res = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        process_timeout_fast_runner,
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            timeout_per_trial_sec=1.0,
            timeout_backend="process",
            use_profile_auto_constraints=False,
        ),
    )
    assert res.trials_count_by_status["completed"] == 1
    assert res.best_trial.metrics["net_profit"] == 1


def test_baseline_comparison_warns_when_recommendation_worse(tmp_path):
    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend="json",
        baseline_params={"x": 5},
        max_trials=1,
        use_profile_auto_constraints=False,
    )
    res = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        lambda p: {
            "net_profit": p["x"],
            "max_drawdown_percent": 1,
            "profit_factor": 1.1,
            "sharpe_ratio": 1,
        },
        cfg,
    )
    assert res.baseline_trial.params["x"] == 5
    assert res.baseline_comparison["recommended_worse_than_baseline"] is True
    assert any(d.code == "RECOMMENDED_WORSE_THAN_BASELINE" for d in res.diagnostics)
