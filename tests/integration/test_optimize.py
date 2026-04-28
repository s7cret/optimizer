import pytest

from optimizer import OptimizerConfig, Parameter, RunnerCapabilities, optimize


def test_optimize_grid_json(tmp_path):
    def runner(p):
        return {
            "net_profit": p["x"],
            "max_drawdown_percent": 10 - p["x"],
            "profit_factor": 1 + p["x"],
            "sharpe_ratio": p["x"],
        }

    cfg = OptimizerConfig(
        max_trials=3,
        output_dir=tmp_path,
        storage_backend="json",
        constraints={"max_drawdown_percent": {"max": 9}},
    )
    res = optimize([Parameter("x", "int", 1, 1, 3, 1)], runner, cfg)
    assert res.recommended_trial.params["x"] == 3
    assert (tmp_path / "trials.jsonl").exists()


def test_advanced_runner_request(tmp_path):
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=True, supports_seed=True)

        def __call__(self, req):
            assert req.seed == 42 and "net_profit" in req.required_metrics
            return {"net_profit": req.params["x"], "max_drawdown_percent": 1}

    res = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        R(),
        OptimizerConfig(output_dir=tmp_path, storage_backend="json"),
    )
    assert res.best_trial.id == 1


def test_parallel_thread_execution(tmp_path):
    def runner(p):
        return {"net_profit": p["x"], "max_drawdown_percent": 1}

    cfg = OptimizerConfig(output_dir=tmp_path, storage_backend="json", max_parallel=2, max_trials=4)
    res = optimize([Parameter("x", "int", 1, 1, 4, 1)], runner, cfg)
    assert res.trials_count_by_status["completed"] == 4


def _params():
    return [Parameter("x", "int", 1, 1, 4, 1), Parameter("y", "int", 1, 1, 2, 1)]


def _runner(p):
    return {
        "net_profit": p["x"] * 10 + p["y"],
        "max_drawdown_percent": 1,
        "profit_factor": 1.2,
        "sharpe_ratio": 1.0,
    }


def test_public_optimize_routes_grid_random_adaptive(tmp_path):
    cases = [
        ("grid", {"max_trials": 3}),
        ("random", {"max_trials": 3, "random_trials": 3}),
        ("adaptive_grid", {"max_trials": 5, "adaptive_grid_top_n": 1}),
    ]
    for name, extra in cases:
        cfg = OptimizerConfig(
            algorithm=name, output_dir=tmp_path / name, storage_backend="json", **extra
        )
        res = optimize(_params(), _runner, cfg)
        assert res.trials_count_by_status["completed"] >= 1
        assert res.recommended_trial is not None


def test_public_optimize_routes_genetic_and_bayesian(tmp_path):
    genetic_cfg = OptimizerConfig(
        algorithm="genetic",
        output_dir=tmp_path / "genetic",
        storage_backend="json",
        max_trials=6,
        genetic_population_size=3,
        genetic_generations=3,
        seed=11,
    )
    genetic_res = optimize(_params(), _runner, genetic_cfg)
    assert genetic_res.trials_count_by_status["completed"] >= 3

    bayesian_cfg = OptimizerConfig(
        algorithm="bayesian",
        output_dir=tmp_path / "bayesian",
        storage_backend="json",
        max_trials=6,
        bayesian_trials=6,
        bayesian_warmup_random_trials=2,
        seed=12,
    )
    bayesian_res = optimize(_params(), _runner, bayesian_cfg)
    assert bayesian_res.trials_count_by_status["completed"] == 6


def test_public_optimize_walk_forward_requires_explicit_range(tmp_path):
    cfg = OptimizerConfig(algorithm="walk_forward", output_dir=tmp_path, storage_backend="json")
    with pytest.raises(ValueError, match="requires explicit start=.*end="):
        optimize(_params(), _runner, cfg)


def test_public_optimize_routes_walk_forward_with_range_runner(tmp_path):
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=True, supports_range=True)

        def __call__(self, req):
            assert req.range is not None
            bonus = 100 if req.tags.get("walk_forward") == "test" else 0
            return {"net_profit": req.params["x"] + bonus, "max_drawdown_percent": 1}

    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=2,
        walk_forward_windows=2,
    )
    wf = optimize([Parameter("x", "int", 1, 1, 2, 1)], R(), cfg, start=0, end=100)
    assert wf["status"] == "ok"
    assert len(wf["windows"]) == 2
    assert wf["windows"][0]["test_trial"].status == "completed"


def test_public_optimize_walk_forward_requires_range_capable_runner(tmp_path):
    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=1,
        walk_forward_windows=1,
    )
    with pytest.raises(
        ValueError, match=r"requires runner\.capabilities\.supports_range|with_range"
    ):
        optimize([Parameter("x", "int", 1, 1, 1, 1)], _runner, cfg, start=0, end=100)
