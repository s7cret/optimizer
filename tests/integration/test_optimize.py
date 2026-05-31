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
        capabilities = RunnerCapabilities(
            supports_runner_request=True,
            supports_range=True,
            supports_required_outputs=True,
            supported_outputs={"summary_metrics"},
            supports_seed=True,
        )

        def __init__(self):
            self.requests = []

        def __call__(self, req):
            self.requests.append(req)
            assert req.range is not None
            assert req.required_metrics == {"net_profit"}
            assert req.required_outputs == {"summary_metrics"}
            assert req.seed == 123
            assert req.trial_id > 0
            assert req.fingerprints["parameter_space_hash"]
            bonus = 100 if req.tags.get("walk_forward") == "test" else 0
            return {"metrics": {"net_profit": req.params["x"] + bonus}}

    runner = R()
    cfg = OptimizerConfig(
        algorithm="walk_forward",
        seed=123,
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=2,
        walk_forward_windows=2,
        report_profiles=False,
        use_profile_auto_constraints=False,
    )
    wf = optimize([Parameter("x", "int", 1, 1, 2, 1)], runner, cfg, start=0, end=100)
    assert wf["status"] == "ok"
    assert len(wf["windows"]) == 2
    assert wf["windows"][0]["test_trial"].status == "completed"
    assert {req.tags["walk_forward"] for req in runner.requests} == {"train", "test"}


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


def test_public_optimize_walk_forward_without_valid_windows_fails_fast(tmp_path):
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=True, supports_range=True)

        def __call__(self, req):
            return {"net_profit": req.params["x"], "max_drawdown_percent": 1}

    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=1,
        walk_forward_windows=2,
    )
    with pytest.raises(ValueError, match="produced no valid windows"):
        optimize([Parameter("x", "int", 1, 1, 1, 1)], R(), cfg, start=0, end=1)


# ─── D5-F: Walk-Forward Prehistory ─────────────────────────────────────────────

def test_walk_forward_include_prehistory_false_uses_default_behavior(tmp_path):
    """D5-F: Default (include_prehistory=False) does not change walk-forward behavior."""
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=True, supports_range=True)

        def __call__(self, req):
            bonus = 10 if req.tags.get("walk_forward") == "test" else 0
            return {"net_profit": req.params["x"] + bonus, "max_drawdown_percent": 1}

    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        walk_forward_windows=2,
        walk_forward_include_prehistory=False,  # D5-F default
    )
    wf = optimize([Parameter("x", "int", 1, 1, 2, 1)], R(), cfg, start=0, end=100)
    assert wf["status"] == "ok"
    assert len(wf["windows"]) == 2


def test_walk_forward_include_prehistory_true_adds_pre_bars_param(tmp_path):
    """D5-F: include_prehistory=True and pre_bars sets _effective_pre_bars in test runner."""
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=True, supports_range=True)
        _received_pre_bars = None
        _received_test_range = None
        _received_test_tag = None

        def __call__(self, req):
            # D5-F: when include_prehistory is set, effective_pre_bars appears in params
            R._received_pre_bars = req.params.get("_effective_pre_bars")
            if req.params.get("_effective_pre_bars") is not None:
                R._received_test_range = req.range
                R._received_test_tag = req.tags.get("walk_forward")
            bonus = 10 if req.tags.get("walk_forward") == "test" else 0
            return {"net_profit": req.params["x"] + bonus, "max_drawdown_percent": 1}

    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        walk_forward_windows=2,
        walk_forward_include_prehistory=True,
        walk_forward_pre_bars=50,
    )
    wf = optimize([Parameter("x", "int", 1, 1, 2, 1)], R(), cfg, start=0, end=100)
    assert wf["status"] == "ok"
    assert len(wf["windows"]) == 2
    # _effective_pre_bars should be set for test runner
    assert R._received_pre_bars == 50
    assert R._received_test_range == wf["windows"][-1]["ranges"]["test"]
    assert R._received_test_tag == "test"
    assert wf["windows"][0]["test_trial"].metrics["net_profit"] >= 11


def test_walk_forward_prehistory_requires_range_aware_runner(tmp_path):
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=False, supports_range=False)

        def with_range(self, _start, _end):
            return self

        def __call__(self, params):
            return {"net_profit": params["x"], "max_drawdown_percent": 1}

    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        walk_forward_windows=1,
        walk_forward_include_prehistory=True,
        walk_forward_pre_bars=10,
    )

    with pytest.raises(ValueError, match="prehistory requires"):
        optimize([Parameter("x", "int", 1, 1, 1, 1)], R(), cfg, start=0, end=100)


def test_walk_forward_pre_bars_zero_means_no_pre_bars(tmp_path):
    """D5-F: pre_bars=0 means no pre-bars (all bars scored)."""
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=True, supports_range=True)
        _received_pre_bars = None

        def __call__(self, req):
            R._received_pre_bars = req.params.get("_effective_pre_bars")
            return {"net_profit": req.params["x"], "max_drawdown_percent": 1}

    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        walk_forward_windows=1,
        walk_forward_include_prehistory=True,
        walk_forward_pre_bars=0,  # explicitly zero
    )
    wf = optimize([Parameter("x", "int", 1, 1, 1, 1)], R(), cfg, start=0, end=100)
    assert wf["status"] == "ok"
    # pre_bars=0 should not set _effective_pre_bars
    assert R._received_pre_bars is None


def test_walk_forward_without_pre_bars_option_works_as_before(tmp_path):
    """D5-F: walk_forward without include_prehistory option works exactly as before."""
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=True, supports_range=True)

        def __call__(self, req):
            bonus = 20 if req.tags.get("walk_forward") == "test" else 0
            return {"net_profit": req.params["x"] + bonus, "max_drawdown_percent": 1}

    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        walk_forward_windows=2,
        # No walk_forward_include_prehistory set — backward compat
    )
    wf = optimize([Parameter("x", "int", 1, 1, 2, 1)], R(), cfg, start=0, end=100)
    assert wf["status"] == "ok"
    assert len(wf["windows"]) == 2
    # Test trial should be completed
    assert wf["windows"][0]["test_trial"].status == "completed"
