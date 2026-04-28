from optimizer import OptimizerConfig, Parameter, RunnerCapabilities, optimize
from optimizer.algorithms.walk_forward import run as walk_forward_run
from optimizer.analysis.monte_carlo import analyze_trial as monte_carlo_trial
from optimizer.analysis.profit_concentration import analyze_trial as concentration_trial
from optimizer.reporting.diff_report import diff
from optimizer.reporting.plot_report import export as plot_export


def params():
    return [Parameter("x", "int", 1, 1, 6, 1), Parameter("y", "int", 1, 1, 3, 1)]


def runner(p):
    x = p["x"]
    y = p["y"]
    return {
        "net_profit": -((x - 5) ** 2) + y,
        "max_drawdown_percent": 10 - y,
        "profit_factor": 1 + y / 10,
        "sharpe_ratio": x / 10,
    }


def test_genetic_optimizer_runs_real_generations(tmp_path):
    cfg = OptimizerConfig(
        algorithm="genetic",
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=10,
        genetic_population_size=4,
        genetic_generations=4,
        seed=7,
    )
    res = optimize(params(), runner, cfg)
    assert res.trials_count_by_status["completed"] > 4
    assert res.recommended_trial.objective_value is not None
    assert res.diagnostics[0].code == "ADVANCED_FEATURES_ACTIVE"


def test_bayesian_optimizer_proposes_after_warmup(tmp_path):
    cfg = OptimizerConfig(
        algorithm="bayesian",
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=8,
        bayesian_trials=8,
        bayesian_warmup_random_trials=3,
        seed=3,
    )
    res = optimize(params(), runner, cfg)
    assert res.trials_count_by_status["completed"] == 8
    assert res.recommended_trial.params["x"] in range(1, 7)


def test_analysis_features_use_saved_trade_and_train_test_data(tmp_path):
    def trade_runner(p):
        return {
            "net_profit": p["x"],
            "train_net_profit": p["x"] * 2,
            "test_net_profit": p["x"],
            "max_drawdown_percent": 1,
            "closed_trades": [{"profit": 10}, {"profit": 1}, {"profit": -2}],
        }

    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=3,
        save_backtest_result=True,
        analysis_profile="full",
        robustness_min_neighbors=1,
        monte_carlo_simulations=20,
    )
    res = optimize([Parameter("x", "int", 1, 1, 3, 1)], trade_runner, cfg)
    assert res.analysis["overfitting"]["status"] == "ok"
    assert res.analysis["profit_concentration"]["status"] == "ok"
    assert res.analysis["monte_carlo"]["status"] == "ok"
    assert res.analysis["sensitivity"]["status"] == "ok"


def test_profit_concentration_and_monte_carlo_trial_helpers():
    raw = {"closed_trades": [{"pnl": 5}, {"pnl": 1}, {"pnl": -1}]}
    assert concentration_trial(raw)["status"] == "ok"
    assert monte_carlo_trial(raw, simulations=10, seed=1)["probability_positive"] is not None


def test_diff_and_plot_html(tmp_path):
    res_a = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        lambda p: {"net_profit": 1, "max_drawdown_percent": 1},
        OptimizerConfig(output_dir=tmp_path / "a", storage_backend="json"),
    )
    res_b = optimize(
        [Parameter("x", "int", 2, 2, 2, 1)],
        lambda p: {"net_profit": 2, "max_drawdown_percent": 1},
        OptimizerConfig(output_dir=tmp_path / "b", storage_backend="json"),
    )
    assert diff(res_a, res_b)["objective_delta"] == 1
    out = tmp_path / "plot.html"
    assert plot_export(res_b, out, "html")["status"] == "ok"
    assert out.exists()


def test_walk_forward_range_runner(tmp_path):
    class R:
        capabilities = RunnerCapabilities(supports_runner_request=True, supports_range=True)

        def __call__(self, req):
            assert req.range is not None
            bonus = 1 if req.tags.get("walk_forward") == "test" else 0
            return {"net_profit": req.params["x"] + bonus, "max_drawdown_percent": 1}

    cfg = OptimizerConfig(
        algorithm="grid",
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=2,
        walk_forward_windows=2,
    )
    wf = walk_forward_run([Parameter("x", "int", 1, 1, 2, 1)], R(), cfg, start=0, end=100)
    assert wf["status"] == "ok"
    assert len(wf["windows"]) == 2
    assert wf["windows"][0]["test_trial"].status == "completed"
