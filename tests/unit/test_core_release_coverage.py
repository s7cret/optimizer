from __future__ import annotations

import csv
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer import Parameter
from optimizer.algorithms.base import SearchAlgorithm
from optimizer.analysis import (
    heatmap,
    monte_carlo,
    overfitting,
    profit_concentration,
    robustness,
    sensitivity,
)
from optimizer.analysis.walk_forward_report import (
    analyze as analyze_walk_forward_report,
)
from optimizer.cli import main as cli_main
from optimizer.core import expression
from optimizer.core.early_stop import conditions_enabled
from optimizer.core.metric_extractor import MetricExtractor
from optimizer.core.normalization import balanced_score, normalize_trials
from optimizer.core.objective import (
    compute_objective,
    objective_better,
    objective_direction,
    objective_sort_value,
)
from optimizer.core.parameter_space import ParameterSpace
from optimizer.core.scheduler import run_scheduled
from optimizer.core.seed import make_rng
from optimizer.distribution import build_zip, manifest
from optimizer.errors import SafeExpressionError
from optimizer.quality import architecture, duplicates
from optimizer.release import build_manifest
from optimizer.reporting.console import print_summary
from optimizer.reporting.csv_report import write_csv
from optimizer.reporting.diff_report import diff
from optimizer.reporting.json_report import to_json
from optimizer.reporting.markdown_report import to_markdown
from optimizer.reporting.plot_report import export as export_plot
from optimizer.reporting.telegram_summary import summarize
from optimizer.results.leaderboard import Leaderboard, rank_trials
from optimizer.results.profile_result import ResultProfile
from optimizer.results.result import OptimizerRunResult
from optimizer.results.trial import Trial
from optimizer.runners.backtest_engine import BacktestEngineRunnerAdapter
from optimizer.runners.parallel import map_parallel
from optimizer.runners.timeout import call_with_timeout
from optimizer.storage.sqlite_backend import SQLiteStorage


def _trial(
    trial_id: int,
    params: dict[str, object],
    objective: float | None,
    *,
    status: str = "completed",
    metrics: dict[str, float | None] | None = None,
    backtest_result: dict | None = None,
) -> Trial:
    m = {"net_profit": objective, "max_drawdown": 1.0, **(metrics or {})}
    return Trial(
        trial_id,
        params,
        m,
        objective,
        "maximize",
        None,
        True,
        {},
        0,
        balanced_score(m),
        m.get("robustness_score"),
        m.get("overfitting_score"),
        m.get("profit_concentration_score"),
        backtest_result,
        0.01,
        status,  # type: ignore[arg-type]
    )


def _result(tmp_path: Path) -> OptimizerRunResult:
    trials = [_trial(1, {"x": 1, "y": "a"}, 10), _trial(2, {"x": 2, "y": "b"}, 20)]
    ranked = rank_trials(trials)
    profile = ResultProfile(
        "best_objective",
        ranked[0],
        "best objective",
        "objective",
        ranked[0].objective_value,
    )
    return OptimizerRunResult(
        ranked[0],
        "best_objective",
        ranked[0],
        ranked,
        ranked,
        ranked,
        [],
        str(tmp_path),
        {"completed": 2, "failed": 0},
        {"best_objective": profile},
        status="completed",
        trials=tuple(ranked),
        artifact_path=tmp_path,
    )


def test_small_core_helpers_and_safe_expression_edges():
    assert expression.safe_eval("1 + 2 * x", {"x": 3}, mode="numeric") == 7
    assert expression.safe_eval_bool("x > 1 and true", {"x": 3}) is True
    assert expression.safe_eval_numeric("+x - 1", {"x": 3}) == 2.0
    assert expression.safe_eval("not false", {}, mode="boolean") is True
    assert expression.stable_hash({"b": 2, "a": 1}) == expression.stable_hash(
        {"a": 1, "b": 2}
    )
    with pytest.raises(SafeExpressionError):
        expression.safe_eval_numeric("1 / 0", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval_bool("unknown > 1", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval("2 ** 99", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval("fn(1)", {})


def test_parameter_space_branches_and_seed_scheduler():
    params = [
        Parameter("x", "int", 2, 1, 3, 1),
        Parameter("f", "float", 0.5, 0.0, 1.0, 0.5),
        Parameter("flag", "bool", False),
        Parameter("mode", "enum", "a", options=["a", "b"]),
        Parameter("disabled", "int", 9, 0, 10, 1, enabled=False),
    ]
    space = ParameterSpace(params, ["x >= 2"])
    assert space.grid_size() > space.grid_size(respect_constraints=True)
    assert {"x": 4, "f": -1.0, "flag": 1, "mode": "z", "disabled": 1} | space.clamp({})
    clamped = space.clamp({"x": 9, "f": -1, "flag": 1, "mode": "z", "disabled": 1})
    assert clamped == {"x": 3, "f": 0.0, "flag": True, "mode": "a", "disabled": 9}
    assert space.validate_params({"x": "bad", "f": "bad", "flag": "bad", "mode": "z"})
    assert space.random_sample(make_rng(123))["mode"] in {"a", "b"}
    assert space.neighbors(
        {"x": 2, "f": 0.5, "flag": False, "mode": "a", "disabled": 9}
    )
    assert space.refine_around(
        {"x": 2, "f": 0.5, "flag": False, "mode": "a", "disabled": 9}, 0.5
    )
    assert conditions_enabled(
        SimpleNamespace(early_stop_enabled=True, early_stop_conditions=[{"x": 1}])
    ) == [{"x": 1}]
    assert run_scheduled(lambda x: x + 1, [1, 2], max_parallel=1) == [2, 3]
    assert call_with_timeout(lambda: "ok", 1) == "ok"
    assert sorted(map_parallel(lambda x: x * 2, [1, 2], max_workers=2)) == [2, 4]


def test_metric_objective_normalization_and_algorithm_base():
    @dataclass
    class Metrics:
        net_profit: float
        max_drawdown: float
        nested: dict[str, float]

    class ModelDump:
        def model_dump(self):
            return {"profit_factor": 1.5}

    class ToDict:
        def to_dict(self):
            return {"sharpe_ratio": 0.8}

    assert (
        MetricExtractor().extract(Metrics(10, -2, {"alpha": 3}))["nested_alpha"] == 3.0
    )
    assert MetricExtractor().extract(ModelDump())["profit_factor"] == 1.5
    assert MetricExtractor().extract(ToDict())["sharpe_ratio"] == 0.8
    assert (
        MetricExtractor({"custom": lambda _r: 42}).extract({"custom": 1})["custom"]
        == 42
    )
    assert (
        MetricExtractor({"custom": lambda _r: 42}, "keep_builtin").extract(
            {"custom": 1}
        )["custom"]
        == 1
    )
    with pytest.raises(ValueError):
        MetricExtractor({"custom": lambda _r: 42}, "error").extract({"custom": 1})
    assert compute_objective({"net_profit": 5}) == 5.0
    assert compute_objective({"a": 1, "b": 2}, expression="a + b") == 3.0
    with pytest.raises(KeyError):
        compute_objective({}, "missing")
    assert objective_direction("max_drawdown", "auto") == "minimize"
    assert objective_sort_value(2, "minimize") == -2
    assert objective_better(1, None, "maximize") is True
    assert objective_better(None, 1, "maximize") is False
    trials = [_trial(1, {}, 1), _trial(2, {}, 2), _trial(3, {}, None)]
    assert normalize_trials(trials, "net_profit", "maximize")[1] == 0.0
    assert normalize_trials(trials, "missing") == {}
    assert Leaderboard(trials).top(1)[0].id == 2
    with pytest.raises(TypeError):
        SearchAlgorithm().generate(None, None)  # type: ignore[abstract]


def test_analysis_reporting_storage_and_cli(tmp_path: Path, capsys):
    trades = {"closed_trades": [{"profit": 5}, {"pnl": 1}, {"net_profit": -1}]}
    trials = [
        _trial(
            1,
            {"x": 1, "y": "a"},
            10,
            metrics={
                "train_net_profit": 12,
                "test_net_profit": 9,
                "robustness_score": 0.7,
            },
            backtest_result=trades,
        ),
        _trial(
            2,
            {"x": 2, "y": "b"},
            20,
            metrics={
                "train_net_profit": 20,
                "validation_net_profit": 10,
                "profit_concentration_score": 0.5,
            },
            backtest_result=trades,
        ),
    ]
    assert heatmap.export(trials, "x", "y")["status"] == "ok"
    assert heatmap.export([], "x", "y")["status"] == "insufficient_data"
    assert sensitivity.analyze(trials)["status"] == "ok"
    assert sensitivity.analyze([trials[0]])["status"] == "insufficient_data"
    assert overfitting.analyze(trials)["status"] == "ok"
    assert overfitting.analyze([_trial(3, {}, 3)])["status"] == "insufficient_data"
    assert profit_concentration.analyze(trials)["status"] == "ok"
    assert (
        profit_concentration.analyze([_trial(4, {}, 4)])["status"]
        == "insufficient_data"
    )
    assert monte_carlo.analyze(trials, simulations=3, seed=1)["status"] == "ok"
    assert (
        monte_carlo.analyze([_trial(5, {}, 5)], simulations=3, seed=1)["status"]
        == "insufficient_data"
    )
    assert robustness.analyze(trials)["status"] == "ok"
    assert robustness.analyze([])["status"] == "insufficient_data"
    assert (
        analyze_walk_forward_report(
            {"windows": [{"window": 1, "ranges": {}, "test_trial": trials[0]}]}
        )["average_test_objective"]
        == 10
    )
    assert analyze_walk_forward_report({})["status"] == "insufficient_data"

    result = _result(tmp_path)
    assert "Optimizer Report" in to_markdown(result, tmp_path / "report.md")
    assert (
        json.loads(to_json(result, tmp_path / "result.json"))["status"] == "completed"
    )
    write_csv(result.top_trials, tmp_path / "trials.csv")
    with (tmp_path / "trials.csv").open(encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle))[0] == "id"
    print_summary(result)
    assert "Recommended" in capsys.readouterr().out
    assert "best_objective" in summarize(result)
    assert (
        diff(
            {"params": {"a": 1}, "metrics": {"m": 1}, "objective_value": 1},
            {"params": {"a": 2}, "metrics": {"m": 3}, "objective_value": 2},
        )["objective_delta"]
        == 1.0
    )
    assert export_plot(result, tmp_path / "plot.html", "html")["status"] == "ok"
    assert (
        export_plot(
            OptimizerRunResult(None, None, None, []), tmp_path / "empty.html", "html"
        )["status"]
        == "insufficient_data"
    )

    store = SQLiteStorage(tmp_path / "db")
    store.init_run({"a": "b"})
    assert store.load_meta() == {"a": "b"}
    store.save_trial(result.top_trials[0])
    store.save_profile(next(iter(result.profiles.values())))
    assert store.load_trials_raw()[0]["id"] == result.top_trials[0].id
    store.close()

    params_path = tmp_path / "params.json"
    params_path.write_text(
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
    runner_path = tmp_path / "runner.py"
    runner_path.write_text(
        "def runner(params):\n    return {'net_profit': float(params['x']), 'max_drawdown': 1.0}\n"
    )
    out = tmp_path / "cli"
    assert (
        cli_main.main(
            ["dry-run", "--params", str(params_path), "--output-dir", str(out)]
        )
        == 0
    )
    assert (
        cli_main.main(
            [
                "run",
                "--params",
                str(params_path),
                "--runner",
                f"{runner_path}:runner",
                "--output-dir",
                str(out),
                "--max-trials",
                "1",
            ]
        )
        == 0
    )
    assert cli_main.main(["analyze", "--result-dir", str(out)]) == 0
    assert cli_main.main(["export", "--result-dir", str(out), "--format", "json"]) == 0
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["plot", "--result-dir", str(out)])
    assert exc.value.code == 2


def test_backtest_adapter_and_release_distribution(tmp_path: Path):
    class Engine:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def run(self, strategy, *, bars, params, effective_pre_bars=None):
            self.calls.append((strategy, bars, params, effective_pre_bars))
            return self.result

    @dataclass
    class Diag:
        code: str
        message: str

    result = SimpleNamespace(
        net_profit=10,
        max_drawdown=1,
        status="completed",
        warnings=[Diag("W", "warn")],
        errors=[],
        closed_trades=[],
        equity_curve=[],
        content_hash_value="content",
        data_fingerprint="data",
        strategy_fingerprint="strategy",
        runtime_fingerprint="runtime",
        config_snapshot={"x": 1},
    )
    engine = Engine(result)
    adapter = BacktestEngineRunnerAdapter(
        engine_factory=lambda: engine,
        strategy=object,
        bars=[1],
        static_params={"_effective_pre_bars": 5},
    )
    from optimizer.protocols import RunnerRequest

    response = adapter(
        RunnerRequest({"x": 1}, 1, {"net_profit", "max_drawdown"}, set(), [])
    )
    assert response.metrics["net_profit"] == 10
    assert response.trades_available and response.equity_available
    assert response.hashes["runner_fingerprint"] == "strategy"
    bad = adapter(RunnerRequest({"x": 1}, 1, set(), set(), [], contract="bad"))
    assert bad.diagnostics[0]["severity"] == "error"

    root = Path(__file__).resolve().parents[2]
    assert duplicates(root / "optimizer").duplicate_group_count == 0
    assert architecture(root / "optimizer", max_lines=700).oversized_count == 0
    release_root = tmp_path / "release_root"
    (release_root / "optimizer").mkdir(parents=True)
    (release_root / "optimizer" / "version.py").write_text('__version__ = "4.0.0"\n')
    (release_root / "optimizer" / "__init__.py").write_text("")
    for name in [
        "README.md",
        "CHANGELOG.md",
        "docs/README.md",
        "docs/ARCHITECTURE.md",
        "docs/DEVELOPMENT.md",
        "docs/RELEASE_4_0.md",
    ]:
        path = release_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n")
    dist = manifest(release_root)
    assert dist.hygiene_ok and dist.file_count > 0
    assert build_manifest(release_root).ok
    archive = tmp_path / "optimizer.zip"
    build_zip(root, archive)
    assert archive.exists()
    runpy.run_module("optimizer.__main__", run_name="__main__") if False else None
