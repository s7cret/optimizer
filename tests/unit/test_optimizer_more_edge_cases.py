from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer import (
    OptimizerConfig,
    Parameter,
    RunnerCapabilities,
    RunnerRequest,
    RunnerResponse,
    optimize,
)
from optimizer.algorithms import bayesian, genetic, grid_search, walk_forward
from optimizer.analysis.heatmap import export as heatmap_export
from optimizer.analysis.overfitting import analyze as overfitting_analyze
from optimizer.analysis.walk_forward_report import (
    analyze as walk_forward_report_analyze,
)
from optimizer.cli import main as cli_main
from optimizer.core import expression
from optimizer.core.metric_extractor import MetricExtractor
from optimizer.core.metric_registry import MetricRegistry
from optimizer.core.normalization import normalize_trials
from optimizer.core.objective import objective_better, objective_sort_value
from optimizer.core.parameter_space import ParameterSpace
from optimizer.core.trial_runner import _runner_process_entry, run_one
from optimizer.distribution import main as distribution_main
from optimizer.release import main as release_main
from optimizer.reporting.diff_report import diff
from optimizer.reporting.json_report import to_json
from optimizer.results.result import OptimizerRunResult
from optimizer.results.trial import Trial
from optimizer.runners.backtest_engine import BacktestEngineRunnerAdapter
from optimizer.selection.baseline import baseline_comparison, baseline_diagnostic
from optimizer.selection.selector import choose_recommended
from optimizer.storage.base import StorageBackend
from optimizer.storage.checkpoint import check_resume
from optimizer.storage.json_backend import JsonStorage
from optimizer.storage.sqlite_backend import SQLiteStorage
from optimizer.errors import FingerprintMismatchError, ParameterValidationError


def _trial(
    trial_id: int,
    objective: float | None,
    *,
    params: dict[str, object] | None = None,
    status: str = "completed",
    passed: bool = True,
    metrics: dict[str, float | None] | None = None,
    direction: str = "maximize",
) -> Trial:
    m = {"net_profit": objective, "max_drawdown_percent": 1.0, **(metrics or {})}
    return Trial(
        trial_id,
        params or {"x": trial_id},
        m,
        objective,
        direction,  # type: ignore[arg-type]
        None,
        passed,
        {} if passed else {"net_profit": "too low"},
        0 if passed else 1,
        None,
        m.get("robustness_score"),
        m.get("overfitting_score"),
        m.get("profit_concentration_score"),
        None,
        0.01,
        status,  # type: ignore[arg-type]
        raw_objective_value=objective,
    )


def _result(trials: list[Trial]) -> OptimizerRunResult:
    return OptimizerRunResult(
        trials[0] if trials else None,
        "best",
        trials[0] if trials else None,
        trials,
        all_trials=trials,
        artifact_path=Path("out"),
    )


def test_small_uncovered_algorithm_and_analysis_edges() -> None:
    space = ParameterSpace([Parameter("mode", "enum", "a", options=["a", "b"])])
    assert bayesian._distance({"mode": "a"}, {"mode": "b"}, space) == 1.0  # type: ignore[attr-defined]
    cfg = OptimizerConfig(
        max_trials=2, bayesian_trials=2, bayesian_warmup_random_trials=1
    )
    assert list(bayesian.generate(space, cfg))
    assert list(grid_search.generate(space, cfg))
    assert list(
        genetic.generate(
            space, OptimizerConfig(max_trials=1, genetic_population_size=1)
        )
    )

    assert (
        heatmap_export([_trial(1, None, status="failed")], "x", "y")["status"]
        == "insufficient_data"
    )
    assert (
        overfitting_analyze(
            [_trial(1, 1, metrics={"train_net_profit": 1, "test_net_profit": 2})]
        )["status"]
        == "ok"
    )
    assert (
        walk_forward_report_analyze(
            {"windows": [{"window": 1, "ranges": {}, "test_trial": None}]}
        )["average_test_objective"]
        is None
    )
    assert walk_forward.windows(10, 5, 1, 0.5) == []
    assert walk_forward.windows(0, 2, 4, 0.9) == []


def test_metric_extractor_registry_normalization_objective_edges() -> None:
    class Plain:
        def __init__(self) -> None:
            self.keep = 1
            self.text = "bad"
            self._private = 2

    assert MetricExtractor().extract(None) == {}
    assert MetricExtractor().extract(Plain())["keep"] == 1.0
    assert MetricExtractor().extract(
        {"nested": {"bad": object(), "ok": 2}, "flag": True}
    ) == {"nested_ok": 2.0}
    with pytest.raises(ValueError):
        MetricExtractor({"net_profit": lambda _r: 1.0}, "error").extract(
            {"net_profit": 2}
        )

    reg = MetricRegistry()
    assert reg.extract_expression_metrics(None) == set()
    assert "net_profit" in reg.get_required_runner_metrics({"return_drawdown_ratio"})
    assert "summary_metrics" in reg.get_required_outputs({"return_drawdown_ratio"})
    assert reg.required_outputs({"net_profit"}) == {"summary_metrics"}
    assert reg.get_required_statistics_profile({"net_profit"}) == "minimal"

    trials = [
        _trial(1, 1, metrics={"m": 1}),
        _trial(2, 2, metrics={"m": 3}),
        _trial(3, 3, metrics={}),
    ]
    assert normalize_trials(trials, "m", "minimize") == {1: 1.0, 2: 0.0, 3: None}
    assert normalize_trials(
        [_trial(1, 1, metrics={"m": 2}), _trial(2, 2, metrics={"m": 2})], "m"
    ) == {1: 1.0, 2: 1.0}
    assert objective_sort_value(None, "maximize") == float("-inf")
    assert objective_better(1, 2, "minimize") is True
    with pytest.raises(expression.SafeExpressionError):
        expression.safe_eval_numeric("1e309", {})
    with pytest.raises(expression.SafeExpressionError):
        expression.safe_eval("1 ** 13", {})


def test_storage_cli_and_reporting_edges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class ConcreteStorage(StorageBackend):
        def init_run(self, *a, **k):  # type: ignore[no-untyped-def]
            return super().init_run(*a, **k)

        def save_trial(self, trial):  # type: ignore[no-untyped-def]
            return super().save_trial(trial)

        def load_trials(self):  # type: ignore[no-untyped-def]
            return super().load_trials()

    store_base = ConcreteStorage()
    assert store_base.init_run({}) is None
    assert store_base.save_trial(_trial(1, 1)) is None
    assert store_base.load_trials() is None
    assert store_base.close() is None

    json_dir = tmp_path / "json"
    store = JsonStorage(json_dir)
    store.init_run({"a": 1})
    store.save_trial(_trial(1, 1))
    store.save_trial(_trial(1, 2))
    assert len(store.load_trials_raw()) == 1
    assert check_resume(store, {"a": 1}) == {"a": 1}
    with pytest.raises(FingerprintMismatchError):
        check_resume(store, {"a": 2})
    assert check_resume(store, {"a": 2}, force=True) == {"a": 1}

    sqlite_dir = tmp_path / "sqlite"
    sqlite = SQLiteStorage(sqlite_dir)
    sqlite.init_run({"s": 1})
    sqlite.save_trial(_trial(1, 1))
    assert sqlite.load_meta() == {"s": 1}
    sqlite.close()

    assert cli_main.main(["analyze", "--result-dir", str(json_dir)]) == 0
    assert "top_objective_rows" in capsys.readouterr().out
    assert (
        cli_main.main(["export", "--result-dir", str(json_dir), "--format", "json"])
        == 0
    )
    assert (json_dir / "trials_export.json").exists()
    assert (
        cli_main.main(["export", "--result-dir", str(json_dir), "--format", "csv"]) == 0
    )
    assert (json_dir / "trials_export.csv").exists()
    assert (
        cli_main.main(["export", "--result-dir", str(json_dir), "--format", "markdown"])
        == 0
    )
    assert (json_dir / "trials_export.md").exists()
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["champions", "--result-dir", str(json_dir)])
    assert exc.value.code == 2

    assert diff(None, None)["status"] == "ok"
    assert json.loads(to_json(_result([_trial(1, 1)])))["recommended_trial"]["id"] == 1


def test_parameter_config_and_validation_edges(tmp_path: Path) -> None:
    with pytest.raises(ParameterValidationError):
        OptimizerConfig(max_trials=0)
    with pytest.raises(ParameterValidationError):
        OptimizerConfig(max_parallel=0)
    with pytest.raises(ParameterValidationError):
        OptimizerConfig(walk_forward_windows=0)
    with pytest.raises(ParameterValidationError):
        OptimizerConfig(walk_forward_train_ratio=1.0)
    with pytest.raises(ParameterValidationError):
        OptimizerConfig(walk_forward_pre_bars=-1)
    with pytest.raises(ParameterValidationError):
        ParameterSpace([Parameter("x", "enum", "a")])
    with pytest.raises(ParameterValidationError):
        ParameterSpace([Parameter("x", "bool", "yes")])
    with pytest.raises(ParameterValidationError):
        ParameterSpace([Parameter("x", "int", 1, 1, None, 1)])
    diagnostics = ParameterSpace([Parameter("x", "int", 1, 1, 2, 1)]).validate_params(
        {}
    )
    assert diagnostics[0].code == "PARAM_MISSING"


def test_trial_runner_process_queue_and_engine_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out: mp.Queue = mp.Queue(maxsize=1)
    _runner_process_entry(out, lambda payload: {"ok": payload}, {"x": 1})
    assert out.get(timeout=1)[0] == "ok"
    out.close()
    out.cancel_join_thread()

    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend="json",
        resume=False,
        max_trials=2,
        fail_fast=True,
    )

    def failing(_params: dict[str, object]) -> dict[str, float]:
        raise RuntimeError("bad trial")

    res = optimize([Parameter("x", "int", 1, 1, 2, 1)], failing, cfg)
    assert res.trials_count_by_status["failed"] == 1
    assert any(d.code == "NO_TRIALS_PASSED_CONSTRAINTS" for d in res.diagnostics)

    cfg_min = OptimizerConfig(
        output_dir=tmp_path / "min",
        storage_backend="json",
        resume=False,
        min_completed_trials=3,
        max_trials=1,
    )
    res_min = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)], lambda _p: {"net_profit": 1}, cfg_min
    )
    assert res_min.status == "failed"
    assert any(d.code == "MIN_COMPLETED_TRIALS_NOT_MET" for d in res_min.diagnostics)

    cfg_base = OptimizerConfig(
        output_dir=tmp_path / "base",
        storage_backend="json",
        resume=False,
        baseline_params={"x": 1},
        max_trials=1,
    )
    res_base = optimize(
        [Parameter("x", "int", 2, 2, 2, 1)],
        lambda p: {"net_profit": float(p["x"])},
        cfg_base,
    )
    assert res_base.baseline_trial is not None
    assert baseline_comparison(res_base.baseline_trial, res_base.recommended_trial)
    assert baseline_diagnostic(None, None) == {}


def test_runner_contract_output_missing_and_backtest_adapter_edges(
    tmp_path: Path,
) -> None:
    class MissingOutputRunner:
        capabilities = RunnerCapabilities(
            supports_runner_request=True,
            supports_required_outputs=True,
            supported_outputs={"summary_metrics", "closed_trades"},
        )

        def __call__(self, request: RunnerRequest) -> RunnerResponse:
            return RunnerResponse(
                metrics={
                    "net_profit": 1.0,
                    "max_drawdown_percent": 1.0,
                    "profit_factor": 1.0,
                    "sharpe_ratio": 1.0,
                    "profit_concentration_score": 0.2,
                },
                trades_available=False,
            )

    cfg = OptimizerConfig(
        output_dir=tmp_path,
        storage_backend="json",
        resume=False,
        objective="profit_concentration_score",
        report_profiles=False,
    )
    trial = run_one(1, {"x": 1}, MissingOutputRunner(), cfg, "s", "c")
    assert trial.status == "failed"
    assert any(d.code == "RUNNER_REQUIRED_OUTPUT_MISSING" for d in trial.diagnostics)

    class EngineNoPre:
        def run(self, strategy, *, bars, params):  # type: ignore[no-untyped-def]
            return {"net_profit": 1.0, "content_hash": "dict"}

    adapter = BacktestEngineRunnerAdapter(
        engine_factory=EngineNoPre,
        strategy=object,
        bars=[],
        static_params={"_effective_pre_bars": 10},
    )
    with pytest.raises(ValueError, match="effective_pre_bars"):
        adapter(RunnerRequest({}, 1, {"net_profit"}, set(), []))

    class BadSignatureEngine:
        run = object()

    bad_adapter = BacktestEngineRunnerAdapter(
        engine_factory=BadSignatureEngine,
        strategy=object,
        bars=[],
        static_params={"_effective_pre_bars": 1},
    )
    with pytest.raises(ValueError, match="verify engine warmup"):
        bad_adapter(RunnerRequest({}, 1, set(), set(), []))


def test_recommendation_and_main_cli_edges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = SimpleNamespace(trial=_trial(1, 1, passed=False))
    trial, name = choose_recommended(
        {"best_passed_constraints": profile}, "best_after_constraints"
    )
    assert trial is None and name == "best_passed_constraints"
    assert choose_recommended({}, "unknown") == (None, "best_passed_constraints")

    release_root = tmp_path / "release_root"
    (release_root / "optimizer").mkdir(parents=True)
    (release_root / "optimizer" / "version.py").write_text('__version__ = "4.0.0"\n')
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
    assert distribution_main(["manifest", "--root", str(release_root)]) == 0
    assert release_main(["--root", str(release_root)]) == 0
    assert "package_version" in capsys.readouterr().out
