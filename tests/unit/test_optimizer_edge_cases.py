from __future__ import annotations

import ast
import builtins
import json
import runpy
import sys
import time
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer import (
    OptimizerConfig,
    OptimizerRunRequest,
    OptimizationConstraints,
    ObjectiveSpec,
    Parameter,
    ParameterSpace,
    RunnerCapabilities,
    RunnerRequest,
    RunnerResponse,
    StrategyRef,
    optimize,
    optimize_request,
)
from optimizer.algorithms import adaptive_grid, bayesian, genetic
from optimizer.analysis import profit_concentration, robustness, sensitivity
from optimizer.core import constraints, expression
from optimizer.core.data_query import _failed_request_result, _validate_data_query
from optimizer.core.diagnostic import Diagnostic
from optimizer.core.metric_registry import MetricRegistry
from optimizer.core.parameter_space import ParameterSpace as Space
from optimizer.core.trial_runner import (
    NormalizedRunnerResponse,
    _call_runner_in_process,
    _normalize_runner_response,
    _required_metrics,
    _select_timeout_backend,
    run_one,
)
from optimizer.distribution import (
    _include,
    build_zip,
    main as distribution_main,
    manifest,
)
from optimizer.engine import (
    _check_parallel_policy,
    _failed_worker_trial,
    _params_for,
    _run_jobs,
    _trial_from_raw,
)
from optimizer.errors import (
    ParameterValidationError,
    SafeExpressionError,
    UnsupportedFeatureError,
)
from optimizer.quality import (
    _function_fingerprints,
    architecture,
    duplicates,
    main as quality_main,
)
from optimizer.release import build_manifest, main as release_main
from optimizer.reporting.plot_report import export as plot_export
from optimizer.results.leaderboard import rank_trials
from optimizer.results.result import OptimizerRunResult
from optimizer.results.trial import Trial
from optimizer.runners.backtest_engine import BacktestEngineRunnerAdapter
from optimizer.validation import dry_run_validate


def _trial(
    trial_id: int,
    params: dict[str, object],
    objective: float | None,
    *,
    direction: str = "maximize",
    status: str = "completed",
    metrics: dict[str, float | None] | None = None,
    backtest_result: dict | None = None,
) -> Trial:
    m = {"net_profit": objective, "max_drawdown_percent": 1.0, **(metrics or {})}
    return Trial(
        trial_id,
        params,
        m,
        objective,
        direction,  # type: ignore[arg-type]
        None,
        True,
        {},
        0,
        None,
        m.get("robustness_score"),
        m.get("overfitting_score"),
        m.get("profit_concentration_score"),
        backtest_result,
        0.01,
        status,  # type: ignore[arg-type]
        raw_objective_value=objective,
    )


def _space() -> ParameterSpace:
    return ParameterSpace(
        [Parameter("x", "int", 1, 1, 3, 1), Parameter("flag", "bool", False)]
    )


def _cfg(tmp_path: Path, **kwargs: object) -> OptimizerConfig:
    data = {
        "output_dir": tmp_path,
        "storage_backend": "json",
        "resume": False,
        **kwargs,
    }
    return OptimizerConfig(**data)


class _Store:
    def __init__(self) -> None:
        self.saved: list[Trial] = []
        self.path = "memory"

    def save_trial(self, trial: Trial) -> None:
        self.saved.append(trial)


def test_algorithm_edge_branches_and_proposals() -> None:
    space = _space()
    cfg = OptimizerConfig(max_trials=2, genetic_population_size=2, seed=1)
    rng = __import__("random").Random(2)
    assert genetic.crossover({"x": 1}, {"x": 2}, rng, 0.0) == {"x": 1}
    with pytest.raises(ValueError):
        genetic.select([], rng, "tournament")
    roulette = genetic.select([({"x": 1}, 1.0), ({"x": 2}, 2.0)], rng, "roulette")
    rank = genetic.select([({"x": 1}, 1.0), ({"x": 2}, 2.0)], rng, "rank")
    assert roulette["x"] in {1, 2}
    assert rank["x"] in {1, 2}

    class InvalidSpace(ParameterSpace):
        def is_valid_combination(self, params):  # type: ignore[no-untyped-def]
            return False

    invalid = InvalidSpace([Parameter("x", "int", 1, 1, 2, 1)])
    assert genetic.initial_population(invalid, cfg) == []
    assert bayesian.warmup(invalid, OptimizerConfig(max_trials=2, seed=1)) == []
    assert bayesian.propose(space, cfg, [], set(), 0) is None
    seen = {json.dumps(p, sort_keys=True, default=str) for p in space.generate_grid()}
    assert (
        bayesian.propose(space, cfg, [({"x": 1, "flag": False}, 1.0)], seen, 0) is None
    )

    trials = [
        _trial(1, {"x": 1, "flag": False}, 10),
        _trial(2, {"x": 2, "flag": False}, 9),
    ]
    cfg.adaptive_grid_top_n = 2
    cfg.adaptive_grid_max_new_points_per_round = 1
    assert len(adaptive_grid.refine(space, trials, cfg)) == 1


def test_parameter_space_validation_and_expression_edges() -> None:
    with pytest.raises(ParameterValidationError):
        ParameterSpace([Parameter("bad-name", "int", 1, 1, 2, 1)])
    with pytest.raises(ParameterValidationError):
        ParameterSpace([Parameter("x", "int", 1), Parameter("x", "int", 2)])
    with pytest.raises(ParameterValidationError):
        ParameterSpace([Parameter("x", "float", 1.0, 0.0, 1.0, 0.0)]).values_for(
            Parameter("x", "float", 1.0, 0.0, 1.0, 0.0)
        )
    assert list(_space().generate_grid(max_combinations=1)) == [{"x": 1, "flag": False}]
    assert (
        Space(
            [Parameter("x", "int", 1, 1, 2, 1)], ["missing > 0"]
        ).is_valid_combination({"x": 1})
        is False
    )
    assert _space().neighbors({"x": 99, "flag": False})

    assert expression.safe_eval("1 < x <= 3 != 4", {"x": 2}, mode="boolean") is True
    assert expression.safe_eval("false or true", {}, mode="boolean") is True
    assert expression.safe_eval("5 % 2", {}) == 1
    with pytest.raises(SafeExpressionError):
        expression.safe_eval("5 % 0", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval("1 << 2", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval("[1, 2]", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval("float('inf')", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval_numeric("true", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval_bool("1 + 2", {})
    with pytest.raises(SafeExpressionError):
        expression.safe_eval_numeric("'not-number'", {})
    with pytest.raises(SafeExpressionError):
        expression._finite(float("inf"))  # type: ignore[attr-defined]
    with pytest.raises(SafeExpressionError):
        expression._apply_bin(ast.MatMult(), 1, 2)  # type: ignore[attr-defined]


def test_constraints_profiles_and_required_metrics() -> None:
    assert "max_drawdown_percent" in constraints.auto_constraints_for_profile(
        "conservative"
    )
    assert "net_profit" in constraints.auto_constraints_for_profile("aggressive")
    assert "profit_factor" in constraints.auto_constraints_for_profile("balanced")
    assert constraints.auto_constraints_for_profile("unknown") == {}
    cfg = SimpleNamespace(
        constraints={"net_profit": {"min": 10.0}},
        use_profile_auto_constraints=False,
        constraints_merge_mode="custom_only",
        selection_mode="balanced",
    )
    assert constraints.merge_constraints(cfg) == {"net_profit": {"min": 10.0}}
    cfg.use_profile_auto_constraints = True
    cfg.constraints_merge_mode = "merge_auto_and_custom"
    merged = constraints.merge_constraints(cfg)
    assert merged["net_profit"]["min"] == 10.0

    result = constraints.evaluate_constraints(
        {"net_profit": 4, "drawdown": 6, "missing": None, "eqm": 3, "neqm": 5},
        {
            "net_profit": {"min": 5, "penalty": 2, "hard": True},
            "drawdown": {"max": 5, "hard": False},
            "missing": {"min": 1},
            "eqm": {"eq": 4},
            "neqm": {"neq": 5},
            "ignored": {"min": 1, "stage": "selection"},
        },
        trial_id=7,
        params_hash="abc",
    )
    assert not result.passed
    assert not result.hard_passed
    assert "drawdown" in result.soft_violations
    assert result.penalty > 0
    assert result.diagnostics[0].trial_id == 7
    assert constraints.evaluate_constraints({}, None).passed is True
    required = constraints.required_constraint_metrics(
        SimpleNamespace(
            constraints={"return_drawdown_ratio": {"min": 1}},
            use_profile_auto_constraints=False,
            constraints_merge_mode="custom_only",
            selection_mode="custom",
        )
    )
    assert {"net_profit", "max_drawdown"}.issubset(required)


def test_metric_registry_leaderboard_and_analysis_fallbacks() -> None:
    reg = MetricRegistry()
    assert reg.extract_expression_metrics(
        "net_profit + max_drawdown_percent + true"
    ) == {
        "net_profit",
        "max_drawdown_percent",
    }
    assert (
        reg.get_required_statistics_profile(
            {"net_profit", "profit_concentration_score"}
        )
        == "minimal"
    )
    assert "net_profit" in reg.profile_required_metrics(["best_profit", "unknown"])

    cfg = OptimizerConfig(
        objective_secondary="profit_factor", objective_secondary_direction="maximize"
    )
    a = _trial(1, {}, 10, metrics={"profit_factor": 1.0})
    b = _trial(2, {}, 10, metrics={"profit_factor": 2.0})
    c = _trial(3, {}, 10, metrics={})
    ranked = rank_trials([a, c, b], cfg)
    assert [t.id for t in ranked] == [2, 1, 3]

    assert sensitivity.analyze([_trial(1, {"x": 1}, 1), _trial(2, {"x": 1}, 2)])[
        "importance"
    ] == {"x": 0.0}
    fallback = profit_concentration.analyze(
        [_trial(1, {}, 1, metrics={"profit_concentration_score": 0.4})]
    )
    assert fallback["status"] == "ok"

    class TradeObj:
        profit = 5

    assert (
        profit_concentration.analyze_trial(SimpleNamespace(closed_trades=[TradeObj()]))[
            "status"
        ]
        == "ok"
    )
    robust = robustness.analyze(
        [_trial(1, {"x": 1, "flag": False}, 10), _trial(2, {"x": 2, "flag": False}, 9)],
        _space(),
        min_neighbors=2,
    )
    assert robust["status"] == "insufficient_data"


def test_plot_report_dependency_and_png_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = OptimizerRunResult(None, None, None, [])
    assert plot_export(empty)["status"] == "insufficient_data"

    result = OptimizerRunResult(
        _trial(1, {}, 1), "best", _trial(1, {}, 1), [_trial(1, {}, 1)]
    )
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("matplotlib"):
            raise ImportError("missing matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert (
        plot_export(result, tmp_path / "out.png", "png")["status"]
        == "dependency_missing"
    )
    monkeypatch.setattr(builtins, "__import__", original_import)

    pyplot = types.SimpleNamespace(
        figure=lambda **_kw: None,
        plot=lambda *_a, **_kw: None,
        xlabel=lambda *_a, **_kw: None,
        ylabel=lambda *_a, **_kw: None,
        tight_layout=lambda: None,
        savefig=lambda path: Path(path).write_text("plot", encoding="utf-8"),
        close=lambda: None,
    )
    matplotlib = types.ModuleType("matplotlib")
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    png = tmp_path / "plot.png"
    assert plot_export(result, png, "png")["status"] == "ok"
    assert png.read_text(encoding="utf-8") == "plot"


def test_distribution_quality_release_and_module_entrypoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "pkg"
    (root / "optimizer").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "optimizer" / "version.py").write_text(
        '__version__ = "4.0.0"\n', encoding="utf-8"
    )
    for name in [
        "README.md",
        "CHANGELOG.md",
        "docs/README.md",
        "docs/ARCHITECTURE.md",
        "docs/DEVELOPMENT.md",
        "docs/RELEASE_4_0.md",
    ]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("doc\n", encoding="utf-8")
    (root / "optimizer" / "a.py").write_text(
        "def alpha():\n    x = 1\n    y = 2\n    z = 3\n    q = 4\n    return x + y + z + q\n",
        encoding="utf-8",
    )
    (root / "optimizer" / "b.py").write_text(
        "def alpha():\n    x = 1\n    y = 2\n    z = 3\n    q = 4\n    return x + y + z + q\n",
        encoding="utf-8",
    )
    (root / "optimizer" / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "ignored.py").write_text("x=1\n", encoding="utf-8")
    (root / "cache.pyc").write_text("x", encoding="utf-8")
    assert not _include(Path("dist/file.py"))
    assert not _include(Path("x.pyc"))
    assert not _include(Path("pkg.egg-info"))
    assert list(_function_fingerprints(root / "optimizer" / "bad.py")) == []
    dup = duplicates(root / "optimizer")
    assert dup.duplicate_group_count == 1
    assert quality_main(["duplicates", str(root / "optimizer")]) == 1
    assert architecture(root / "optimizer", max_lines=1).oversized_count >= 2
    assert (
        quality_main(["architecture", str(root / "optimizer"), "--max-lines", "100"])
        == 0
    )

    dist = manifest(root)
    assert dist.package_version == "4.0.0"
    zip_path = build_zip(root, tmp_path / "out.zip")
    assert zip_path.exists()
    assert distribution_main(["manifest", "--root", str(root)]) == 0
    assert (
        distribution_main(
            ["build-zip", "--root", str(root), "--output", str(tmp_path / "cli.zip")]
        )
        == 0
    )

    rel = build_manifest(root)
    assert rel.ok is False  # duplicate functions are intentional in this fixture
    json_path = tmp_path / "release.json"
    assert release_main(["--root", str(root), "--json", str(json_path)]) == 1
    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["package_version"] == "4.0.0"
    )

    with pytest.raises(SystemExit) as exc, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("optimizer.__main__", run_name="__main__")
    assert exc.value.code == 2  # argparse reports missing command
    assert "usage:" in capsys.readouterr().err


def test_data_query_and_optimize_request_boundaries(tmp_path: Path) -> None:
    safe = {"mode": "historical", "gates": {}}
    assert _validate_data_query(safe) is None
    proven = {
        "data_type": "tick",
        "gates": {
            "tvRealtimeBoundary": "proven",
            "duTickCompleteness": True,
            "intrabarOrderFill": "PROVEN",
        },
    }
    assert _validate_data_query(proven) is None
    risky = {"items": [{"allow_realtime": "yes"}], "lower_tf": "1"}
    diagnostic = _validate_data_query(risky)
    assert diagnostic is not None
    assert set(diagnostic.context["risk_reasons"]) == {"intrabar", "realtime"}
    request = OptimizerRunRequest(
        run_id="bad",
        strategy_ref=StrategyRef(id="dummy.py"),
        parameter_space=ParameterSpace([Parameter("x", "int", 1, 1, 1, 1)]),
        objective=ObjectiveSpec(metric="net_profit"),
        constraints=OptimizationConstraints(),
        data_query=risky,
    )
    failed = _failed_request_result(request, diagnostic, tmp_path)
    assert failed.status == "failed"
    assert failed.data_query is risky
    assert (
        optimize_request(request, lambda _p: {"net_profit": 1}, _cfg(tmp_path)).status
        == "failed"
    )


def test_engine_policies_resume_parallel_and_advanced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diags: list[Diagnostic] = []
    cfg = _cfg(tmp_path, max_parallel=999999, max_parallel_over_cpu_policy="warn")
    monkeypatch.setattr("optimizer.engine.os.cpu_count", lambda: 1)
    _check_parallel_policy(cfg, diags)
    assert diags and diags[0].code == "MAX_PARALLEL_OVER_CPU"
    cfg.max_parallel_over_cpu_policy = "allow"
    _check_parallel_policy(cfg, [])
    cfg.max_parallel_over_cpu_policy = "error"
    with pytest.raises(ValueError):
        _check_parallel_policy(cfg, [])

    with pytest.raises(UnsupportedFeatureError):
        list(_params_for(_space(), _cfg(tmp_path, algorithm="unknown")))
    assert _trial_from_raw(None) is None
    raw = _trial(5, {"x": 1}, 1, metrics={"m": 1}).to_dict()
    raw["diagnostics"] = [{"code": "D", "message": "m", "severity": "warning"}]
    raw["unknown"] = "ignored"
    assert _trial_from_raw(raw).diagnostics[0].code == "D"  # type: ignore[union-attr]

    failed = _failed_worker_trial(
        9, {"x": 1}, RuntimeError("boom"), _cfg(tmp_path), "s", "c"
    )
    assert failed.status == "failed"
    assert failed.diagnostics[0].code == "OPTIMIZER_WORKER_EXCEPTION"

    def boom_run_one(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("worker exploded")

    monkeypatch.setattr("optimizer.engine.run_one", boom_run_one)
    store = _Store()
    cfg_parallel = _cfg(
        tmp_path, max_parallel=2, parallel_backend="thread", ordered_results=True
    )
    trials = _run_jobs(
        [(2, {"x": 2}), (1, {"x": 1})], lambda p: p, cfg_parallel, "s", "c", store
    )
    assert [t.id for t in trials] == [2, 1]
    assert all(t.status == "failed" for t in trials)
    from optimizer.core.trial_runner import run_one as real_run_one

    monkeypatch.setattr("optimizer.engine.run_one", real_run_one)

    def runner(params: dict[str, object]) -> dict[str, float]:
        return {
            "net_profit": float(params["x"]),
            "max_drawdown_percent": 1.0,
            "profit_factor": 1.1,
            "sharpe_ratio": 0.1,
        }

    bayes = optimize(
        [Parameter("x", "int", 1, 1, 2, 1)],
        runner,
        _cfg(
            tmp_path / "bayes",
            algorithm="bayesian",
            max_trials=3,
            bayesian_warmup_random_trials=2,
        ),
    )
    assert bayes.trials
    genetic_result = optimize(
        [Parameter("x", "int", 1, 1, 2, 1)],
        runner,
        _cfg(
            tmp_path / "genetic",
            algorithm="genetic",
            max_trials=2,
            genetic_generations=1,
            genetic_population_size=0,
        ),
    )
    assert genetic_result.status == "failed"


def test_trial_runner_contract_errors_and_timeout(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, objective="net_profit", timeout_per_trial_sec=0)
    assert _required_metrics(cfg) >= {"net_profit", "max_drawdown_percent"}
    assert (
        NormalizedRunnerResponse({"a": 1}, {"content_hash": "h"}, []).hash(
            "content_hash", object()
        )
        == "h"
    )
    assert NormalizedRunnerResponse({}, {}, []).summary_metrics_available is False

    with pytest.raises(ValueError):
        _normalize_runner_response({"contract": "bad", "metrics": {}}, 1, "h")
    with pytest.raises(ValueError):
        _normalize_runner_response(
            {"contract": "pine.optimizer_runner.v1", "metrics": "bad"}, 1, "h"
        )
    with pytest.raises(ValueError):
        _normalize_runner_response(
            {"contract": "pine.optimizer_runner.v1", "metrics": {}, "hashes": "bad"},
            1,
            "h",
        )
    normalized = _normalize_runner_response(
        {
            "contract": "pine.optimizer_runner.v1",
            "metrics": {"net_profit": 1},
            "hashes": {"x": 1},
            "diagnostics": [{"code": "R", "message": "warn", "severity": "warning"}],
        },
        1,
        "h",
    )
    assert normalized.is_contract_response is True
    assert normalized.diagnostics[0].code == "R"

    class UnsupportedOutputs:
        capabilities = RunnerCapabilities(
            supports_required_outputs=True, supported_outputs=set()
        )

        def __call__(self, request):  # type: ignore[no-untyped-def]
            return {"net_profit": 1}

    unsupported = run_one(1, {"x": 1}, UnsupportedOutputs(), cfg, "s", "c")
    assert unsupported.status == "failed"
    assert unsupported.diagnostics[0].code == "RUNNER_REQUIRED_OUTPUT_UNSUPPORTED"

    class RequestRunner:
        capabilities = RunnerCapabilities(
            supports_runner_request=True,
            supports_required_outputs=True,
            supported_outputs={"summary_metrics"},
            supports_seed=True,
        )

        def __call__(self, request: RunnerRequest) -> RunnerResponse:
            assert request.seed == cfg.seed
            return RunnerResponse(metrics={"net_profit": 3.0}, trades_available=False)

    missing_outputs = run_one(2, {"x": 1}, RequestRunner(), cfg, "s", "c")
    assert missing_outputs.status == "failed"
    assert any(
        d.code == "RUNNER_REQUIRED_METRICS_MISSING" for d in missing_outputs.diagnostics
    )

    class ErrorDiagRunner:
        capabilities = RunnerCapabilities(supports_runner_request=True)

        def __call__(self, request):  # type: ignore[no-untyped-def]
            return RunnerResponse(
                metrics={},
                diagnostics=[{"code": "E", "message": "bad", "severity": "error"}],
            )

    error_trial = run_one(3, {"x": 1}, ErrorDiagRunner(), cfg, "s", "c")
    assert error_trial.status == "failed"
    assert "runner returned error diagnostics" in (error_trial.error_message or "")

    def bad_basic(_params):
        return {
            "net_profit": 1,
            "max_drawdown_percent": 1,
            "profit_factor": 1,
            "sharpe_ratio": 1,
            "return_drawdown_ratio": 1,
        }

    cfg_hint = _cfg(tmp_path / "hint", early_stop_enabled=True, timeout_per_trial_sec=0)
    basic = run_one(4, {"x": 1}, bad_basic, cfg_hint, "s", "c")
    assert basic.status == "completed"
    assert any(d.code == "BASIC_RUNNER_CONTRACT_USED" for d in basic.diagnostics)

    def sleepy(_params):
        time.sleep(0.05)
        return {"net_profit": 1}

    timeout_cfg = _cfg(tmp_path / "timeout", timeout_per_trial_sec=0.001)
    timed = run_one(5, {"x": 1}, sleepy, timeout_cfg, "s", "c")
    assert timed.status == "failed"
    assert any(d.code == "TRIAL_TIMEOUT" for d in timed.diagnostics)

    bad_cfg = _cfg(
        tmp_path / "process", timeout_backend="process", timeout_per_trial_sec=1
    )
    with pytest.raises(ValueError):
        _select_timeout_backend(bad_cfg, lambda x: x, {"x": object()}, [], 1, "h")
    auto_cfg = _cfg(tmp_path / "auto", timeout_backend="auto", timeout_per_trial_sec=1)
    warnings: list[Diagnostic] = []
    assert (
        _select_timeout_backend(
            auto_cfg, lambda x: x, {"x": object()}, warnings, 1, "h"
        )
        == "thread"
    )
    assert warnings[0].code == "RUNNER_TIMEOUT_THREAD_FALLBACK"
    assert _call_runner_in_process(lambda p: {"ok": p}, {"x": 1}, 1) == {"ok": {"x": 1}}
    with pytest.raises(RuntimeError):
        _call_runner_in_process(
            lambda _p: (_ for _ in ()).throw(RuntimeError("boom")), {}, 1
        )


def test_backtest_adapter_and_dry_run_edges(tmp_path: Path) -> None:
    class ResultObj:
        content_hash_value = "content"
        data_fingerprint = "data"
        strategy_fingerprint = "strategy"
        runtime_fingerprint = "runtime"
        config_snapshot = {"a": 1}
        status = "failed"
        warnings = [{"message": "warn"}]
        errors = ["fatal"]
        closed_trades = []
        equity_curve = []
        net_profit = "bad"
        max_drawdown_percent = 1.0

    class Engine:
        def run(self, strategy, *, bars, params, effective_pre_bars=None):  # type: ignore[no-untyped-def]
            assert effective_pre_bars == 5
            return ResultObj()

    adapter = BacktestEngineRunnerAdapter(
        engine_factory=Engine,
        strategy=object(),
        bars=[{"close": 1}],
        static_params={"_effective_pre_bars": 5},
    )
    response = adapter(
        RunnerRequest(
            params={},
            trial_id=1,
            required_metrics={"net_profit", "max_drawdown_percent"},
            required_outputs=set(),
            early_stop_conditions=[],
        )
    )
    assert any(
        d["code"] == "BACKTEST_ENGINE_BAD_METRIC_VALUE" for d in response.diagnostics
    )
    assert any(
        d["code"] == "BACKTEST_ENGINE_RUN_NOT_COMPLETED" for d in response.diagnostics
    )
    bad_contract = adapter(
        RunnerRequest(
            params={},
            trial_id=1,
            required_metrics=set(),
            required_outputs=set(),
            early_stop_conditions=[],
            contract="bad",
        )
    )
    assert bad_contract.diagnostics[0]["code"] == "RUNNER_REQUEST_CONTRACT_MISMATCH"

    invalid = dry_run_validate(
        [Parameter("x", "int", 1, 1, 2, 1)],
        OptimizerConfig(output_dir=tmp_path, cross_constraints=["x > 10"]),
        sample_limit=1,
    )
    assert invalid.status == "invalid"
    assert invalid.invalid_samples
    assert any(d.code == "NO_VALID_PARAM_COMBINATIONS" for d in invalid.diagnostics)
