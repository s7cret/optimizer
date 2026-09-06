from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
import pytest

from optimizer import (
    OptimizerConfig,
    Parameter,
    RunnerCapabilities,
    RunnerRequest,
    RunnerResponse,
    optimize,
)
from optimizer.algorithms import bayesian, genetic, walk_forward
from optimizer.core import data_query, expression
from optimizer.core.diagnostic import Diagnostic
from optimizer.core.metric_registry import MetricRegistry
from optimizer.core.parameter_space import ParameterSpace
from optimizer.core.trial_runner import (
    _call_runner_in_process,
    _normalize_runner_response,
    run_one,
)
from optimizer.distribution import _include
from optimizer.engine import _sequential_advanced
from optimizer.quality import main as quality_main
from optimizer.results.trial import Trial
from optimizer.runners.backtest_engine import _stable_hash, BacktestEngineRunnerAdapter
from optimizer.runners.parallel import map_parallel
from optimizer.storage.json_backend import JsonStorage


def _trial(
    trial_id: int,
    params: dict[str, object],
    objective: float | None,
    status: str = "completed",
    metrics: dict[str, float | None] | None = None,
) -> Trial:
    metric_values = {
        "net_profit": objective,
        "max_drawdown_percent": 1.0,
        **(metrics or {}),
    }
    return Trial(
        trial_id,
        params,
        metric_values,
        objective,
        "maximize",
        None,
        True,
        {},
        0,
        None,
        None,
        None,
        None,
        None,
        0.01,
        status,  # type: ignore[arg-type]
        raw_objective_value=objective,
    )


class _Store(JsonStorage):
    pass


def test_bayesian_fallback_and_genetic_invalid_child(tmp_path: Path) -> None:
    class FallbackSpace(ParameterSpace):
        def random_sample(self, rng):  # type: ignore[no-untyped-def]
            return {"x": 999}

        def is_valid_combination(self, params):  # type: ignore[no-untyped-def]
            return params.get("x") != 999

    space = FallbackSpace([Parameter("x", "int", 1, 1, 2, 1)])
    cfg = OptimizerConfig(max_trials=2, seed=1)
    candidate = bayesian.propose(space, cfg, [({"x": 1}, 1.0)], set(), 0)
    assert candidate == {"x": 1}

    class RejectSecond(ParameterSpace):
        def __init__(self) -> None:
            super().__init__([Parameter("x", "int", 1, 1, 3, 1)])
            self.calls = 0

        def is_valid_combination(self, params):  # type: ignore[no-untyped-def]
            self.calls += 1
            return self.calls > 1

    reject_space = RejectSecond()
    cfg_gen = OptimizerConfig(
        max_trials=3, genetic_population_size=2, genetic_mutation_rate=1.0, seed=2
    )
    out = genetic.next_generation(
        reject_space, cfg_gen, [({"x": 1}, 1.0), ({"x": 2}, 2.0)], set(), 0
    )
    assert out


def test_walk_forward_wrappers_and_prehistory(tmp_path: Path) -> None:
    class RequestRunner:
        capabilities = RunnerCapabilities(
            supports_runner_request=True, supports_range=True
        )

        def __call__(self, request: RunnerRequest) -> dict[str, float]:
            assert request.range is not None
            if request.tags["walk_forward"] == "test":
                assert request.params.get("_effective_pre_bars") == 3
            return {
                "net_profit": float(request.params.get("x", 1)),
                "max_drawdown_percent": 1,
                "profit_factor": 1,
                "sharpe_ratio": 1,
            }

    runner = RequestRunner()
    wrapper = walk_forward.ranged_runner(runner, (1, 2), "train")
    assert wrapper({"x": 1})["net_profit"] == 1.0
    req = RunnerRequest({"x": 2}, 1, set(), set(), [], tags={"a": "b"})
    assert wrapper(req)["net_profit"] == 2.0

    with pytest.raises(ValueError):
        walk_forward.ranged_runner(object(), (1, 2), "train")
    with pytest.raises(ValueError):
        walk_forward._pre_bars_runner(object(), (1, 2), "test", 3)  # type: ignore[attr-defined]

    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=1,
        walk_forward_windows=1,
        walk_forward_train_ratio=0.5,
        walk_forward_include_prehistory=True,
        walk_forward_pre_bars=3,
    )
    result = walk_forward.run(
        [Parameter("x", "int", 1, 1, 1, 1)], runner, cfg, start=0, end=10
    )
    assert result["status"] == "ok"
    with pytest.raises(ValueError):
        walk_forward.run(
            [Parameter("x", "int", 1, 1, 1, 1)], runner, cfg, start=0, end=1
        )


def test_data_query_expression_and_response_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert data_query._lookup_nested("not a dict", "x") is None  # type: ignore[attr-defined]
    assert data_query._lookup_nested({"gates": {"b": 2}}, "a", "b") == 2  # type: ignore[attr-defined]
    risky = data_query._validate_data_query({"tick": True, "intrabar": True})
    assert risky is not None
    assert set(risky.context["risk_reasons"]) == {"intrabar", "tick"}
    assert expression.safe_eval("4 / 2", {}) == 2
    assert expression.safe_eval("2 ** 3", {}) == 8
    with pytest.raises(expression.SafeExpressionError):
        expression._apply_bin(ast.Pow(), 1e200, 2)  # type: ignore[attr-defined]

    original = expression.safe_eval
    monkeypatch.setattr(expression, "safe_eval", lambda *_a, **_kw: float("inf"))
    with pytest.raises(expression.SafeExpressionError):
        expression.safe_eval_numeric("ignored", {})
    monkeypatch.setattr(expression, "safe_eval", original)

    diagnostic = Diagnostic("D", "warn", "warning")
    normalized = _normalize_runner_response(
        {"metrics": {}, "diagnostics": [diagnostic]}, 1, "h"
    )
    assert normalized.diagnostics == [diagnostic]
    with pytest.raises(ValueError):
        _normalize_runner_response({"metrics": []}, 1, "h")
    with pytest.raises(ValueError):
        _normalize_runner_response({"metrics": {}, "hashes": []}, 1, "h")
    assert _call_runner_in_process(lambda p: p, {"x": 1}, 0) == {"x": 1}


def test_engine_advanced_direct_and_request_missing_output(tmp_path: Path) -> None:
    store = _Store(tmp_path)
    space = ParameterSpace([Parameter("x", "int", 1, 1, 2, 1)])

    def runner(params: dict[str, object]) -> dict[str, float]:
        return {
            "net_profit": float(params["x"]),
            "max_drawdown_percent": 1,
            "profit_factor": 1,
            "sharpe_ratio": 1,
        }

    cfg = OptimizerConfig(
        algorithm="genetic",
        max_trials=1,
        genetic_population_size=2,
        output_dir=tmp_path,
        storage_backend="json",
        resume=False,
    )
    trials, next_id = _sequential_advanced(space, runner, cfg, "s", "c", store, 1)
    assert len(trials) == 1 and next_id >= 2

    cfg_b = OptimizerConfig(
        algorithm="bayesian",
        max_trials=2,
        bayesian_warmup_random_trials=1,
        output_dir=tmp_path / "b",
        storage_backend="json",
        resume=False,
    )
    trials_b, _ = _sequential_advanced(
        space, runner, cfg_b, "s", "c", _Store(tmp_path / "b"), 1
    )
    assert trials_b

    class MissingEquityRunner:
        capabilities = RunnerCapabilities(
            supports_runner_request=True,
            supports_required_outputs=True,
            supported_outputs={"summary_metrics", "equity_curve"},
        )

        def __call__(self, request: RunnerRequest) -> RunnerResponse:
            return RunnerResponse(
                metrics={
                    "net_profit": 1,
                    "max_drawdown_percent": 1,
                    "profit_factor": 1,
                    "sharpe_ratio": 1,
                },
                equity_available=False,
            )

    # Inject an output requirement not currently used by stock metrics, so the
    # missing-output branch remains covered and documented for future metrics.
    monkey_metric = MetricRegistry.METRICS.get("net_profit")
    from optimizer.core.metric_registry import MetricSpec

    MetricRegistry.METRICS["net_profit"] = MetricSpec(
        "net_profit", "runner", "maximize", required_outputs={"equity_curve"}
    )
    try:
        trial = run_one(1, {"x": 1}, MissingEquityRunner(), cfg_b, "s", "c")
    finally:
        if monkey_metric is not None:
            MetricRegistry.METRICS["net_profit"] = monkey_metric
    assert trial.status == "failed"
    assert any(d.code == "RUNNER_REQUIRED_OUTPUT_MISSING" for d in trial.diagnostics)


def test_release_distribution_main_and_backtest_hash_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not _include(Path("source.zip"))
    assert not _include(Path("optimizer.egg-info") / "PKG-INFO")
    assert quality_main(["architecture", "optimizer", "--max-lines", "1"]) == 1
    # Make runpy warning coverage independent of test collection/import order.
    __import__("optimizer.distribution")
    __import__("optimizer.release")
    with pytest.warns(RuntimeWarning), pytest.raises(SystemExit):
        __import__("runpy").run_module("optimizer.distribution", run_name="__main__")
    with pytest.warns(RuntimeWarning), pytest.raises(SystemExit):
        __import__("runpy").run_module("optimizer.release", run_name="__main__")

    pkg = types.ModuleType("backtest_engine")
    core = types.ModuleType("backtest_engine.core")
    deterministic = types.ModuleType("backtest_engine.core.deterministic_hash")
    deterministic.sha256_obj = lambda value: "hash:" + str(value)
    monkeypatch.setitem(sys.modules, "backtest_engine", pkg)
    monkeypatch.setitem(sys.modules, "backtest_engine.core", core)
    monkeypatch.setitem(
        sys.modules, "backtest_engine.core.deterministic_hash", deterministic
    )
    assert _stable_hash({"a": 1}) == "hash:{'a': 1}"

    class DictResult(dict):
        status = "completed"

    class Engine:
        def run(self, strategy, *, bars, params):  # type: ignore[no-untyped-def]
            return DictResult(net_profit="bad")

    adapter = BacktestEngineRunnerAdapter(
        engine_factory=Engine, strategy=object, bars=[]
    )
    response = adapter(RunnerRequest({}, 1, {"net_profit"}, set(), []))
    assert any(
        d["code"] == "BACKTEST_ENGINE_BAD_METRIC_VALUE" for d in response.diagnostics
    )
    assert map_parallel(lambda x: x + 1, [1, 2], max_workers=1) == [2, 3]


def test_final_small_branch_edges(tmp_path: Path) -> None:
    from optimizer.core.constraints import evaluate_constraints
    from optimizer.core.metric_extractor import MetricExtractor
    from optimizer.results.leaderboard import rank_trials
    from optimizer.runners.parallel import map_parallel

    # _num() TypeError path in constraint evaluation.
    c = evaluate_constraints({"weird": object()}, {"weird": {"min": 1}})
    assert c.violations["weird"] == "missing"
    # MetricExtractor fallback for objects without a useful __dict__.
    assert MetricExtractor().extract(object()) == {}
    # Unsafe comparator branch with constants on both sides.
    with pytest.raises(expression.SafeExpressionError):
        expression.safe_eval("1 is 1", {})
    # Secondary tie branch where both secondaries exist and are equal falls back to id.
    cfg = OptimizerConfig(
        objective_secondary="profit_factor", objective_secondary_direction="maximize"
    )
    ranked = rank_trials(
        [
            _trial(2, {"x": 2}, 10, metrics={"profit_factor": 1.0}),
            _trial(1, {"x": 1}, 10, metrics={"profit_factor": 1.0}),
        ],
        cfg,
    )
    assert [t.id for t in ranked] == [1, 2]
    assert map_parallel(lambda x: x + 1, [1, 2], max_workers=2, ordered=True) == [2, 3]
    with pytest.raises(ValueError):
        optimize(
            [Parameter("x", "int", 1, 1, 1, 1)],
            lambda _p: {"net_profit": 1},
            OptimizerConfig(algorithm="walk_forward", output_dir=tmp_path),
        )
    with pytest.raises(ValueError):
        optimize(
            [Parameter("x", "int", 1, 1, 1, 1)],
            lambda _p: {"net_profit": 1},
            OptimizerConfig(algorithm="walk_forward", output_dir=tmp_path),
            start=2,
            end=1,
        )
