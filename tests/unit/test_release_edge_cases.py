from __future__ import annotations

import queue
import runpy
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer import OptimizerConfig, Parameter, RunnerRequest
from optimizer.algorithms import genetic, grid_search, walk_forward
from optimizer.algorithms.base import SearchAlgorithm
from optimizer.analysis import heatmap, overfitting, profit_concentration
from optimizer.core import data_query
from optimizer.core.metric_registry import MetricRegistry, MetricSpec
from optimizer.core.parameter_space import ParameterSpace
from optimizer.core.trial_runner import (
    _call_runner_in_process,
    _normalize_runner_response,
    _response_diagnostics,
    run_one,
)
from optimizer.engine import _sequential_advanced, optimize
from optimizer.protocols import (
    AdvancedBacktestRunner,
    BacktestRunner,
    RangeAwareBacktestRunner,
    RunnerCapabilities,
)
from optimizer.results.result import OptimizerRunResult
from optimizer.results.trial import Trial
from optimizer.selection.baseline import baseline_comparison
from optimizer.storage.json_backend import JsonStorage


def _trial(
    trial_id: int,
    params: dict[str, object],
    objective: float | None,
    *,
    status: str = "completed",
    direction: str = "maximize",
) -> Trial:
    return Trial(
        trial_id,
        params,
        {"net_profit": objective, "max_drawdown_percent": 1.0},
        objective,
        direction,  # type: ignore[arg-type]
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


def _space() -> ParameterSpace:
    return ParameterSpace([Parameter("x", "int", 1, 1, 3, 1)])


def _runner(params: dict[str, object]) -> dict[str, float]:
    return {
        "net_profit": float(params.get("x", 0)),
        "max_drawdown_percent": 1.0,
        "profit_factor": 1.0,
        "sharpe_ratio": 1.0,
    }


class _Store(JsonStorage):
    pass


def test_entrypoints_protocol_stubs_and_grid_overflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import optimizer.__main__ as _optimizer_main_module

    assert _optimizer_main_module.main is not None
    monkeypatch.setattr("optimizer.cli.main.main", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["python -m optimizer"])
    with pytest.raises(SystemExit) as main_exit, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("optimizer.__main__", run_name="__main__")
    assert main_exit.value.code == 0

    monkeypatch.setattr(
        sys, "argv", ["python -m optimizer.quality", "duplicates", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as quality_exit, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("optimizer.quality", run_name="__main__")
    assert quality_exit.value.code == 0

    assert BacktestRunner.__call__(object(), {}) is None
    assert (
        AdvancedBacktestRunner.__call__(
            object(), RunnerRequest({}, 1, set(), set(), [])
        )
        is None
    )
    assert RangeAwareBacktestRunner.__call__(object(), {}) is None
    assert RangeAwareBacktestRunner.with_range(object(), 1, 2) is None
    assert SearchAlgorithm.generate(object(), None, None) is None

    cfg = OptimizerConfig(grid_max_combinations=1, grid_overflow_policy="error")
    with pytest.raises(ValueError, match="grid size"):
        list(grid_search.generate(_space(), cfg))


def test_analysis_metric_registry_parameter_and_storage_edges(tmp_path: Path) -> None:
    assert (
        heatmap.export([_trial(1, {"x": 1}, 1.0)], "x", "y")["status"]
        == "insufficient_data"
    )
    assert (
        overfitting.analyze([_trial(1, {}, 1.0, status="failed")])["status"]
        == "insufficient_data"
    )
    assert (
        profit_concentration.analyze_trial(
            SimpleNamespace(closed_trades=[SimpleNamespace(profit=3.0)])
        )["status"]
        == "ok"
    )
    assert (
        profit_concentration.analyze_trial({"closed_trades": [{}]})["trade_count"] == 1
    )

    original = MetricRegistry.METRICS.get("custom_full")
    MetricRegistry.METRICS["custom_full"] = MetricSpec(
        "custom_full", "external", required_statistics_profile="full"
    )
    try:
        assert (
            MetricRegistry().get_required_statistics_profile(["custom_full"]) == "full"
        )
    finally:
        if original is None:
            MetricRegistry.METRICS.pop("custom_full", None)
        else:
            MetricRegistry.METRICS["custom_full"] = original

    constrained = ParameterSpace(
        [Parameter("x", "int", 2, 1, 3, 1)], cross_constraints=["x == 2"]
    )
    assert constrained.refine_around({"x": 2}, 1.0) == [{"x": 2}]

    result = OptimizerRunResult(None, None, None, [])
    assert result.to_dict()["artifact_path"] is None
    assert (
        baseline_comparison(_trial(1, {}, None), _trial(2, {}, 1.0))["objective_delta"]
        is None
    )

    store = JsonStorage(tmp_path)
    row = _trial(1, {"x": 1}, 1.0).to_dict()
    store.path.write_text(
        "\n".join([__import__("json").dumps(row), __import__("json").dumps(row)]) + "\n"
    )
    store.save_trial(_trial(1, {"x": 2}, 2.0))
    assert len(store.load_trials_raw()) == 1

    assert data_query._lookup_nested({"gates": {"other": True}}, "missing") is None  # type: ignore[attr-defined]
    proven = {
        "source": "live",
        "oracleGates": {
            "tvRealtimeBoundary": "proven",
            "duTickCompleteness": True,
            "intrabarOrderFill": "proven",
        },
    }
    assert data_query._validate_data_query(proven) is None  # type: ignore[attr-defined]


def test_trial_runner_process_and_basic_runner_no_hint_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQueue:
        def __init__(self, *, empty: bool = True) -> None:
            self.empty = empty
            self.closed = False

        def get_nowait(self):  # type: ignore[no-untyped-def]
            if self.empty:
                raise queue.Empty
            return ("ok", {"value": 1})

        def close(self) -> None:
            self.closed = True

        def cancel_join_thread(self) -> None:
            pass

    class TimeoutProc:
        exitcode = None

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.calls = 0
            self.killed = False

        def start(self) -> None:
            pass

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            pass

        def is_alive(self) -> bool:
            self.calls += 1
            return self.calls <= 2

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            self.killed = True

    class EmptyProc:
        def __init__(self, *args, exitcode: int = 0, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.exitcode = exitcode

        def start(self) -> None:
            pass

        def join(self, _timeout=None) -> None:  # type: ignore[no-untyped-def]
            pass

        def is_alive(self) -> bool:
            return False

    class FakeContext:
        def __init__(self, proc_cls) -> None:  # type: ignore[no-untyped-def]
            self.proc_cls = proc_cls

        def Queue(self, maxsize=1):  # type: ignore[no-untyped-def]
            return FakeQueue()

        def Process(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self.proc_cls(*args, **kwargs)

    monkeypatch.setattr(
        "optimizer.core.trial_runner.mp.get_all_start_methods", lambda: ["fork"]
    )
    monkeypatch.setattr(
        "optimizer.core.trial_runner.mp.get_context",
        lambda _name: FakeContext(TimeoutProc),
    )
    with pytest.raises(__import__("concurrent.futures").futures.TimeoutError):
        _call_runner_in_process(lambda payload: payload, {"x": 1}, 0.001)

    monkeypatch.setattr(
        "optimizer.core.trial_runner.mp.get_context",
        lambda _name: FakeContext(EmptyProc),
    )
    with pytest.raises(RuntimeError, match="without a result"):
        _call_runner_in_process(lambda payload: payload, {"x": 1}, 0.001)

    class ExitOneContext(FakeContext):
        def Process(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return EmptyProc(*args, exitcode=1, **kwargs)

    monkeypatch.setattr(
        "optimizer.core.trial_runner.mp.get_context",
        lambda _name: ExitOneContext(EmptyProc),
    )
    with pytest.raises(RuntimeError, match="exited with code 1"):
        _call_runner_in_process(lambda payload: payload, {"x": 1}, 0.001)

    assert _response_diagnostics({"diagnostics": ["ignored"]}, 1, "h") == []

    from optimizer.core.trial_runner import _runner_process_entry

    err_queue = __import__("multiprocessing").Queue(maxsize=1)
    _runner_process_entry(
        err_queue, lambda _payload: (_ for _ in ()).throw(ValueError("boom")), {}
    )
    status, exc_name, message, _tb = err_queue.get(timeout=1)
    err_queue.close()
    err_queue.cancel_join_thread()
    assert (status, exc_name, message) == ("err", "ValueError", "boom")

    cfg = OptimizerConfig(
        objective_expression="x",
        timeout_per_trial_sec=0,
        report_profiles=False,
        use_profile_auto_constraints=False,
    )
    trial = run_one(1, {"x": 1}, lambda params: {"x": 3}, cfg, "s", "c")
    assert trial.status == "completed"
    assert not any(d.code == "BASIC_RUNNER_CONTRACT_USED" for d in trial.diagnostics)

    from optimizer.core.contracts import RUNNER_CONTRACT

    response = _normalize_runner_response(
        SimpleNamespace(contract=RUNNER_CONTRACT, metrics=None, hashes=None), 1, "h"
    )
    assert response.metrics_source == {}
    assert response.hashes == {}


def test_walk_forward_extra_params_and_no_recommendation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[RunnerRequest] = []

    class RequestRunner:
        capabilities = RunnerCapabilities(
            supports_runner_request=True, supports_range=True
        )

        def __call__(self, request: RunnerRequest):  # type: ignore[no-untyped-def]
            seen.append(request)
            return _runner(request.params)

    wrapped = walk_forward._RunnerRequestRangeWrapper(  # type: ignore[attr-defined]
        RequestRunner(), (10, 20), "test", {"z": 9}
    )
    assert wrapped({"x": 1})["net_profit"] == 1.0
    assert seen[-1].params["z"] == 9

    monkeypatch.setattr(
        "optimizer.optimizer.optimize",
        lambda *_args, **_kwargs: SimpleNamespace(recommended_trial=None),
    )
    cfg = OptimizerConfig(
        algorithm="walk_forward",
        output_dir=tmp_path,
        storage_backend="json",
        max_trials=1,
        walk_forward_windows=1,
        walk_forward_train_ratio=0.5,
    )
    result = walk_forward.run(_space(), RequestRunner(), cfg, start=0, end=10)
    assert result["windows"][0]["test_trial"] is None


def test_genetic_initial_population_duplicate_branch() -> None:
    class DuplicateSpace(ParameterSpace):
        def random_sample(self, _rng):  # type: ignore[no-untyped-def]
            return {"x": 1}

    cfg = OptimizerConfig(max_trials=2, genetic_population_size=2, seed=1)
    assert genetic.initial_population(
        DuplicateSpace([Parameter("x", "int", 1, 1, 2, 1)]), cfg
    ) == [{"x": 1}]


def test_leaderboard_secondary_auto_direction() -> None:
    from optimizer.results.leaderboard import rank_trials

    cfg = SimpleNamespace(
        objective_tie_epsilon=10.0,
        objective_secondary="max_drawdown_percent",
        objective_secondary_direction="auto",
    )
    left = _trial(1, {}, 10.0)
    right = _trial(2, {}, 11.0)
    left.metrics["max_drawdown_percent"] = 5.0
    right.metrics["max_drawdown_percent"] = 1.0
    assert [t.id for t in rank_trials([left, right], cfg)] == [right.id, left.id]


def test_genetic_and_bayesian_advanced_engine_edge_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _Store(tmp_path)
    completed = _trial(1, {"x": 1}, 1.0)
    failed = _trial(2, {"x": 2}, None, status="failed")

    # Genetic duplicate candidate branch and natural loop exhaustion.
    monkeypatch.setattr(
        genetic, "initial_population", lambda _space, _cfg: [{"x": 1}, {"x": 1}]
    )
    monkeypatch.setattr(genetic, "next_generation", lambda *_args: [{"x": 2}])
    monkeypatch.setattr(
        "optimizer.engine._run_jobs",
        lambda jobs, *_args: (
            [replace(completed, id=jobs[0][0], params=jobs[0][1])] if jobs else []
        ),
    )
    cfg = OptimizerConfig(
        algorithm="genetic",
        max_trials=3,
        genetic_generations=1,
        output_dir=tmp_path,
        storage_backend="json",
        resume=False,
    )
    trials, _next = _sequential_advanced(_space(), _runner, cfg, "s", "c", store, 1)
    assert [t.params for t in trials] == [{"x": 1}]

    # Genetic max-trials break before enqueueing another candidate.
    monkeypatch.setattr(
        genetic, "initial_population", lambda _space, _cfg: [{"x": 1}, {"x": 2}]
    )
    cfg_one = replace(cfg, max_trials=1)
    trials_one, _ = _sequential_advanced(_space(), _runner, cfg_one, "s", "c", store, 1)
    assert len(trials_one) == 1

    # Genetic fail-fast stop branch.
    monkeypatch.setattr("optimizer.engine._run_jobs", lambda jobs, *_args: [failed])
    cfg_fail = replace(cfg, fail_fast=True)
    trials_fail, _ = _sequential_advanced(
        _space(), _runner, cfg_fail, "s", "c", store, 1
    )
    assert trials_fail[0].status == "failed"

    # Genetic no next generation branch.
    monkeypatch.setattr("optimizer.engine._run_jobs", lambda jobs, *_args: [completed])
    monkeypatch.setattr(genetic, "next_generation", lambda *_args: [])
    trials_none, _ = _sequential_advanced(_space(), _runner, cfg, "s", "c", store, 1)
    assert trials_none

    # Bayesian duplicate warmup, max-trials warmup break, fail-fast warmup and propose stop branches.
    from optimizer.algorithms import bayesian

    monkeypatch.setattr(
        bayesian, "warmup", lambda _space, _cfg: [{"x": 1}, {"x": 1}, {"x": 2}]
    )
    monkeypatch.setattr(bayesian, "propose", lambda *_args: None)
    cfg_b = OptimizerConfig(
        algorithm="bayesian",
        max_trials=3,
        bayesian_trials=3,
        output_dir=tmp_path / "b",
        storage_backend="json",
        resume=False,
    )
    trials_b, _ = _sequential_advanced(
        _space(), _runner, cfg_b, "s", "c", _Store(tmp_path / "b"), 1
    )
    assert len(trials_b) == 2

    cfg_b_one = replace(cfg_b, max_trials=1)
    trials_b_one, _ = _sequential_advanced(
        _space(), _runner, cfg_b_one, "s", "c", _Store(tmp_path / "b1"), 1
    )
    assert len(trials_b_one) == 1

    monkeypatch.setattr("optimizer.engine._run_jobs", lambda jobs, *_args: [failed])
    cfg_b_fail = replace(cfg_b, fail_fast=True)
    trials_b_fail, _ = _sequential_advanced(
        _space(), _runner, cfg_b_fail, "s", "c", _Store(tmp_path / "bf"), 1
    )
    assert trials_b_fail[0].status == "failed"

    monkeypatch.setattr(
        "optimizer.engine._run_jobs",
        lambda jobs, *_args: [completed] if jobs[0][1]["x"] == 1 else [failed],
    )
    monkeypatch.setattr(bayesian, "warmup", lambda _space, _cfg: [{"x": 1}])
    monkeypatch.setattr(bayesian, "propose", lambda *_args: {"x": 3})
    trials_b_prop, _ = _sequential_advanced(
        _space(), _runner, cfg_b_fail, "s", "c", _Store(tmp_path / "bp"), 1
    )
    assert [t.status for t in trials_b_prop] == ["completed", "failed"]

    cfg_unknown = replace(cfg_b, algorithm="unsupported")  # type: ignore[arg-type]
    with pytest.raises(Exception, match="Unknown optimizer algorithm"):
        _sequential_advanced(_space(), _runner, cfg_unknown, "s", "c", store, 1)


def test_adaptive_grid_empty_refine_and_distribution_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from optimizer import distribution
    from optimizer.algorithms import adaptive_grid

    monkeypatch.setattr(adaptive_grid, "refine", lambda *_args: [])
    cfg = OptimizerConfig(
        algorithm="adaptive_grid",
        max_trials=1,
        output_dir=tmp_path / "adaptive",
        storage_backend="json",
        resume=False,
        timeout_per_trial_sec=0,
    )
    result = optimize(_space(), _runner, cfg)
    assert result.status == "completed"

    bad = tmp_path / "build" / "bad.py"
    bad.parent.mkdir()
    bad.write_text("x = 1")
    monkeypatch.setattr(distribution, "_version", lambda _root: "4.0.0")
    monkeypatch.setattr(
        distribution, "iter_files", lambda root: [Path(root) / "build" / "bad.py"]
    )
    report = distribution.manifest(tmp_path)
    assert report.hygiene_ok is False
    assert report.forbidden_files == ["build", "build/bad.py"]
