"""Identity and admission edges must not create false successful trials."""

import pytest

from optimizer import OptimizerConfig
from optimizer.core.trial_runner import run_one
from optimizer.protocols import RunnerRequest
from optimizer.results.leaderboard import rank_trials
from optimizer.runners.backtest_engine import (
    BacktestEngineRunnerAdapter,
    _identity_value,
)


def test_binary_and_unordered_state_has_stable_lossless_identity():
    assert _identity_value(b"\x00\xff") == {"bytes": "00ff"}
    assert _identity_value({2, 1}) == {"set": [1, 2]}
    assert _identity_value(frozenset({1, 2})) == {"set": [1, 2]}


def test_function_identity_tracks_captured_state_and_defaults():
    def factory(captured):
        def runner(value=3, *, scale=2):
            return captured, value, scale

        return runner

    first = _identity_value(factory(b"\x00\xff"))
    assert first["defaults"] == [3]
    assert first["kwdefaults"] == {"scale": 2}
    assert first["closure"] == [{"bytes": "00ff"}]
    assert first == _identity_value(factory(b"\x00\xff"))
    assert first != _identity_value(factory(b"\x00\xfe"))
    assert _identity_value(lambda: None)["closure"] == []


def test_strict_identity_flag_does_not_coerce_truthy_values():
    with pytest.raises(TypeError, match="must be a bool"):
        BacktestEngineRunnerAdapter(
            engine_factory=lambda: None,
            strategy=object,
            bars=[],
            strict_identity="false",
        )


def test_missing_adapter_metric_is_not_invented_as_zero():
    class Engine:
        def run(self, strategy, *, bars, params):
            return {"status": "completed", "net_profit": None}

    adapter = BacktestEngineRunnerAdapter(
        engine_factory=Engine,
        strategy=object,
        bars=[],
        runner_fingerprint="1" * 64,
        data_fingerprint="2" * 64,
        engine_config_hash="3" * 64,
    )
    result = adapter(RunnerRequest({}, 1, {"net_profit"}, set(), []))
    assert result.metrics == {}


def test_overflow_after_finite_penalty_inputs_cannot_rank(tmp_path):
    config = OptimizerConfig(
        output_dir=tmp_path,
        timeout_per_trial_sec=0,
        report_profiles=False,
        use_profile_auto_constraints=False,
        constraint_mode="penalty",
        constraint_penalty_multiplier=1e308,
        constraints={"net_profit": {"min": 1e308, "hard": False}},
    )
    trial = run_one(1, {}, lambda _: {"net_profit": 1}, config, "s", "c")
    assert trial.status == "failed"
    assert trial.objective_value is None
    assert rank_trials([trial], config) == []
    assert any(
        "objective after penalties is not finite" in d.message
        for d in trial.diagnostics
    )
