from __future__ import annotations

from dataclasses import dataclass, field

from optimizer import BacktestEngineRunnerAdapter, OptimizerConfig, Parameter, optimize
from optimizer.protocols import RunnerCapabilities, RunnerRequest, RunnerResponse


@dataclass
class FakeResult:
    net_profit: object = 10.0
    max_drawdown_percent: float = 2.0
    status: str = "completed"
    closed_trades: list[dict] | None = field(default_factory=lambda: [{"id": "t"}])
    equity_curve: list[dict] | None = field(default_factory=lambda: [{"equity": 10.0}])
    warnings: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    config_snapshot: dict = field(
        default_factory=lambda: {"symbol": "S", "timeframe": "1D"}
    )
    data_fingerprint: str = "data"
    strategy_fingerprint: str = "strategy"
    runtime_fingerprint: str = "runtime"

    def content_hash(self):
        return "content"


class FakeStrategy:
    pass


def cfg(tmp_path, **kw):
    d = dict(
        output_dir=tmp_path,
        storage_backend="json",
        objective="net_profit",
        report_profiles=False,
        use_profile_auto_constraints=False,
        timeout_per_trial_sec=0,
    )
    d.update(kw)
    return OptimizerConfig(**d)


def adapter_for(result):
    class Engine:
        def run(self, strategy, *, bars, params):
            return result

    return BacktestEngineRunnerAdapter(
        engine_factory=Engine, strategy=FakeStrategy, bars=[{"t": 1}]
    )


def one_param():
    return [Parameter("x", "int", 1, 1, 1, 1)]


def test_engine_failed_status_becomes_failed_trial_with_diagnostics(tmp_path):
    result = FakeResult(status="failed")
    opt = optimize(one_param(), adapter_for(result), cfg(tmp_path))
    trial = opt.all_trials[0]

    assert trial.status == "failed"
    assert (
        trial.error_message
        and "BACKTEST_ENGINE_RUN_NOT_COMPLETED" in trial.error_message
    )
    assert any(d.code == "BACKTEST_ENGINE_RUN_NOT_COMPLETED" for d in trial.diagnostics)


def test_missing_required_output_fails_post_run(tmp_path):
    result = FakeResult(equity_curve=None)
    opt = optimize(
        one_param(), adapter_for(result), cfg(tmp_path, objective="sharpe_ratio")
    )
    trial = opt.all_trials[0]

    assert trial.status == "failed"
    assert any(d.code == "RUNNER_REQUIRED_OUTPUT_MISSING" for d in trial.diagnostics)


def test_bad_metric_type_or_nan_fails_as_missing_required_metric(tmp_path):
    for bad in ("not-a-number", float("nan")):
        opt = optimize(
            one_param(), adapter_for(FakeResult(net_profit=bad)), cfg(tmp_path)
        )
        trial = opt.all_trials[0]
        assert trial.status == "failed"
        assert any(
            d.code == "BACKTEST_ENGINE_BAD_METRIC_VALUE" for d in trial.diagnostics
        )


def test_runner_request_carries_seed_only_when_capability_is_declared(tmp_path):
    seen: list[RunnerRequest] = []

    class SeedRunner:
        capabilities = RunnerCapabilities(
            supports_runner_request=True,
            supports_required_outputs=True,
            supported_outputs={"summary_metrics"},
            supports_seed=True,
        )

        def __call__(self, request: RunnerRequest) -> RunnerResponse:
            seen.append(request)
            return RunnerResponse(
                metrics={"net_profit": 1.0},
                hashes={"content_hash": "c"},
                trades_available=True,
                equity_available=False,
            )

    opt = optimize(one_param(), SeedRunner(), cfg(tmp_path, seed=123))

    assert opt.all_trials[0].status == "completed"
    assert seen[0].seed == 123
    assert seen[0].params == {"x": 1}
    assert seen[0].fingerprints["parameter_space_hash"]


def test_backtest_adapter_hashes_are_stable_for_same_result(tmp_path):
    result = FakeResult(config_snapshot={"b": 2, "a": 1})
    opt1 = optimize(one_param(), adapter_for(result), cfg(tmp_path / "a"))
    opt2 = optimize(one_param(), adapter_for(result), cfg(tmp_path / "b"))

    t1 = opt1.all_trials[0]
    t2 = opt2.all_trials[0]
    assert t1.result_content_hash == t2.result_content_hash == "content"
    assert t1.engine_config_hash == t2.engine_config_hash
    assert t1.data_fingerprint == t2.data_fingerprint == "data"
