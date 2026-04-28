from dataclasses import dataclass, field

from optimizer import BacktestEngineRunnerAdapter, OptimizerConfig, Parameter, optimize


@dataclass
class FakeBacktestResult:
    net_profit: float = 12.0
    max_drawdown_percent: float = 3.0
    profit_factor: float = 1.5
    sharpe_ratio: float = 0.75
    status: str = "completed"
    closed_trades: list[dict] = field(default_factory=lambda: [{"id": "t1"}])
    equity_curve: list[dict] = field(default_factory=lambda: [{"equity": 10012.0}])
    warnings: list[dict] = field(default_factory=lambda: [{"code": "ENGINE_NOTE", "message": "ok"}])
    errors: list[dict] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=lambda: {"symbol": "S", "timeframe": "1D"})
    data_fingerprint: str = "data-1"
    strategy_fingerprint: str = "strategy-1"
    runtime_fingerprint: str = "runtime-1"

    def content_hash(self):
        return "content-1"


class FakeEngine:
    def __init__(self):
        self.calls = []

    def run(self, strategy, *, bars, params):
        self.calls.append((strategy, list(bars), dict(params)))
        return FakeBacktestResult(net_profit=float(params["x"]))


class FakeStrategy:
    pass


def cfg(tmp_path):
    return OptimizerConfig(output_dir=tmp_path, storage_backend="json", use_profile_auto_constraints=False)


def test_backtest_engine_runner_adapter_propagates_metrics_hashes_and_lineage(tmp_path):
    engine = FakeEngine()
    adapter = BacktestEngineRunnerAdapter(
        engine_factory=lambda: engine,
        strategy=FakeStrategy,
        bars=[{"time": 1, "close": 10}],
        static_params={"static": "yes"},
    )

    res = optimize([Parameter("x", "int", 7, 7, 7, 1)], adapter, cfg(tmp_path))
    trial = res.all_trials[0]

    assert trial.status == "completed"
    assert trial.metrics["net_profit"] == 7.0
    assert trial.result_content_hash == "content-1"
    assert trial.data_fingerprint == "data-1"
    assert trial.runner_fingerprint == "strategy-1"
    assert trial.engine_config_hash
    assert any(d.code == "ENGINE_NOTE" for d in trial.diagnostics)
    assert engine.calls[0][0] is FakeStrategy
    assert engine.calls[0][2] == {"static": "yes", "x": 7}


def test_backtest_engine_runner_adapter_failed_status_is_diagnostic(tmp_path):
    class FailedEngine(FakeEngine):
        def run(self, strategy, *, bars, params):
            result = FakeBacktestResult(net_profit=1.0)
            result.status = "failed"
            return result

    adapter = BacktestEngineRunnerAdapter(
        engine_factory=FailedEngine,
        strategy=FakeStrategy,
        bars=[],
    )

    res = optimize([Parameter("x", "int", 1, 1, 1, 1)], adapter, cfg(tmp_path))
    trial = res.all_trials[0]

    assert trial.status == "completed"
    assert any(d.code == "BACKTEST_ENGINE_RUN_NOT_COMPLETED" for d in trial.diagnostics)
