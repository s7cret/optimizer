from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from optimizer import BacktestEngineRunnerAdapter, OptimizerConfig, Parameter, optimize


@dataclass
class ContractBacktestResult:
    net_profit: float
    max_drawdown_percent: float = 1.0
    profit_factor: float = 2.0
    sharpe_ratio: float = 1.0
    status: str = "completed"
    closed_trades: list[dict] = field(default_factory=lambda: [{"profit": 1.0}])
    equity_curve: list[dict] = field(default_factory=lambda: [{"equity": 101.0}])
    warnings: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    config_snapshot: dict = field(
        default_factory=lambda: {"symbol": "SELF", "timeframe": "1D"}
    )
    data_fingerprint: str = "self-data"
    strategy_fingerprint: str = "self-strategy"
    runtime_fingerprint: str = "self-runtime"

    def content_hash(self) -> str:
        return f"content-{self.net_profit}"


class ContractEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, strategy, *, bars, params):  # type: ignore[no-untyped-def]
        self.calls.append(
            {"strategy": strategy, "bars": list(bars), "params": dict(params)}
        )
        return ContractBacktestResult(net_profit=10.0 * float(params["qty"]))


class ContractStrategy:
    pass


def test_backtest_engine_runner_optimizer_selected_trial_hash_lineage(
    tmp_path: Path,
) -> None:
    engine = ContractEngine()
    adapter = BacktestEngineRunnerAdapter(
        engine_factory=lambda: engine,
        strategy=ContractStrategy,
        bars=[{"time": 1, "close": 10}, {"time": 2, "close": 15}],
    )

    result = optimize(
        [Parameter("qty", "int", 1, 1, 2, 1)],
        adapter,
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            objective="net_profit",
            report_profiles=False,
            use_profile_auto_constraints=False,
        ),
    )

    selected = result.recommended_trial
    assert selected is not None
    assert selected.params == {"qty": 2}
    assert selected.metrics["net_profit"] == 20.0
    assert selected.result_content_hash == "content-20.0"
    assert selected.data_fingerprint == "self-data"
    assert selected.runner_fingerprint == "self-strategy"
    assert selected.runtime_fingerprint == "self-runtime"
    assert selected.engine_config_hash
    assert selected.parameter_space_hash
    assert engine.calls[-1]["strategy"] is ContractStrategy
    assert all(d.severity != "error" for d in selected.diagnostics)
