from __future__ import annotations

from pathlib import Path

from backtest_engine import BacktestConfig, BacktestEngine, Bar  # noqa: E402
from optimizer import BacktestEngineRunnerAdapter, OptimizerConfig, Parameter, optimize  # noqa: E402


class ParamQtyCloseAll:
    def __init__(self, params, runtime, ctx):
        self.ctx = ctx
        self.qty = float(params["qty"])

    def _process_bar(self, bar, bar_index):
        if bar_index == 0:
            self.ctx.entry("L", "long", qty=self.qty)
        if bar_index == 1:
            self.ctx.close_all()


def test_backtest_engine_runner_optimizer_selected_trial_hash_lineage(tmp_path: Path):
    bars = [
        Bar(1, 10, 10, 10, 10),
        Bar(2, 15, 15, 15, 15),
        Bar(3, 15, 15, 15, 15),
    ]
    config = BacktestConfig(
        symbol="STAGE12",
        timeframe="1D",
        start_time=1,
        end_time=3,
        commission_type="none",
        process_orders_on_close=True,
        data_fingerprint="stage12-data",
        strategy_fingerprint="stage12-strategy",
        runtime_fingerprint="stage12-runtime",
    )
    adapter = BacktestEngineRunnerAdapter(
        engine_factory=lambda: BacktestEngine(config),
        strategy=ParamQtyCloseAll,
        bars=bars,
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
    assert selected.metrics["net_profit"] == 10.0
    assert selected.result_content_hash
    assert selected.data_fingerprint == "stage12-data"
    assert selected.runner_fingerprint == "stage12-strategy"
    assert selected.engine_config_hash
    assert selected.parameter_space_hash
    assert all(d.severity != "error" for d in selected.diagnostics)
