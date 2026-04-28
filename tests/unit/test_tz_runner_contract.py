import pytest

from optimizer import OptimizerConfig, Parameter, RunnerCapabilities, RunnerResponse, optimize


def cfg(tmp_path):
    return OptimizerConfig(output_dir=tmp_path, storage_backend="json", use_profile_auto_constraints=False)


class ContractRunner:
    capabilities = RunnerCapabilities(supports_runner_request=True, supports_required_outputs=True, supported_outputs={"summary_metrics"})

    def __call__(self, req):
        assert req.contract == "pain.optimizer_runner.v1"
        assert req.fingerprints["parameter_space_hash"]
        return RunnerResponse(
            metrics={"net_profit": req.params["x"], "max_drawdown_percent": 1, "profit_factor": 1.1, "sharpe_ratio": 1.0},
            hashes={"content_hash": "c1", "data_fingerprint": "d1", "runner_fingerprint": "r1", "engine_config_hash": "e1"},
            diagnostics=[{"code": "RUNNER_NOTE", "message": "ok", "severity": "warning"}],
        )


def test_runner_request_response_contract_and_hashes(tmp_path):
    res = optimize([Parameter("x", "int", 2, 2, 2, 1)], ContractRunner(), cfg(tmp_path))
    trial = res.all_trials[0]
    assert trial.status == "completed"
    assert trial.metrics["net_profit"] == 2
    assert trial.result_content_hash == "c1"
    assert trial.data_fingerprint == "d1"
    assert trial.runner_fingerprint == "r1"
    assert trial.engine_config_hash == "e1"
    assert any(d.code == "RUNNER_NOTE" for d in trial.diagnostics)


class BadContractRunner:
    capabilities = RunnerCapabilities(supports_runner_request=True)

    def __call__(self, req):
        return {"contract": "wrong", "metrics": {"net_profit": 1, "max_drawdown_percent": 1}}


def test_runner_response_contract_mismatch_fails_closed(tmp_path):
    res = optimize([Parameter("x", "int", 1, 1, 1, 1)], BadContractRunner(), cfg(tmp_path))
    trial = res.all_trials[0]
    assert trial.status == "failed"
    assert "contract mismatch" in trial.error_message


class MissingRequiredOutputRunner:
    capabilities = RunnerCapabilities(
        supports_runner_request=True,
        supports_required_outputs=True,
        supported_outputs={"summary_metrics", "closed_trades"},
    )

    def __call__(self, req):
        return RunnerResponse(
            metrics={"profit_concentration_score": 1.0},
            trades_available=False,
            diagnostics=[{"code": "NO_TRADES_COLLECTED", "message": "closed trades unavailable"}],
        )


def test_runner_response_missing_required_output_fails_closed(tmp_path):
    res = optimize(
        [Parameter("x", "int", 1, 1, 1, 1)],
        MissingRequiredOutputRunner(),
        OptimizerConfig(
            output_dir=tmp_path,
            storage_backend="json",
            objective="profit_concentration_score",
            use_profile_auto_constraints=False,
        ),
    )
    trial = res.all_trials[0]
    assert trial.status == "failed"
    assert "closed_trades" in trial.error_message
    assert any(d.code == "RUNNER_REQUIRED_OUTPUT_MISSING" for d in trial.diagnostics)
    assert any(d.code == "NO_TRADES_COLLECTED" for d in trial.diagnostics)
