"""Failed, partial and non-finite results must never become winning trials."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from optimizer import OptimizerConfig, RunnerResponse
from optimizer.core.contracts import LEGACY_RUNNER_CONTRACTS, RUNNER_CONTRACT
from optimizer.core.metric_extractor import MetricExtractor
from optimizer.core.normalization import balanced_score
from optimizer.core.objective import compute_objective
from optimizer.core.trial_runner import _normalize_runner_response, run_one
from optimizer.results.leaderboard import rank_trials


@pytest.mark.parametrize(
    "status", ["failed", "cancelled", "canceled", "running", "partial", "timeout", "unknown"]
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_noncompleted_results_fail_even_with_profitable_metrics(status, wrapped, tmp_path):
    payload = {"status": status, "net_profit": 1_000_000}
    response = (
        RunnerResponse(metrics={"net_profit": 1_000_000}, raw_result=payload)
        if wrapped
        else payload
    )
    config = OptimizerConfig(
        output_dir=tmp_path,
        timeout_per_trial_sec=0,
        report_profiles=False,
        use_profile_auto_constraints=False,
    )
    trial = run_one(1, {}, lambda _: response, config, "s", "c")
    assert trial.status == "failed" and trial.objective_value is None
    assert rank_trials([trial], config) == []


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -float("inf"), "NaN", "Infinity", True, 10**1000]
)
def test_invalid_primary_metric_fails_trial(bad, tmp_path):
    config = OptimizerConfig(
        output_dir=tmp_path,
        timeout_per_trial_sec=0,
        report_profiles=False,
        use_profile_auto_constraints=False,
    )
    trial = run_one(1, {}, lambda _: {"net_profit": bad}, config, "s", "c")
    assert trial.status == "failed" and trial.objective_value is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_direct_objective_and_custom_metrics_reject_invalid_numbers(bad):
    with pytest.raises(ValueError):
        compute_objective({"net_profit": bad})
    with pytest.raises(ValueError):
        MetricExtractor({"net_profit": lambda _: bad}).extract({})


def test_extractor_retains_zero_and_omits_nonfinite_optional_values():
    assert MetricExtractor().extract(
        {"net_profit": 0, "ratio": float("inf"), "status": "completed"}
    ) == {"net_profit": 0.0}


def test_zero_profit_factor_profit_and_drawdown_are_not_defaults():
    assert (
        balanced_score(
            {
                "net_profit": 0,
                "net_profit_percent": 100,
                "profit_factor": 0,
                "max_drawdown_percent": 0,
                "max_drawdown": 500,
            }
        )
        == 0
    )
    assert balanced_score({"net_profit_percent": 3, "max_drawdown": 2}) == 11


@pytest.mark.parametrize("contract", [RUNNER_CONTRACT, *LEGACY_RUNNER_CONTRACTS])
def test_accepted_contract_aliases_cannot_bypass_output_validation(contract):
    response = _normalize_runner_response(
        RunnerResponse(contract=contract, metrics={"net_profit": 1}), 1, "p"
    )
    assert response.is_contract_response


@pytest.mark.parametrize(
    "change",
    [
        {"trades_available": "false"},
        {"equity_available": 1},
        {"hashes": {"content_hash": {"x": 1}}},
        {"hashes": {"x": 1}},
    ],
)
def test_response_metadata_is_not_coerced_to_valid_types(change):
    with pytest.raises(ValueError):
        _normalize_runner_response(replace(RunnerResponse(), **change), 1, "p")


def test_error_bearing_legacy_result_rejected_even_when_status_says_completed():
    response = _normalize_runner_response(
        {"status": "completed", "net_profit": 1, "errors": ["bad fill"]}, 1, "p"
    )
    assert any(
        d.code == "RUNNER_RESULT_ERRORS" and d.severity == "error" for d in response.diagnostics
    )


def test_restored_nonfinite_scores_and_stale_ranks_are_excluded():
    trials = [
        SimpleNamespace(
            id=i, status="completed", objective_value=v, objective_direction="maximize", rank=1
        )
        for i, v in enumerate([float("nan"), float("inf"), 0.0, -1.0])
    ]
    assert [t.id for t in rank_trials(trials)] == [2, 3]
    assert trials[0].rank is None and trials[1].rank is None
