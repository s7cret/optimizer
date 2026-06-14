"""Public optimizer runner contract identifiers."""

RUNNER_CONTRACT = "pine.optimizer_runner.v1"
LEGACY_RUNNER_CONTRACTS = {"pain.optimizer_runner.v1"}
ACCEPTED_RUNNER_CONTRACTS = {RUNNER_CONTRACT, *LEGACY_RUNNER_CONTRACTS}

__all__ = ["RUNNER_CONTRACT", "LEGACY_RUNNER_CONTRACTS", "ACCEPTED_RUNNER_CONTRACTS"]
