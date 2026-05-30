from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal
from optimizer.results.trial import Trial
from optimizer.results.profile_result import ResultProfile


@dataclass
class OptimizerRunResult:
    recommended_trial: Trial | None
    recommended_profile: str | None
    best_trial: Trial | None
    top_trials: list[Trial]
    all_trials: list[Trial] | None = None
    passed_trials: list[Trial] | None = None
    failed_constraint_trials: list[Trial] | None = None
    storage_ref: str | None = None
    trials_count_by_status: dict[str, int] = field(default_factory=dict)
    profiles: dict[str, ResultProfile] = field(default_factory=dict)
    best_objective: Trial | None = None
    best_passed_constraints: Trial | None = None
    best_balanced: Trial | None = None
    most_robust: Trial | None = None
    best_profit: Trial | None = None
    best_drawdown: Trial | None = None
    best_profit_factor: Trial | None = None
    best_sharpe: Trial | None = None
    pareto_front: list[Trial] | None = None
    diagnostics: list = field(default_factory=list)
    analysis: dict[str, object] = field(default_factory=dict)
    baseline_trial: Trial | None = None
    baseline_comparison: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None
    status: Literal["completed", "failed"] = "failed"
    best_params: dict[str, object] | None = None
    best_score: float | None = None
    trials: tuple[Trial, ...] = field(default_factory=tuple)
    artifact_path: Path | None = None
    data_query: Any | None = None

    def to_dict(self):
        data = asdict(self)
        if self.artifact_path is not None:
            data["artifact_path"] = str(self.artifact_path)
        return data


@dataclass
class DryRunValidationResult:
    status: Literal["valid", "invalid"]
    parameters_count: int
    grid_combinations: int
    valid_combinations: int
    invalid_combinations: int
    invalid_samples: list[dict[str, object]] = field(default_factory=list)
    diagnostics: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
