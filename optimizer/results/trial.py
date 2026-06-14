from dataclasses import dataclass, field, asdict
from typing import Literal
from optimizer.core.diagnostic import Diagnostic


@dataclass
class Trial:
    id: int
    params: dict[str, object]
    metrics: dict[str, float | None]
    objective_value: float | None
    objective_direction: Literal["maximize", "minimize"]
    rank: int | None
    passed_constraints: bool
    constraint_violations: dict[str, str]
    constraint_violation_count: int
    balanced_score: float | None
    robustness_score: float | None
    overfitting_score: float | None
    profit_concentration_score: float | None
    backtest_result: dict | None
    execution_time_sec: float
    status: Literal["completed", "failed"]
    early_stopped: bool = False
    is_baseline: bool = False
    baseline_name: str | None = None
    result_content_hash: str | None = None
    data_fingerprint: str | None = None
    runner_fingerprint: str | None = None
    engine_config_hash: str | None = None
    parameter_space_hash: str | None = None
    optimizer_config_hash: str | None = None
    code_version: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    missing_metrics: set[str] = field(default_factory=set)
    started_at: int | None = None
    finished_at: int | None = None
    params_hash: str | None = None
    raw_objective_value: float | None = None
    runtime_fingerprint: str | None = None

    def to_dict(self):
        d = asdict(self)
        d["missing_metrics"] = sorted(self.missing_metrics)
        return d
