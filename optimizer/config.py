from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from optimizer.errors import ParameterValidationError


@dataclass
class OptimizerConfig:
    algorithm: Literal["grid", "adaptive_grid", "random", "genetic", "bayesian", "walk_forward"] = (
        "grid"
    )
    seed: int = 42
    objective: str = "net_profit"
    objective_direction: Literal["maximize", "minimize", "auto"] = "auto"
    objective_secondary: str | None = None
    objective_secondary_direction: Literal["maximize", "minimize", "auto"] = "auto"
    objective_tie_epsilon: float = 1e-12
    objective_expression: str | None = None
    selection_mode: Literal[
        "best_objective",
        "best_after_constraints",
        "balanced",
        "robust",
        "conservative",
        "aggressive",
        "pareto",
        "pareto_knee",
        "custom",
    ] = "best_after_constraints"
    report_champions: bool = True
    report_pareto_front: bool = False
    report_profiles: bool = True
    top_n: int = 20
    baseline_params: dict[str, object] | None = None
    baseline_name: str = "baseline"
    run_baseline_first: bool = True
    include_baseline_in_optimization_candidates: bool = False
    pareto_knee_enabled: bool = False
    pareto_knee_metrics: tuple[str, str] = ("max_drawdown_percent", "net_profit")
    pareto_knee_method: Literal["distance_to_ideal", "max_curvature", "normalized_elbow"] = (
        "distance_to_ideal"
    )
    constraints: dict[str, dict[str, float]] = field(default_factory=dict)
    cross_constraints: list[str] = field(default_factory=list)
    use_profile_auto_constraints: bool = True
    constraints_merge_mode: Literal[
        "custom_overrides_auto", "merge_auto_and_custom", "custom_only"
    ] = "custom_overrides_auto"
    constraint_mode: Literal["penalty", "filter", "both"] = "both"
    constraint_penalty_multiplier: float = 1.0
    max_trials: int = 1000
    min_completed_trials: int = 1
    timeout_per_trial_sec: float = 300.0
    fail_fast: bool = False
    max_failed_trials: int | None = None
    grid_max_combinations: int = 10000
    grid_overflow_policy: Literal["error", "truncate", "switch_to_random"] = "error"
    adaptive_grid_rounds: int = 2
    adaptive_grid_top_n: int = 10
    adaptive_grid_refinement_factor: float = 0.5
    adaptive_grid_min_step: dict[str, float] = field(default_factory=dict)
    adaptive_grid_max_new_points_per_round: int = 5000
    random_trials: int = 500
    genetic_population_size: int = 50
    genetic_generations: int = 20
    genetic_mutation_rate: float = 0.1
    genetic_crossover_rate: float = 0.8
    genetic_elitism: int = 5
    genetic_selection: Literal["tournament", "roulette", "rank"] = "tournament"
    bayesian_trials: int = 200
    bayesian_warmup_random_trials: int = 25
    walk_forward_windows: int = 4
    walk_forward_train_ratio: float = 0.7
    walk_forward_anchor_mode: Literal["rolling", "expanding"] = "rolling"
    # D5-F: walk-forward prehistory options
    walk_forward_include_prehistory: bool = False  # opt-in pre-bars before each test window
    walk_forward_pre_bars: int | None = None  # number of pre-bars to fetch before test window
    early_stop_enabled: bool = False
    early_stop_conditions: list[dict] = field(default_factory=list)
    include_early_stopped_in_recommendations: bool = False
    robustness_enabled: bool = False
    robustness_neighbor_radius_steps: int = 1
    robustness_min_neighbors: int = 8
    max_parallel: int = 1
    parallel_backend: Literal["process", "thread"] = "thread"
    timeout_backend: Literal["thread", "process", "auto"] = "thread"
    max_parallel_over_cpu_policy: Literal["warn", "error", "allow"] = "warn"
    ordered_results: bool = False
    save_all_trials: bool = True
    save_backtest_result: bool = False
    checkpoint_every: int = 50
    resume: bool = True
    force_resume_on_fingerprint_mismatch: bool = False
    run_id: str | None = None
    output_dir: Path = Path("./optimizer_results")
    storage_backend: Literal["sqlite", "json"] = "sqlite"
    analysis_profile: Literal["off", "mvp", "full", "custom"] = "mvp"
    compute_parameter_importance: bool = False
    compute_sensitivity: bool = False
    compute_robustness: bool = False
    compute_overfitting: bool = False
    compute_profit_concentration: bool = False
    compute_monte_carlo: bool = False
    monte_carlo_methods: list[str] = field(default_factory=lambda: ["reshuffle_trades"])
    monte_carlo_simulations: int = 1000
    enable_plot_exports: bool = False
    plot_format: Literal["png", "svg", "html"] = "png"

    def __post_init__(self) -> None:
        if self.max_trials < 1:
            raise ParameterValidationError("max_trials must be >= 1")
        if self.max_parallel < 1:
            raise ParameterValidationError("max_parallel must be >= 1")
        if self.walk_forward_windows < 1:
            raise ParameterValidationError("walk_forward_windows must be >= 1")
        if not 0.0 < self.walk_forward_train_ratio < 1.0:
            raise ParameterValidationError("walk_forward_train_ratio must be > 0 and < 1")
        if self.walk_forward_pre_bars is not None and self.walk_forward_pre_bars < 0:
            raise ParameterValidationError("walk_forward_pre_bars must be >= 0")

    def to_dict(self):
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d
