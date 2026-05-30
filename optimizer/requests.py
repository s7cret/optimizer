from dataclasses import dataclass, field
from typing import Any, Literal

from optimizer.core.parameter_space import ParameterSpace


@dataclass(frozen=True)
class StrategyRef:
    id: str
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveSpec:
    metric: str = "net_profit"
    direction: Literal["maximize", "minimize", "auto"] = "auto"
    expression: str | None = None


@dataclass(frozen=True)
class OptimizationConstraints:
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    cross_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizerRunRequest:
    run_id: str
    strategy_ref: StrategyRef | None
    parameter_space: ParameterSpace
    data_query: Any | None = None
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)
    tags: dict[str, str] = field(default_factory=dict)
