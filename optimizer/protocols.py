from dataclasses import dataclass, field
from typing import Any, Protocol

class BacktestRunner(Protocol):
    def __call__(self, params: dict[str, Any]) -> Any: ...

@dataclass(frozen=True)
class RunnerRequest:
    params: dict[str, Any]
    trial_id: int
    required_metrics: set[str]
    required_outputs: set[str]
    early_stop_conditions: list[dict]
    range: tuple[int, int] | None = None
    seed: int | None = None
    tags: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class RunnerCapabilities:
    supports_runner_request: bool = False
    supports_range: bool = False
    supports_early_stop: bool = False
    supports_required_outputs: bool = False
    supported_outputs: set[str] = field(default_factory=set)
    supports_seed: bool = False
    supports_content_hash: bool = False
    supports_data_fingerprint: bool = False
    supports_engine_config_hash: bool = False

class AdvancedBacktestRunner(Protocol):
    capabilities: RunnerCapabilities
    def __call__(self, request: RunnerRequest) -> Any: ...

class RangeAwareBacktestRunner(Protocol):
    def __call__(self, params: dict[str, object]) -> object: ...
    def with_range(self, start_time: int, end_time: int) -> 'RangeAwareBacktestRunner': ...
