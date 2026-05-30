from optimizer.version import __version__
from optimizer.config import OptimizerConfig
from optimizer.protocols import (
    BacktestRunner,
    AdvancedBacktestRunner,
    RangeAwareBacktestRunner,
    RunnerCapabilities,
    RunnerRequest,
    RunnerResponse,
)
from optimizer.core.parameter import Parameter
from optimizer.core.parameter_space import ParameterSpace
from optimizer.core.metric_extractor import MetricExtractor
from optimizer.core.metric_registry import MetricRegistry, MetricSpec
from optimizer.core.diagnostic import Diagnostic
from optimizer.results.trial import Trial
from optimizer.results.profile_result import ResultProfile
from optimizer.results.result import DryRunValidationResult, OptimizerRunResult
from optimizer.validation import dry_run_validate
from optimizer.optimizer import optimize
from optimizer.runners.backtest_engine import BacktestEngineRunnerAdapter

__all__ = [
    "__version__",
    "OptimizerConfig",
    "BacktestRunner",
    "AdvancedBacktestRunner",
    "RangeAwareBacktestRunner",
    "RunnerCapabilities",
    "RunnerRequest",
    "RunnerResponse",
    "Parameter",
    "ParameterSpace",
    "MetricExtractor",
    "MetricRegistry",
    "MetricSpec",
    "Diagnostic",
    "Trial",
    "ResultProfile",
    "OptimizerRunResult",
    "DryRunValidationResult",
    "optimize",
    "dry_run_validate",
    "BacktestEngineRunnerAdapter",
]
