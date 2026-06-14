from pathlib import Path

from optimizer.config import OptimizerConfig
from optimizer.core.diagnostic import Diagnostic
from optimizer.core.parameter_space import ParameterSpace
from optimizer.results.result import DryRunValidationResult


def dry_run_validate(
    parameters,
    config: OptimizerConfig | None = None,
    *,
    cross_constraints=None,
    sample_limit: int = 20,
) -> DryRunValidationResult:
    config = config or OptimizerConfig()
    config.output_dir = Path(config.output_dir)
    space = (
        parameters
        if isinstance(parameters, ParameterSpace)
        else ParameterSpace(
            parameters,
            (
                cross_constraints
                if cross_constraints is not None
                else config.cross_constraints
            ),
        )
    )

    grid_combinations = space.grid_size(respect_constraints=False)
    valid_combinations = 0
    invalid_samples: list[dict[str, object]] = []
    for params, valid in space.iter_grid_with_validity():
        if valid:
            valid_combinations += 1
        elif len(invalid_samples) < sample_limit:
            invalid_samples.append(dict(params))
    invalid_combinations = grid_combinations - valid_combinations
    diagnostics = []
    if invalid_combinations:
        diagnostics.append(
            Diagnostic(
                "INVALID_PARAM_COMBINATIONS",
                "Some generated parameter combinations fail cross constraints",
                "warning",
                context={
                    "invalid_combinations": invalid_combinations,
                    "sample_limit": sample_limit,
                },
            )
        )
    if valid_combinations == 0:
        diagnostics.append(
            Diagnostic(
                "NO_VALID_PARAM_COMBINATIONS",
                "No parameter combinations passed validation",
                "error",
            )
        )

    return DryRunValidationResult(
        "valid" if valid_combinations and not invalid_combinations else "invalid",
        len(space.parameters),
        grid_combinations,
        valid_combinations,
        invalid_combinations,
        invalid_samples,
        diagnostics,
    )
