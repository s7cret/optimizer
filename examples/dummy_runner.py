from __future__ import annotations

from pathlib import Path

from optimizer import OptimizerConfig, Parameter, optimize


def runner(params: dict[str, float]) -> dict[str, float]:
    x = params["x"]
    y = params["y"]
    return {
        "net_profit": 100 - (x - 3) ** 2 * 10 - (y - 2) ** 2 * 5,
        "max_drawdown_percent": 5 + abs(x - y),
        "profit_factor": 1 + x / 10,
        "sharpe_ratio": y / 2,
    }


if __name__ == "__main__":
    params = [Parameter("x", "int", 1, 1, 5, 1), Parameter("y", "int", 1, 1, 4, 1)]
    result = optimize(params, runner, OptimizerConfig(max_trials=20, output_dir=Path("example_results")))
    print(result.recommended_profile, result.recommended_trial.params)
