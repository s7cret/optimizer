import random
from statistics import mean
from typing import Any
from optimizer.analysis.profit_concentration import _profit, _trades_from


def analyze_trial(
    raw: Any, simulations: int = 1000, seed: int = 42
) -> dict[str, object]:
    profits = [_profit(t) for t in _trades_from(raw)]
    if not profits:
        return {
            "status": "insufficient_data",
            "simulations": 0,
            "p05_net_profit": None,
            "median_net_profit": None,
            "probability_positive": None,
        }
    rng = random.Random(seed)
    totals = []
    for _ in range(max(1, simulations)):
        sample = [rng.choice(profits) for _ in profits]
        totals.append(sum(sample))
    totals.sort()

    def q(p: float) -> float:
        return totals[min(len(totals) - 1, max(0, int(round(p * (len(totals) - 1)))))]

    return {
        "status": "ok",
        "simulations": len(totals),
        "p05_net_profit": q(0.05),
        "median_net_profit": q(0.5),
        "mean_net_profit": mean(totals),
        "probability_positive": sum(1 for x in totals if x > 0) / len(totals),
    }


def analyze(
    trials: list[Any], simulations: int = 1000, seed: int = 42
) -> dict[str, object]:
    per = {
        t.id: analyze_trial(t.backtest_result, simulations, seed + t.id) for t in trials
    }
    ok = any(v["status"] == "ok" for v in per.values())
    return {
        "status": "ok" if ok else "insufficient_data",
        "method": "bootstrap_trades",
        "by_trial_id": per,
        "diagnostics": (
            []
            if ok
            else [
                {
                    "code": "MONTE_CARLO_REQUIRES_TRADES",
                    "severity": "warning",
                    "message": "Expected saved backtest_result trades",
                }
            ]
        ),
    }
