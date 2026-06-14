import json
import math
import random
from typing import Any, cast


def _key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _distance(a: dict[str, object], b: dict[str, object], space: Any) -> float:
    total = 0.0
    for p in space.parameters:
        av = a.get(p.name)
        bv = b.get(p.name)
        if p.param_type in {"int", "float"} and p.max_val != p.min_val:
            total += (
                (float(cast(Any, av)) - float(cast(Any, bv)))
                / (float(cast(Any, p.max_val)) - float(cast(Any, p.min_val)))
            ) ** 2
        else:
            total += 0.0 if av == bv else 1.0
    return math.sqrt(total)


def warmup(space: Any, config: Any) -> list[dict[str, object]]:
    rng = random.Random(config.seed)
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    target = min(
        config.bayesian_warmup_random_trials, config.bayesian_trials, config.max_trials
    )
    attempts = 0
    while len(out) < target and attempts < max(100, target * 50):
        attempts += 1
        params = space.random_sample(rng)
        if not space.is_valid_combination(params):
            continue
        k = _key(params)
        if k not in seen:
            seen.add(k)
            out.append(params)
    return out


def propose(
    space: Any,
    config: Any,
    evaluated: list[tuple[dict[str, object], float]],
    seen: set[str],
    iteration: int,
) -> dict[str, object] | None:
    rng = random.Random(config.seed + 10_000 + iteration)
    if not evaluated:
        return None
    best_score = max(score for _, score in evaluated)
    worst_score = min(score for _, score in evaluated)
    span = max(1e-12, best_score - worst_score)
    best_candidate: dict[str, object] | None = None
    best_acq = -float("inf")
    for _ in range(256):
        params = space.random_sample(rng)
        if not space.is_valid_combination(params):
            continue
        k = _key(params)
        if k in seen:
            continue
        weighted = 0.0
        weight_sum = 0.0
        nearest = float("inf")
        for prev, score in evaluated:
            dist = _distance(params, prev, space)
            nearest = min(nearest, dist)
            weight = 1.0 / (dist + 1e-6)
            weighted += weight * ((score - worst_score) / span)
            weight_sum += weight
        predicted = weighted / weight_sum if weight_sum else 0.0
        exploration = min(1.0, nearest)
        acq = predicted + 0.25 * exploration
        if acq > best_acq:
            best_acq = acq
            best_candidate = params
    if best_candidate is None:
        for params in space.generate_grid(config.max_trials * 2):
            if _key(params) not in seen:
                best_candidate = params
                break
    return best_candidate


def generate(space: Any, config: Any):
    yield from warmup(space, config)
