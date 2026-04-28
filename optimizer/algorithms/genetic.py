import json
import random
from typing import Any


def _key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _values(space: Any) -> dict[str, list[object]]:
    return {p.name: list(space.values_for(p)) for p in space.parameters}


def crossover(a: dict[str, object], b: dict[str, object], rng: random.Random, rate: float) -> dict[str, object]:
    if rng.random() > rate:
        return dict(a)
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def mutate(child: dict[str, object], values: dict[str, list[object]], rng: random.Random, rate: float) -> dict[str, object]:
    out = dict(child)
    for name, vals in values.items():
        if vals and rng.random() < rate:
            out[name] = rng.choice(vals)
    return out


def select(population: list[tuple[dict[str, object], float]], rng: random.Random, mode: str) -> dict[str, object]:
    if not population:
        raise ValueError('empty genetic population')
    if mode == 'roulette':
        lo = min(score for _, score in population)
        weights = [max(0.0, score - lo) + 1e-12 for _, score in population]
        return dict(rng.choices([p for p, _ in population], weights=weights, k=1)[0])
    if mode == 'rank':
        ranked = sorted(population, key=lambda x: x[1])
        weights = list(range(1, len(ranked) + 1))
        return dict(rng.choices([p for p, _ in ranked], weights=weights, k=1)[0])
    k = min(3, len(population))
    return dict(max(rng.sample(population, k), key=lambda x: x[1])[0])


def initial_population(space: Any, config: Any) -> list[dict[str, object]]:
    rng = random.Random(config.seed)
    seen: set[str] = set()
    pop: list[dict[str, object]] = []
    attempts = 0
    target = min(config.genetic_population_size, config.max_trials)
    while len(pop) < target and attempts < max(100, target * 50):
        attempts += 1
        params = space.random_sample(rng)
        if not space.is_valid_combination(params):
            continue
        k = _key(params)
        if k not in seen:
            seen.add(k)
            pop.append(params)
    return pop


def next_generation(space: Any, config: Any, evaluated: list[tuple[dict[str, object], float]], seen: set[str], generation: int) -> list[dict[str, object]]:
    rng = random.Random(config.seed + generation + 1)
    vals = _values(space)
    ranked = sorted(evaluated, key=lambda x: x[1], reverse=True)
    out: list[dict[str, object]] = []
    local_seen = set(seen)
    attempts = 0
    target = min(config.genetic_population_size, max(0, config.max_trials - len(seen)))
    while len(out) < target and attempts < max(100, target * 80):
        attempts += 1
        a = select(ranked, rng, config.genetic_selection)
        b = select(ranked, rng, config.genetic_selection)
        child = mutate(crossover(a, b, rng, config.genetic_crossover_rate), vals, rng, config.genetic_mutation_rate)
        child = space.clamp(child)
        if not space.is_valid_combination(child):
            continue
        k = _key(child)
        if k in local_seen:
            continue
        local_seen.add(k)
        out.append(child)
    return out


def generate(space: Any, config: Any):
    yield from initial_population(space, config)
