import random
import json


def generate_with_validity(space, config):
    rng = random.Random(config.seed)
    seen = set()
    n = min(config.random_trials, config.max_trials)
    attempts = 0
    while len(seen) < n and attempts < n * 20:
        attempts += 1
        p = space.random_sample(rng)
        h = json.dumps(p, sort_keys=True, default=str)
        if h in seen:
            continue
        seen.add(h)
        yield p, space.is_valid_combination(p)


def generate(space, config):
    for params, valid in generate_with_validity(space, config):
        if valid:
            yield params
