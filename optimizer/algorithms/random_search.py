import random
def generate(space, config):
    rng=random.Random(config.seed); seen=set(); n=min(config.random_trials, config.max_trials)
    import json
    attempts=0
    while len(seen)<n and attempts<n*20:
        attempts+=1; p=space.random_sample(rng)
        if not space.is_valid_combination(p): continue
        h=json.dumps(p,sort_keys=True,default=str)
        if h in seen: continue
        seen.add(h); yield p
