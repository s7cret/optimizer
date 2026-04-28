def initial(space, config): yield from space.generate_grid(min(config.grid_max_combinations, config.max_trials))
def refine(space, trials, config):
    pts=[]
    top=sorted([t for t in trials if t.status=='completed'], key=lambda t:t.objective_value or float('-inf'), reverse=True)[:config.adaptive_grid_top_n]
    seen=set()
    import json
    for t in top:
        for p in space.refine_around(t.params, config.adaptive_grid_refinement_factor, config.adaptive_grid_min_step):
            h=json.dumps(p,sort_keys=True,default=str)
            if h not in seen:
                seen.add(h); pts.append(p)
                if len(pts)>=config.adaptive_grid_max_new_points_per_round: return pts
    return pts
