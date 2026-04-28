def generate(space, config):
    size=space.grid_size()
    if size>config.grid_max_combinations and config.grid_overflow_policy=='error': raise ValueError(f'grid size {size} exceeds grid_max_combinations')
    maxc=config.grid_max_combinations if config.grid_overflow_policy=='truncate' else config.max_trials
    yield from space.generate_grid(maxc)
