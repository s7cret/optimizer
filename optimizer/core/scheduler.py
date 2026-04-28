from optimizer.runners.parallel import map_parallel


def run_scheduled(fn, items, max_parallel=1, backend="thread", ordered=False):
    return map_parallel(fn, items, max_parallel, backend, ordered)
