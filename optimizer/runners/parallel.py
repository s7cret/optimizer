from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def map_parallel(fn, items, max_workers=1, backend='thread', ordered=False):
    if max_workers<=1:
        return [fn(x) for x in items]
    Ex=ProcessPoolExecutor if backend=='process' else ThreadPoolExecutor
    with Ex(max_workers=max_workers) as ex:
        futs=[ex.submit(fn,x) for x in items]
        if ordered: return [f.result() for f in futs]
        return [f.result() for f in as_completed(futs)]
