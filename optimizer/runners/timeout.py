import concurrent.futures
def call_with_timeout(fn, timeout):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=timeout)
