def pareto_front(
    trials, metrics=("net_profit", "max_drawdown_percent"), directions=("maximize", "minimize")
):
    pts = [t for t in trials if t.status == "completed"]
    front = []
    for a in pts:
        dominated = False
        for b in pts:
            if a is b:
                continue
            better_or_equal = True
            strictly = False
            for m, d in zip(metrics, directions, strict=True):
                av = a.metrics.get(m)
                bv = b.metrics.get(m)
                if av is None or bv is None:
                    better_or_equal = False
                    break
                if d == "maximize":
                    better_or_equal &= bv >= av
                    strictly |= bv > av
                else:
                    better_or_equal &= bv <= av
                    strictly |= bv < av
            if better_or_equal and strictly:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front
