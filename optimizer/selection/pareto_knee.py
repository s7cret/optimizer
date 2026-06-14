import math
from optimizer.selection.pareto import pareto_front


def pareto_knee(trials, metrics=("max_drawdown_percent", "net_profit")):
    front = pareto_front(trials, metrics, ("minimize", "maximize"))
    vals = []
    for t in front:
        x = t.metrics.get(metrics[0])
        y = t.metrics.get(metrics[1])
        if x is not None and y is not None:
            vals.append((t, float(x), float(y)))
    if not vals:
        return None
    minx, maxx = min(v[1] for v in vals), max(v[1] for v in vals)
    miny, maxy = min(v[2] for v in vals), max(v[2] for v in vals)

    def norm(v, lo, hi):
        return 1.0 if hi == lo else (v - lo) / (hi - lo)

    return min(
        vals,
        key=lambda r: math.dist(
            (norm(r[1], minx, maxx), norm(r[2], miny, maxy)), (0, 1)
        ),
    )[0]
