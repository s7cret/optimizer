def normalize_trials(trials, metric, direction="maximize"):
    vals = [t.metrics.get(metric) for t in trials if t.metrics.get(metric) is not None]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    out = {}
    for t in trials:
        v = t.metrics.get(metric)
        if v is None:
            out[t.id] = None
        elif hi == lo:
            out[t.id] = 1.0
        else:
            n = (v - lo) / (hi - lo)
            out[t.id] = n if direction == "maximize" else 1 - n
    return out


def balanced_score(metrics):
    # Absence and an actual zero are different observations.
    def observed(primary, fallback=None, default=0.0):
        value = metrics.get(primary)
        if value is None and fallback is not None:
            value = metrics.get(fallback)
        return default if value is None else value

    profit = observed("net_profit", "net_profit_percent")
    pf = observed("profit_factor", default=1.0)
    sharpe = observed("sharpe_ratio")
    dd = abs(observed("max_drawdown_percent", "max_drawdown"))
    return float(profit + 10 * pf + 5 * sharpe - dd)
