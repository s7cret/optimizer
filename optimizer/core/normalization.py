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
    profit = metrics.get("net_profit") or metrics.get("net_profit_percent") or 0.0
    pf = metrics.get("profit_factor") or 1.0
    sharpe = metrics.get("sharpe_ratio") or 0.0
    dd = abs(metrics.get("max_drawdown_percent") or metrics.get("max_drawdown") or 0.0)
    return float(profit + 10 * pf + 5 * sharpe - dd)
