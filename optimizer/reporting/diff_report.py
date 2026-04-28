from typing import Any


def _trial_dict(t: Any) -> dict[str, Any]:
    if t is None:
        return {}
    if isinstance(t, dict):
        return t
    return {
        "id": getattr(t, "id", None),
        "params": getattr(t, "params", {}),
        "metrics": getattr(t, "metrics", {}),
        "objective_value": getattr(t, "objective_value", None),
    }


def diff(a: Any, b: Any) -> dict[str, object]:
    ta = _trial_dict(getattr(a, "recommended_trial", a))
    tb = _trial_dict(getattr(b, "recommended_trial", b))
    params_a = ta.get("params") or {}
    params_b = tb.get("params") or {}
    metrics_a = ta.get("metrics") or {}
    metrics_b = tb.get("metrics") or {}
    param_changes = {
        k: {"a": params_a.get(k), "b": params_b.get(k)}
        for k in sorted(set(params_a) | set(params_b))
        if params_a.get(k) != params_b.get(k)
    }
    metric_changes = {}
    for k in sorted(set(metrics_a) | set(metrics_b)):
        av = metrics_a.get(k)
        bv = metrics_b.get(k)
        if av == bv:
            continue
        metric_changes[k] = {
            "a": av,
            "b": bv,
            "delta": float(bv) - float(av)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float))
            else None,
        }
    oa = ta.get("objective_value")
    ob = tb.get("objective_value")
    objective_delta = (
        float(ob) - float(oa)
        if isinstance(oa, (int, float)) and isinstance(ob, (int, float))
        else None
    )
    return {
        "status": "ok",
        "recommended_trial_ids": {"a": ta.get("id"), "b": tb.get("id")},
        "objective_delta": objective_delta,
        "param_changes": param_changes,
        "metric_changes": metric_changes,
    }
