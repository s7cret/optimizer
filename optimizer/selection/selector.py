from optimizer.core.metric_registry import MetricRegistry
from optimizer.results.leaderboard import rank_trials
from optimizer.results.profile_result import ResultProfile
from optimizer.selection.pareto import pareto_front
from optimizer.selection.pareto_knee import pareto_knee


def _best(trials, key, reverse=True):
    vals = [t for t in trials if t.status == "completed" and key(t) is not None]
    return sorted(vals, key=key, reverse=reverse)[0] if vals else None


def _metric_best(trials, metric):
    direction = MetricRegistry().direction(metric)
    reverse = direction != "minimize"
    return _best(trials, lambda x, m=metric: x.metrics.get(m), reverse)


def build_profiles(trials, config):
    pool = (
        list(trials)
        if getattr(config, "include_baseline_in_optimization_candidates", False)
        else [t for t in trials if not t.is_baseline]
    )
    completed = [t for t in pool if t.status == "completed"]
    passed = [t for t in completed if t.passed_constraints]
    # Threshold-first selection: profiles other than best_objective must never
    # promote hard-constraint violators. If nothing passed, those profiles are
    # intentionally empty and choose_recommended() returns None for risk modes.
    candidates = passed
    profiles = {}
    ranked_completed = rank_trials(completed, config)
    bo = ranked_completed[0] if ranked_completed else None
    profiles["best_objective"] = ResultProfile(
        "best_objective",
        bo,
        "best objective value respecting objective direction",
        "objective_value",
        None if bo is None else bo.objective_value,
    )
    ranked_passed = rank_trials(passed, config)
    bp = ranked_passed[0] if ranked_passed else None
    profiles["best_passed_constraints"] = ResultProfile(
        "best_passed_constraints",
        bp,
        "best objective among trials passing hard thresholds",
        "objective_value",
        None if bp is None else bp.objective_value,
    )
    bb = _best(candidates, lambda t: t.balanced_score, True)
    profiles["best_balanced"] = ResultProfile(
        "best_balanced",
        bb,
        "best combined profit/risk balanced score",
        "balanced_score",
        None if bb is None else bb.balanced_score,
    )
    mr = _metric_best(candidates, "robustness_score")
    profiles["most_robust"] = ResultProfile(
        "most_robust",
        mr,
        "highest robustness/neighborhood score",
        "robustness_score",
        None if mr is None else mr.robustness_score,
    )
    for name, metric, reason in [
        ("best_profit", "net_profit", "highest net profit"),
        ("best_drawdown", "max_drawdown_percent", "best drawdown by registry direction"),
        ("best_profit_factor", "profit_factor", "highest profit factor"),
        ("best_sharpe", "sharpe_ratio", "highest Sharpe ratio"),
    ]:
        t = _metric_best(candidates, metric)
        profiles[name] = ResultProfile(
            name, t, reason, metric, None if t is None else t.metrics.get(metric)
        )
    pf = pareto_front(candidates)
    profiles["pareto_front"] = ResultProfile(
        "pareto_front",
        pf[0] if pf else None,
        f"{len(pf)} non-dominated candidates",
        "pareto",
        float(len(pf)) if pf else 0.0,
    )
    pk = pareto_knee(candidates)
    profiles["pareto_knee"] = ResultProfile(
        "pareto_knee",
        pk,
        "closest Pareto knee to ideal risk/profit point",
        "pareto_knee",
        None if pk is None else pk.objective_value,
    )
    return profiles, pf


def choose_recommended(profiles, mode="best_after_constraints"):
    order = {
        "best_objective": "best_objective",
        "best_after_constraints": "best_passed_constraints",
        "balanced": "best_balanced",
        "robust": "most_robust",
        "conservative": "best_drawdown",
        "aggressive": "best_profit",
        "pareto": "pareto_front",
        "pareto_knee": "pareto_knee",
    }
    name = order.get(mode, "best_passed_constraints")
    p = profiles.get(name)
    trial = p.trial if p else None
    if name != "best_objective" and trial is not None and not trial.passed_constraints:
        trial = None
    return trial, name
