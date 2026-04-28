from optimizer.core.objective import objective_sort_value


def baseline_comparison(baseline, trial):
    if not baseline or not trial:
        return {}
    delta = None if baseline.raw_objective_value is None or trial.raw_objective_value is None else trial.raw_objective_value - baseline.raw_objective_value
    direction = trial.objective_direction
    worse = False
    if baseline.raw_objective_value is not None and trial.raw_objective_value is not None:
        worse = objective_sort_value(trial.raw_objective_value, direction) < objective_sort_value(baseline.raw_objective_value, direction)
    return {'objective_delta': delta, 'recommended_worse_than_baseline': worse, 'baseline_trial_id': baseline.id, 'recommended_trial_id': trial.id}


def baseline_diagnostic(baseline, trial):
    return baseline_comparison(baseline, trial)
