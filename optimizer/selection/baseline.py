def baseline_diagnostic(baseline, trial):
    if not baseline or not trial: return {}
    return {'objective_delta': None if baseline.objective_value is None or trial.objective_value is None else trial.objective_value-baseline.objective_value}
