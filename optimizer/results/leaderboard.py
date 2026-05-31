from functools import cmp_to_key

from optimizer.core.objective import objective_sort_value


def _trial_compare(config):
    epsilon = float(getattr(config, "objective_tie_epsilon", 1e-12) or 0.0)
    secondary = getattr(config, "objective_secondary", None)
    secondary_direction = getattr(config, "objective_secondary_direction", "auto")

    def secondary_value(trial):
        if secondary is None:
            return None
        direction = secondary_direction
        if direction == "auto":
            from optimizer.core.objective import objective_direction

            direction = objective_direction(secondary, "auto")
        value = trial.metrics.get(secondary)
        if value is None:
            return None
        return objective_sort_value(float(value), direction)

    def compare(left, right):
        left_primary = objective_sort_value(left.objective_value, left.objective_direction)
        right_primary = objective_sort_value(right.objective_value, right.objective_direction)
        if abs(left_primary - right_primary) > epsilon:
            return -1 if left_primary > right_primary else 1
        left_secondary = secondary_value(left)
        right_secondary = secondary_value(right)
        if left_secondary is not None or right_secondary is not None:
            if left_secondary is None:
                return 1
            if right_secondary is None:
                return -1
            if left_secondary != right_secondary:
                return -1 if left_secondary > right_secondary else 1
        return left.id - right.id

    return compare


def rank_trials(trials, config=None):
    completed = [t for t in trials if t.status == "completed" and t.objective_value is not None]
    if config is None:
        completed.sort(
            key=lambda t: objective_sort_value(t.objective_value, t.objective_direction),
            reverse=True,
        )
    else:
        completed.sort(key=cmp_to_key(_trial_compare(config)))
    for i, t in enumerate(completed, 1):
        t.rank = i
    return completed


class Leaderboard:
    def __init__(self, trials):
        self.trials = rank_trials(list(trials))

    def top(self, n=20):
        return self.trials[:n]
