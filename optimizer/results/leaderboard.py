from optimizer.core.objective import objective_sort_value


def rank_trials(trials):
    completed = [t for t in trials if t.status == 'completed' and t.objective_value is not None]
    completed.sort(key=lambda t: objective_sort_value(t.objective_value, t.objective_direction), reverse=True)
    for i, t in enumerate(completed, 1): t.rank = i
    return completed


class Leaderboard:
    def __init__(self, trials): self.trials = rank_trials(list(trials))
    def top(self, n=20): return self.trials[:n]
