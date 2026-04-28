def compute_neighborhood_robustness(trials):
    # Lightweight MVP: percentile-like score from objective consistency is left as diagnostic placeholder.
    return {t.id: t.metrics.get('robustness_score') for t in trials}
