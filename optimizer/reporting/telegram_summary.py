def summarize(result):
    return f"Optimizer recommended {result.recommended_profile}: trial {getattr(result.recommended_trial, 'id', None)}"
