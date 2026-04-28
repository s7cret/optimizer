def print_summary(result):
    print(
        f"Recommended: {result.recommended_profile} trial={getattr(result.recommended_trial, 'id', None)}"
    )
