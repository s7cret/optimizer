def to_markdown(result, path=None):
    lines = [
        "# Optimizer Report",
        "",
        f"Recommended profile: {result.recommended_profile}",
        f"Recommended trial: {None if result.recommended_trial is None else result.recommended_trial.id}",
        "",
        "## Profiles",
    ]
    for name, p in result.profiles.items():
        lines.append(
            f"- **{name}**: trial={None if p.trial is None else p.trial.id}, {p.reason}, {p.score_name}={p.score_value}"
        )
    lines += [
        "",
        "## Top trials",
        "| rank | id | objective | passed | params |",
        "|---:|---:|---:|:---:|---|",
    ]
    for t in result.top_trials:
        lines.append(
            f"| {t.rank} | {t.id} | {t.objective_value} | {t.passed_constraints} | `{t.params}` |"
        )
    text = "\n".join(lines) + "\n"
    if path:
        open(path, "w").write(text)
    return text
