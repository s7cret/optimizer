from pathlib import Path
from typing import Any


def export(result: Any, path: str | Path | None = None, fmt: str = "png") -> dict[str, object]:
    trials = [
        t
        for t in (getattr(result, "all_trials", None) or getattr(result, "top_trials", []) or [])
        if getattr(t, "objective_value", None) is not None
    ]
    if not trials:
        return {
            "status": "insufficient_data",
            "diagnostics": [
                {
                    "code": "PLOT_REQUIRES_TRIALS",
                    "severity": "warning",
                    "message": "No plottable trials",
                }
            ],
        }
    out = Path(path or f"optimizer_objective.{fmt}")
    if fmt == "html":
        rows = "\n".join(f"<tr><td>{t.id}</td><td>{t.objective_value}</td></tr>" for t in trials)
        out.write_text(
            f"<html><body><table><tr><th>trial</th><th>objective</th></tr>{rows}</table></body></html>"
        )
        return {"status": "ok", "path": str(out), "format": fmt}
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        return {
            "status": "dependency_missing",
            "dependency": "matplotlib",
            "diagnostics": [
                {"code": "PLOT_DEPENDENCY_MISSING", "severity": "warning", "message": str(exc)}
            ],
        }
    xs = [t.id for t in trials]
    ys = [t.objective_value for t in trials]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("trial")
    plt.ylabel("objective")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return {"status": "ok", "path": str(out), "format": fmt}
