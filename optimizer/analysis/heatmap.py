from collections import defaultdict
from typing import Any


def export(trials: list[Any], x: str, y: str) -> dict[str, object]:
    grid: dict[object, dict[object, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for t in trials:
        if getattr(t, "status", None) != "completed" or t.objective_value is None:
            continue
        if x in t.params and y in t.params:
            grid[t.params[x]][t.params[y]].append(float(t.objective_value))
    matrix = {
        kx: {ky: sum(vals) / len(vals) for ky, vals in row.items()}
        for kx, row in grid.items()
    }
    return {
        "status": "ok" if matrix else "insufficient_data",
        "x": x,
        "y": y,
        "matrix": matrix,
    }
