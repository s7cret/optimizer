from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

from optimizer import OptimizerConfig, Parameter, dry_run_validate, optimize
from optimizer.reporting.csv_report import write_csv
from optimizer.reporting.json_report import to_json
from optimizer.reporting.markdown_report import to_markdown
from optimizer.storage.json_backend import JsonStorage
from optimizer.storage.sqlite_backend import SQLiteStorage


def load_obj(spec: str) -> Callable[[dict[str, Any]], Any]:
    if ":" not in spec:
        raise ValueError("runner must use FILE:OBJECT format")
    file, name = spec.split(":", 1)
    if not file or not name:
        raise ValueError("runner must use FILE:OBJECT format")
    path = Path(file)
    spec_obj = importlib.util.spec_from_file_location(
        "optimizer_user_" + path.stem, path
    )
    if spec_obj is None or spec_obj.loader is None:
        raise ValueError(f"cannot load runner module from {path}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    obj = getattr(module, name, None)
    if not callable(obj):
        raise ValueError(f"runner object {name!r} in {path} is not callable")
    return obj


def load_params(path: str | Path) -> list[Parameter]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in parameter file {source}: {exc.msg}") from exc
    rows = data.get("parameters", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(
            "parameter file must contain a list or a {'parameters': [...]} object"
        )
    params: list[Parameter] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"parameter row {index} must be an object")
        params.append(Parameter(**row))
    return params


def _storage(result_dir: str | Path) -> JsonStorage | SQLiteStorage:
    directory = Path(result_dir)
    if (directory / "optimizer.sqlite").exists():
        return SQLiteStorage(directory)
    return JsonStorage(directory)


def _load_trials(result_dir: str | Path) -> list[dict[str, object]]:
    store = _storage(result_dir)
    try:
        return store.load_trials_raw()
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _cmd_export(result_dir: str | Path, fmt: str) -> None:
    rows = _load_trials(result_dir)
    out = Path(result_dir) / f"trials_export.{'md' if fmt == 'markdown' else fmt}"
    if fmt == "json":
        out.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n")
    elif fmt == "csv":
        keys = sorted({key for row in rows for key in row})
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    else:
        lines = ["# Optimizer trials export", "", f"Trials: {len(rows)}", ""]
        for row in rows[:50]:
            lines.append(
                f"- id={row.get('id')} status={row.get('status')} "
                f"objective={row.get('objective_value')} params={row.get('params')}"
            )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


def _cmd_analyze(result_dir: str | Path) -> None:
    rows = _load_trials(result_dir)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1

    def objective_key(row: dict[str, object]) -> float:
        value = row.get("objective_value")
        return float(value) if isinstance(value, int | float) else float("-inf")

    best = sorted(
        [row for row in rows if row.get("objective_value") is not None],
        key=objective_key,
        reverse=True,
    )[:5]
    print(
        json.dumps(
            {"trials": len(rows), "counts": counts, "top_objective_rows": best},
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def _run_or_resume(ns: argparse.Namespace) -> int:
    try:
        params = load_params(ns.params)
        runner = load_obj(ns.runner)
    except (OSError, TypeError, ValueError) as exc:
        print(f"optimizer: {exc}", file=sys.stderr)
        return 2
    config = OptimizerConfig(
        output_dir=Path(ns.output_dir),
        algorithm=ns.algorithm,
        objective=ns.objective,
        max_trials=ns.max_trials,
        resume=True,
    )
    result = optimize(params, runner, config)
    to_json(result, Path(ns.output_dir) / "result.json")
    write_csv(result.all_trials or [], Path(ns.output_dir) / "trials.csv")
    to_markdown(result, Path(ns.output_dir) / "report.md")
    print(to_markdown(result))
    return 0


def _dry_run(ns: argparse.Namespace) -> int:
    try:
        params = load_params(ns.params)
    except (OSError, TypeError, ValueError) as exc:
        print(f"optimizer: {exc}", file=sys.stderr)
        return 2
    result = dry_run_validate(params, OptimizerConfig(output_dir=Path(ns.output_dir)))
    Path(ns.output_dir).mkdir(parents=True, exist_ok=True)
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str)
    (Path(ns.output_dir) / "dry_run_validation.json").write_text(
        text + "\n", encoding="utf-8"
    )
    print(text)
    return 0 if result.status == "valid" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optimizer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run an optimization with a FILE:OBJECT runner")
    run.add_argument("--params", required=True)
    run.add_argument("--runner", required=True)
    run.add_argument("--output-dir", default="./optimizer_results")
    run.add_argument("--algorithm", default="grid")
    run.add_argument("--objective", default="net_profit")
    run.add_argument("--max-trials", type=int, default=1000)
    dry = sub.add_parser(
        "dry-run", help="validate a parameter file without executing a runner"
    )
    dry.add_argument("--params", required=True)
    dry.add_argument("--output-dir", default="./optimizer_results")
    analyze = sub.add_parser("analyze", help="summarize an existing result directory")
    analyze.add_argument("--result-dir", default="./optimizer_results")
    export = sub.add_parser(
        "export", help="export trials from an existing result directory"
    )
    export.add_argument("--result-dir", default="./optimizer_results")
    export.add_argument("--format", choices=["json", "csv", "markdown"], default="json")
    resume = sub.add_parser("resume", help="resume using the same inputs as run")
    resume.add_argument("--params", required=True)
    resume.add_argument("--runner", required=True)
    resume.add_argument("--output-dir", default="./optimizer_results")
    resume.add_argument("--algorithm", default="grid")
    resume.add_argument("--objective", default="net_profit")
    resume.add_argument("--max-trials", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    if ns.cmd in {"run", "resume"}:
        return _run_or_resume(ns)
    if ns.cmd == "dry-run":
        return _dry_run(ns)
    if ns.cmd == "analyze":
        _cmd_analyze(ns.result_dir)
        return 0
    _cmd_export(ns.result_dir, ns.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
