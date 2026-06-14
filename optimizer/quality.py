"""Small dependency-free quality gates used by release scripts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DuplicateReport:
    duplicate_group_count: int
    groups: list[list[dict[str, object]]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureReport:
    max_lines: int
    oversized_count: int
    oversized_files: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _python_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.startswith(".")
    )


def _function_fingerprints(path: Path) -> Iterable[tuple[str, str, int, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out: list[tuple[str, str, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Ignore tiny one-liners; duplicate tiny protocol stubs are not useful signal.
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        if end - start < 4:
            continue
        clone = ast.fix_missing_locations(node)
        dump = ast.dump(clone, include_attributes=False)
        out.append((hashlib.sha256(dump.encode()).hexdigest(), node.name, start, end))
    return out


def duplicates(root: str | Path) -> DuplicateReport:
    root_path = Path(root)
    seen: dict[str, list[dict[str, object]]] = {}
    for path in _python_files(root_path):
        for digest, name, start, end in _function_fingerprints(path):
            seen.setdefault(digest, []).append(
                {"path": str(path), "name": name, "start": start, "end": end}
            )
    groups = [items for items in seen.values() if len(items) > 1]
    return DuplicateReport(len(groups), groups)


def architecture(root: str | Path, *, max_lines: int = 700) -> ArchitectureReport:
    oversized: list[dict[str, object]] = []
    for path in _python_files(Path(root)):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > max_lines:
            oversized.append({"path": str(path), "lines": line_count})
    return ArchitectureReport(max_lines, len(oversized), oversized)


def _print(report: object) -> None:
    data = report.to_dict() if hasattr(report, "to_dict") else report
    print(json.dumps(data, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m optimizer.quality")
    sub = parser.add_subparsers(dest="command", required=True)
    dup = sub.add_parser("duplicates")
    dup.add_argument("root", nargs="?", default="optimizer")
    arch = sub.add_parser("architecture")
    arch.add_argument("root", nargs="?", default="optimizer")
    arch.add_argument("--max-lines", type=int, default=700)
    ns = parser.parse_args(argv)
    if ns.command == "duplicates":
        report = duplicates(ns.root)
        _print(report)
        return 1 if report.duplicate_group_count else 0
    architecture_report = architecture(ns.root, max_lines=ns.max_lines)
    _print(architecture_report)
    return 1 if architecture_report.oversized_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
