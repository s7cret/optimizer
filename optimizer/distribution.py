"""Deterministic source distribution helpers for Optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
BUILD_ARTIFACT_PARTS = {".git", "dist", "build", "optimizer_results"}
EXCLUDED_PARTS = {*CACHE_PARTS, *BUILD_ARTIFACT_PARTS}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".coverage", ".log"}
EXCLUDED_NAMES = {".coverage"}


@dataclass(frozen=True)
class DistributionManifest:
    package_version: str
    file_count: int
    sha256: str
    hygiene_ok: bool
    forbidden_files: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _version(root: Path) -> str:
    namespace: dict[str, str] = {}
    exec((root / "optimizer" / "version.py").read_text(encoding="utf-8"), namespace)
    return str(namespace["__version__"])


def _has_excluded_part(path: Path) -> bool:
    return any(
        part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.parts
    )


def _is_release_artifact(path: Path) -> bool:
    return (
        any(
            part in BUILD_ARTIFACT_PARTS or part.endswith(".egg-info")
            for part in path.parts
        )
        or path.suffix == ".zip"
    )


def _is_forbidden(path: Path) -> bool:
    return (
        _has_excluded_part(path)
        or path.suffix in EXCLUDED_SUFFIXES
        or path.name in EXCLUDED_NAMES
        or path.suffix == ".zip"
    )


def _include(path: Path) -> bool:
    return not _is_forbidden(path) and path.is_file()


def iter_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(p for p in root_path.rglob("*") if _include(p.relative_to(root_path)))


def _forbidden_entries(root_path: Path) -> list[str]:
    return sorted(
        path.relative_to(root_path).as_posix()
        for path in root_path.rglob("*")
        if _is_release_artifact(path.relative_to(root_path))
    )


def manifest(root: str | Path) -> DistributionManifest:
    root_path = Path(root)
    files = iter_files(root_path)
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root_path).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    forbidden = _forbidden_entries(root_path)
    return DistributionManifest(
        _version(root_path), len(files), digest.hexdigest(), not forbidden, forbidden
    )


def build_zip(root: str | Path, output: str | Path) -> Path:
    root_path = Path(root)
    output_path = Path(output)
    prefix = f"optimizer-{_version(root_path)}"
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in iter_files(root_path):
            rel = path.relative_to(root_path).as_posix()
            info = zipfile.ZipInfo(f"{prefix}/{rel}")
            info.date_time = (2024, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m optimizer.distribution")
    sub = parser.add_subparsers(dest="command", required=True)
    man = sub.add_parser("manifest")
    man.add_argument("--root", default=".")
    build = sub.add_parser("build-zip")
    build.add_argument("--root", default=".")
    build.add_argument("--output", required=True)
    ns = parser.parse_args(argv)
    if ns.command == "manifest":
        print(json.dumps(manifest(ns.root).to_dict(), indent=2, sort_keys=True))
        return 0
    path = build_zip(ns.root, ns.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
