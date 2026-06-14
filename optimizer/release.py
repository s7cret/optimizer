"""Optimizer release readiness manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from optimizer.distribution import manifest as distribution_manifest
from optimizer.quality import architecture, duplicates
from optimizer.version import __version__

REQUIRED_DOCS = (
    "README.md",
    "CHANGELOG.md",
    "docs/README.md",
    "docs/ARCHITECTURE.md",
    "docs/DEVELOPMENT.md",
    "docs/RELEASE_4_0.md",
)


@dataclass(frozen=True)
class ReleaseManifest:
    package_version: str
    ok: bool
    docs_ok: bool
    missing_docs: list[str]
    duplicate_group_count: int
    architecture_oversized_count: int
    distribution_hygiene_ok: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_manifest(root: str | Path = ".") -> ReleaseManifest:
    root_path = Path(root)
    missing = [name for name in REQUIRED_DOCS if not (root_path / name).exists()]
    dup = duplicates(root_path / "optimizer")
    arch = architecture(root_path / "optimizer", max_lines=700)
    dist = distribution_manifest(root_path)
    ok = (
        not missing
        and not dup.duplicate_group_count
        and not arch.oversized_count
        and dist.hygiene_ok
    )
    return ReleaseManifest(
        package_version=__version__,
        ok=ok,
        docs_ok=not missing,
        missing_docs=missing,
        duplicate_group_count=dup.duplicate_group_count,
        architecture_oversized_count=arch.oversized_count,
        distribution_hygiene_ok=dist.hygiene_ok,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m optimizer.release")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", dest="json_path")
    ns = parser.parse_args(argv)
    report = build_manifest(ns.root)
    text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if ns.json_path:
        Path(ns.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
