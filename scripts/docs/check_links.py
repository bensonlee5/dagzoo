#!/usr/bin/env python3
"""Lightweight link checker for docs Markdown/HTML sources.

Checks local links in:
- `docs/**`
- `site/content/**`
- `site/static/canonical/**`

It validates file targets and Hugo-style route links used by the generated docs
site (for example `/docs/usage-guide/` and `/canonical/transforms.html`).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROOTS = [
    "docs",
    "site/content",
    "site/.generated/content",
    "site/.generated/static/canonical",
]

MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK_RE = re.compile(r"(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

SKIP_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "javascript:",
    "data:",
)


def _iter_doc_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".html"}:
            yield path


def _normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target and not target.startswith("http"):
        # Handles inline title form: path "title"
        target = target.split(" ", 1)[0]
    return target


def _route_candidates(route: str) -> list[Path]:
    route = route.strip("/")
    if not route:
        return [
            REPO_ROOT / "site/content/_index.md",
            REPO_ROOT / "site/.generated/content/_index.md",
        ]

    if route.startswith("canonical/"):
        remainder = route.removeprefix("canonical/")
        return [
            REPO_ROOT / "site/static/canonical" / remainder,
            REPO_ROOT / "site/.generated/static/canonical" / remainder,
        ]

    candidates: list[Path] = []
    for content_root in (REPO_ROOT / "site/content", REPO_ROOT / "site/.generated/content"):
        route_path = content_root / route
        candidates.extend(
            [
                route_path.with_suffix(".md"),
                route_path / "index.md",
                route_path / "_index.md",
                route_path.with_suffix(".html"),
                route_path / "index.html",
            ]
        )
    return candidates


def _exists_target(source: Path, target: str) -> bool:
    if target.startswith("/"):
        return any(candidate.exists() for candidate in _route_candidates(target))

    target_path = source.parent / target
    if target_path.exists():
        return True

    if target_path.suffix:
        return False

    candidates = [
        target_path.with_suffix(".md"),
        target_path.with_suffix(".html"),
        target_path / "index.md",
        target_path / "_index.md",
        target_path / "index.html",
    ]
    return any(candidate.exists() for candidate in candidates)


def _collect_targets(line: str, suffix: str) -> list[str]:
    regex = MD_LINK_RE if suffix == ".md" else HTML_LINK_RE
    return [match.group(1) for match in regex.finditer(line)]


def _scan_file(path: Path) -> list[tuple[int, str]]:
    errors: list[tuple[int, str]] = []
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    for lineno, line in enumerate(text.splitlines(), start=1):
        for raw_target in _collect_targets(line, suffix):
            target = _normalize_target(raw_target)
            if not target or target.startswith(SKIP_PREFIXES) or target.startswith("#"):
                continue

            target_path = target.split("#", 1)[0]
            if not target_path:
                continue

            if not _exists_target(path, target_path):
                errors.append((lineno, target))

    return errors


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Root directories to scan (relative to repo root).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    all_errors: list[tuple[Path, int, str]] = []
    for root_rel in args.roots:
        root = REPO_ROOT / root_rel
        for path in _iter_doc_files(root):
            for lineno, target in _scan_file(path):
                all_errors.append((path, lineno, target))

    if all_errors:
        print("Broken links found:")
        for path, lineno, target in all_errors:
            rel = path.relative_to(REPO_ROOT)
            print(f"- {rel}:{lineno} -> {target}")
        return 1

    print("Link check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
