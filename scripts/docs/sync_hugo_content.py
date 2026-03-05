#!/usr/bin/env python3
"""Sync docs sources into the Hugo docs site.

This script keeps canonical docs in `docs/` and renders generated site inputs
under `site/.generated/`:

- Markdown user guides -> `site/.generated/content/docs/**`
- Canonical HTML docs  -> `site/.generated/static/canonical/**`
- Wrapper pages + development links page -> `site/.generated/content/docs/**`

Use `--check` in CI to fail when generated site inputs are out of date.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
SITE_ROOT = REPO_ROOT / "site"
GENERATED_ROOT = SITE_ROOT / ".generated"
CONTENT_DOCS_ROOT = GENERATED_ROOT / "content" / "docs"
STATIC_CANONICAL_ROOT = GENERATED_ROOT / "static" / "canonical"

GITHUB_DEV_DOCS_BASE = "https://github.com/bensonlee5/dagzoo/blob/main/docs/development"

CANONICAL_FILES = [
    "how-it-works.html",
    "transforms.html",
    "canonical.css",
]

USER_MD_SOURCES = [
    "usage-guide.md",
    "output-format.md",
    "features/diagnostics.md",
    "features/missingness.md",
    "features/many-class.md",
    "features/shift.md",
    "features/noise.md",
    "features/benchmark-guardrails.md",
]

MD_DESCRIPTIONS: dict[str, str] = {
    "usage-guide.md": "Command workflows and practical usage patterns for generation and benchmarking.",
    "output-format.md": "Artifact schema, metadata contract, and shard layout.",
    "features/diagnostics.md": "Runtime observability metrics and diagnostic outputs.",
    "features/missingness.md": "Controlled injection of missing values into generated datasets.",
    "features/many-class.md": "Multi-class target generation with configurable class counts.",
    "features/shift.md": "Distribution-shift controls for graph, mechanism, and noise.",
    "features/noise.md": "Noise family selection, mixture modes, and per-dataset resolution.",
    "features/benchmark-guardrails.md": "Automated quality checks for benchmark suite runs.",
}

MD_WEIGHTS: dict[str, int] = {
    "usage-guide.md": 30,
    "output-format.md": 40,
    "features/diagnostics.md": 60,
    "features/missingness.md": 61,
    "features/many-class.md": 62,
    "features/shift.md": 63,
    "features/noise.md": 64,
    "features/benchmark-guardrails.md": 65,
}

WRAPPER_PAGES: dict[str, tuple[str, int, str, str]] = {
    "how-it-works.md": (
        "How It Works",
        10,
        "/canonical/how-it-works.html",
        "End-to-end runtime behavior, core concepts, and pipeline walkthrough.",
    ),
    "transforms.md": (
        "Transforms (Canonical Math)",
        20,
        "/canonical/transforms.html",
        "Canonical mathematical specification for generation transforms.",
    ),
}

# Standard Markdown inline-link pattern: [label](target)
MD_LINK_RE = re.compile(r"(!?\[[^\]]+\]\()([^\)]+)(\))")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_base_path() -> str:
    """Extract URL path prefix from Hugo baseURL config."""
    text = _read_text(SITE_ROOT / "hugo.yaml")
    match = re.search(r"^baseURL:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return ""
    return urlparse(match.group(1).strip()).path.rstrip("/")


def _title_from_markdown(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def _strip_matching_h1(content: str, title: str) -> str:
    lines = content.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx < len(lines) and lines[idx].strip().startswith("# "):
        heading = lines[idx].strip()[2:].strip()
        if heading.lower() == title.lower():
            del lines[idx]
            if idx < len(lines) and not lines[idx].strip():
                del lines[idx]

    stripped = "\n".join(lines).rstrip()
    return f"{stripped}\n" if stripped else ""


def _slug_title(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def _front_matter(title: str, weight: int, description: str | None = None) -> str:
    safe_title = title.replace('"', '\\"')
    lines = [f'title: "{safe_title}"']
    if description:
        safe_desc = description.replace('"', '\\"')
        lines.append(f'description: "{safe_desc}"')
    lines.append(f"weight: {weight}")
    return "---\n" + "\n".join(lines) + "\n---\n\n"


def _normalize_link(source_rel: str, target: str) -> str:
    source_dir = posixpath.dirname(source_rel)
    joined = posixpath.join(source_dir, target)
    return posixpath.normpath(joined)


def _rewrite_markdown_links(content: str, source_rel: str, route_map: dict[str, str]) -> str:
    def replacer(match: re.Match[str]) -> str:
        prefix, raw_target, suffix = match.groups()
        target = raw_target.strip()

        if not target:
            return match.group(0)

        if target.startswith(("http://", "https://", "mailto:", "tel:", "#", "/")):
            return match.group(0)

        path, anchor = (target.split("#", 1) + [""])[:2]
        normalized = _normalize_link(source_rel, path)

        new_target = route_map.get(normalized)
        if new_target is None and normalized.startswith("development/"):
            new_target = f"{GITHUB_DEV_DOCS_BASE}/{normalized.removeprefix('development/')}"

        if new_target is None:
            return match.group(0)

        if anchor:
            new_target = f"{new_target}#{anchor}"

        return f"{prefix}{new_target}{suffix}"

    return MD_LINK_RE.sub(replacer, content)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sync_text(path: Path, expected: str, check: bool, changed: list[Path]) -> None:
    _ensure_parent(path)
    current = _read_text(path) if path.exists() else None
    if current == expected:
        return
    changed.append(path)
    if not check:
        path.write_text(expected, encoding="utf-8")


def _sync_bytes(path: Path, expected: bytes, check: bool, changed: list[Path]) -> None:
    _ensure_parent(path)
    current = path.read_bytes() if path.exists() else None
    if current == expected:
        return
    changed.append(path)
    if not check:
        path.write_bytes(expected)


def _build_route_map(base_path: str) -> dict[str, str]:
    route_map: dict[str, str] = {
        "how-it-works.html": f"{base_path}/canonical/how-it-works.html",
        "transforms.html": f"{base_path}/canonical/transforms.html",
    }

    for rel in USER_MD_SOURCES:
        if rel.startswith("features/"):
            slug = Path(rel).stem
            route_map[rel] = f"{base_path}/docs/features/{slug}/"
        else:
            slug = Path(rel).stem
            route_map[rel] = f"{base_path}/docs/{slug}/"

    return route_map


def _sync_user_markdown(route_map: dict[str, str], check: bool, changed: list[Path]) -> None:
    for rel in USER_MD_SOURCES:
        src = DOCS_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing source Markdown file: {src}")

        source_text = _read_text(src)
        rewritten = _rewrite_markdown_links(source_text, rel, route_map)

        title = _title_from_markdown(rewritten, fallback=_slug_title(Path(rel).stem))
        rewritten = _strip_matching_h1(rewritten, title=title)
        weight = MD_WEIGHTS.get(rel, 100)
        description = MD_DESCRIPTIONS.get(rel)
        out_text = _front_matter(title=title, weight=weight, description=description) + rewritten

        dest = CONTENT_DOCS_ROOT / rel
        _sync_text(dest, out_text, check=check, changed=changed)


def _sync_canonical_assets(check: bool, changed: list[Path]) -> None:
    for rel in CANONICAL_FILES:
        src = DOCS_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing canonical file: {src}")
        dest = STATIC_CANONICAL_ROOT / rel
        _sync_bytes(dest, src.read_bytes(), check=check, changed=changed)


def _render_wrapper_page(
    title: str, weight: int, canonical_url: str, description: str | None = None
) -> str:
    fm = _front_matter(title=title, weight=weight, description=description)
    body = (
        "Canonical source page:\n\n"
        f"- [Open in full view]({canonical_url})\n\n"
        '<p class="canonical-doc-note">'
        "Use the embedded view below for inline reading, or open the full page if you prefer a standalone tab."
        "</p>\n\n"
        f'<iframe class="canonical-doc-frame" src="{canonical_url}" loading="lazy"></iframe>\n'
    )
    return fm + body


def _sync_wrapper_pages(base_path: str, check: bool, changed: list[Path]) -> None:
    for filename, (title, weight, canonical_url, description) in WRAPPER_PAGES.items():
        dest = CONTENT_DOCS_ROOT / filename
        prefixed_url = f"{base_path}{canonical_url}"
        content = _render_wrapper_page(
            title=title,
            weight=weight,
            canonical_url=prefixed_url,
            description=description,
        )
        _sync_text(dest, content, check=check, changed=changed)


def _humanize_dev_name(filename: str) -> str:
    return filename.replace("_", " ").replace("-", " ").replace(".md", "").title()


def _sync_development_links_page(check: bool, changed: list[Path]) -> None:
    dev_dir = DOCS_ROOT / "development"
    dev_docs = sorted(p.name for p in dev_dir.glob("*.md"))
    if not dev_docs:
        raise FileNotFoundError(f"No development docs found in: {dev_dir}")

    lines: list[str] = [
        _front_matter(title="Development References", weight=90),
        "Phase 1 keeps development docs canonical in repo Markdown and links them directly from the docs site.\n",
        "",
    ]
    for filename in dev_docs:
        label = _humanize_dev_name(filename)
        url = f"{GITHUB_DEV_DOCS_BASE}/{filename}"
        lines.append(f"- [{label}]({url})")

    lines.append("")
    text = "\n".join(lines)
    _sync_text(CONTENT_DOCS_ROOT / "development.md", text, check=check, changed=changed)


def sync(check: bool) -> list[Path]:
    changed: list[Path] = []
    base_path = _read_base_path()
    route_map = _build_route_map(base_path)

    _sync_canonical_assets(check=check, changed=changed)
    _sync_wrapper_pages(base_path=base_path, check=check, changed=changed)
    _sync_user_markdown(route_map=route_map, check=check, changed=changed)
    _sync_development_links_page(check=check, changed=changed)

    return changed


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write files; fail if outputs are out of sync.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    changed = sync(check=args.check)

    if args.check:
        if changed:
            print("Docs sync is out of date. Regenerate with:")
            print("  .venv/bin/python scripts/docs/sync_hugo_content.py")
            print("\nFiles needing update:")
            for path in changed:
                print(f"- {path.relative_to(REPO_ROOT)}")
            return 1
        print("Docs sync check passed.")
        return 0

    if changed:
        print(f"Updated {len(changed)} file(s):")
        for path in changed:
            print(f"- {path.relative_to(REPO_ROOT)}")
    else:
        print("No updates needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
