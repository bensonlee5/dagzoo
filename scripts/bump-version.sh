#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<EOF
Usage: $(basename "$0") <major|minor|patch> [--dry-run] [--tag]

Bump the project version in pyproject.toml.

Arguments:
  major|minor|patch   Which semver component to bump.

Options:
  --dry-run           Print old/new version without making changes.
  --tag               Git-commit the change and create a vX.Y.Z tag.
  -h, --help          Show this help message.
EOF
}

# --- Parse arguments ---
BUMP_TYPE=""
DRY_RUN=false
TAG=false

for arg in "$@"; do
  case "$arg" in
    major|minor|patch) BUMP_TYPE="$arg" ;;
    --dry-run)         DRY_RUN=true ;;
    --tag)             TAG=true ;;
    -h|--help)         usage; exit 0 ;;
    *)                 echo "Error: unknown argument '$arg'" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$BUMP_TYPE" ]]; then
  echo "Error: bump type required (major, minor, or patch)." >&2
  usage >&2
  exit 1
fi

# --- Read current version (same regex as CI) ---
CURRENT=$(grep -m1 '^version' pyproject.toml | sed 's/.*= *"\(.*\)"/\1/')
if [[ -z "$CURRENT" ]]; then
  echo "Error: could not parse version from pyproject.toml" >&2
  exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

# --- Compute new version ---
case "$BUMP_TYPE" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"

echo "$CURRENT -> $NEW_VERSION"

if $DRY_RUN; then
  exit 0
fi

# --- Update pyproject.toml ---
sed -i.bak "s/^version = \"${CURRENT}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
rm -f pyproject.toml.bak

echo "Updated pyproject.toml"

# --- Optionally commit and tag ---
if $TAG; then
  # Warn about uncommitted changes in other files
  if [[ -n "$(git diff --name-only HEAD 2>/dev/null | grep -v '^pyproject.toml$' || true)" ]] || \
     [[ -n "$(git diff --cached --name-only HEAD 2>/dev/null | grep -v '^pyproject.toml$' || true)" ]]; then
    echo "Warning: other files have uncommitted changes." >&2
  fi

  git add pyproject.toml
  git commit -m "chore: bump version to ${NEW_VERSION}"
  git tag "v${NEW_VERSION}"
  echo "Created tag v${NEW_VERSION}"
  echo "Run: git push origin main --tags"
fi
