from __future__ import annotations

from dataclasses import dataclass

from .impact import ImpactReport


@dataclass(frozen=True)
class ContractResult:
    ok: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def evaluate_release_contract(report: ImpactReport, *, strict: bool = False) -> ContractResult:
    changed = set(report.changed_files)
    version_changed = "pyproject.toml" in changed
    changelog_changed = "CHANGELOG.md" in changed
    has_release_risk = "release-risk" in report.tags
    python_code_changed = any(path.startswith("src/dagzoo/") for path in changed)

    errors: list[str] = []
    warnings: list[str] = []
    if has_release_risk and not (version_changed and changelog_changed):
        errors.append(
            "release-risk changes require updates to both `pyproject.toml` and `CHANGELOG.md`."
        )
    elif python_code_changed and not (version_changed and changelog_changed):
        warnings.append(
            "Internal-only refactors may skip a version bump, but user-facing behavior, schema, CLI, "
            "or artifact changes may not."
        )
        if strict:
            errors.append(
                "strict release-contract mode treats missing `pyproject.toml`/`CHANGELOG.md` updates as a failure."
            )

    return ContractResult(
        ok=not errors,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def render_contract_result(result: ContractResult) -> str:
    lines: list[str] = []
    lines.extend(f"warning: {warning}" for warning in result.warnings)
    lines.extend(f"error: {error}" for error in result.errors)
    if not lines:
        lines.append("release contract checks passed.")
    return "\n".join(lines) + "\n"
