from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "linear" / "github_to_linear.py"
SPEC = importlib.util.spec_from_file_location("github_to_linear", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _issue(
    number: int,
    *,
    body: str = "",
    state: str = "OPEN",
    labels: list[str] | None = None,
) -> object:
    label_objs = [
        MODULE.GitHubLabel(name=name, color="a2eeef", description=None) for name in labels or []
    ]
    return MODULE.GitHubIssue(
        number=number,
        title=f"Issue {number}",
        body=body,
        state=state,
        labels=label_objs,
        created_at="2026-03-08T00:00:00Z",
        updated_at="2026-03-08T01:00:00Z",
        closed_at="2026-03-08T02:00:00Z" if state == "CLOSED" else None,
        url=f"https://github.com/bensonlee5/dagzoo/issues/{number}",
    )


def test_parse_issue_references_deduplicates_and_ignores_text() -> None:
    refs = MODULE.parse_issue_references("Depends on #10, blocks #12, and repeats #10.")
    assert refs == [10, 12]


def test_build_migrated_description_includes_metadata_header() -> None:
    issue = _issue(42, body="Original body", labels=["enhancement", "P1"])
    description = MODULE.build_migrated_description(issue)
    assert "## GitHub Migration Metadata" in description
    assert "Original GitHub issue: `#42`" in description
    assert "Original body" in description


def test_desired_state_and_priority_mapping_follow_defaults() -> None:
    open_issue = _issue(10, labels=["P2"])
    closed_issue = _issue(11, state="CLOSED", labels=["P0"])

    assert MODULE.desired_linear_state_name(open_issue) == "Backlog"
    assert MODULE.desired_linear_priority(open_issue) == 3
    assert MODULE.desired_linear_state_name(closed_issue) == "Done"
    assert MODULE.desired_linear_priority(closed_issue) == 1


def test_linear_label_specs_omit_priority_labels() -> None:
    issues = [_issue(1, labels=["enhancement", "P1"]), _issue(2, labels=["documentation"])]
    specs = MODULE.linear_label_specs(issues)
    assert set(specs) == {"enhancement", "documentation"}


def test_epic_parent_map_links_children_from_epic_body() -> None:
    issues = [
        _issue(148, body="Tracks #145, #146, #147", labels=["epic"]),
        _issue(145),
        _issue(146),
        _issue(147),
    ]
    assert MODULE.epic_parent_map(issues) == {145: 148, 146: 148, 147: 148}


def test_epic_parent_map_skips_epic_children_to_avoid_cycles() -> None:
    issues = [
        _issue(148, body="Tracks #66 and #146", labels=["epic"]),
        _issue(66, body="Parent epic", labels=["epic"]),
        _issue(146),
    ]
    assert MODULE.epic_parent_map(issues) == {146: 148}
