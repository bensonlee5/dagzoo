from __future__ import annotations

import importlib.util
import json
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


class _FakeGH:
    def __init__(self, issues: list[object], *, dry_run: bool = False) -> None:
        self._issues = issues
        self.dry_run = dry_run

    def list_issues(self, issue_numbers: set[int] | None = None) -> list[object]:
        if issue_numbers is None:
            return list(self._issues)
        return [issue for issue in self._issues if issue.number in issue_numbers]


class _FakeLinear:
    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.update_calls: list[tuple[str, dict[str, object]]] = []

    def get_project(self, project_slug: str) -> tuple[str, str, str]:
        return ("project-1", "team-1", "DAG")

    def get_team_metadata(self, team_id: str) -> tuple[dict[str, object], dict[str, object]]:
        return (
            {
                "Backlog": MODULE.LinearState(id="state-backlog", name="Backlog", type="backlog"),
                "Done": MODULE.LinearState(id="state-done", name="Done", type="completed"),
                "In Progress": MODULE.LinearState(
                    id="state-in-progress",
                    name="In Progress",
                    type="started",
                ),
            },
            {},
        )

    def ensure_workflow_states(self, team_id: str, states: dict[str, object]) -> dict[str, object]:
        return dict(states)

    def ensure_labels(
        self,
        team_id: str,
        labels: dict[str, object],
        label_specs: dict[str, dict[str, str | None]],
    ) -> dict[str, object]:
        return dict(labels)

    def existing_project_issue_map(self, project_id: str) -> dict[int, object]:
        return {}

    def create_issue(self, input_payload: dict[str, object]) -> object:
        return MODULE.LinearIssueRef(
            id="linear-created",
            identifier="DAG-100",
            url="https://linear.app/dagzoo/issue/DAG-100",
        )

    def update_issue(self, issue_id: str, input_payload: dict[str, object]) -> object:
        self.update_calls.append((issue_id, dict(input_payload)))
        return MODULE.LinearIssueRef(
            id=issue_id,
            identifier=f"{issue_id}-IDENT",
            url=f"https://linear.app/dagzoo/issue/{issue_id}",
        )


def test_migrate_issues_dry_run_does_not_persist_mapping(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    original_mapping = {
        "repo": "bensonlee5/dagzoo",
        "project_slug": "proj",
        "migrated_at": "2026-03-08T00:00:00Z",
        "entries": [
            {
                "github_number": 10,
                "github_url": "https://github.com/bensonlee5/dagzoo/issues/10",
                "github_state": "OPEN",
                "linear_id": "linear-10",
                "linear_identifier": "DAG-10",
                "linear_url": "https://linear.app/dagzoo/issue/DAG-10",
                "linear_state": "Backlog",
                "parent_github_number": None,
                "cutover_applied": False,
                "cutover_applied_at": None,
            }
        ],
    }
    mapping_path.write_text(json.dumps(original_mapping, indent=2) + "\n")

    mapping = MODULE.migrate_issues(
        gh=_FakeGH([_issue(10)], dry_run=True),
        linear=_FakeLinear(dry_run=True),
        repo="bensonlee5/dagzoo",
        project_slug="proj",
        mapping_path=mapping_path,
        issue_numbers=None,
    )

    assert mapping.entry_for(10) is not None
    assert json.loads(mapping_path.read_text()) == original_mapping


def test_main_cutover_dry_run_does_not_persist_mapping(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping = MODULE.MappingFile(
        repo="bensonlee5/dagzoo",
        project_slug="proj",
        migrated_at="2026-03-08T00:00:00Z",
        entries={
            10: MODULE.MappingEntry(
                github_number=10,
                github_url="https://github.com/bensonlee5/dagzoo/issues/10",
                github_state="OPEN",
                linear_id="linear-10",
                linear_identifier="DAG-10",
                linear_url="https://linear.app/dagzoo/issue/DAG-10",
                linear_state="Backlog",
            )
        },
    )
    original_payload = json.loads(mapping.to_json())
    mapping_path.write_text(mapping.to_json())
    api_key_path = tmp_path / "linear.key"
    api_key_path.write_text("test-key\n")

    exit_code = MODULE.main(
        [
            "--repo",
            "bensonlee5/dagzoo",
            "--linear-api-key-file",
            str(api_key_path),
            "--project-slug",
            "proj",
            "--mapping-path",
            str(mapping_path),
            "--mode",
            "cutover",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert json.loads(mapping_path.read_text()) == original_payload


def test_migrate_issues_preserves_linear_state_after_cutover(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping = MODULE.MappingFile(
        repo="bensonlee5/dagzoo",
        project_slug="proj",
        migrated_at="2026-03-08T00:00:00Z",
        entries={
            12: MODULE.MappingEntry(
                github_number=12,
                github_url="https://github.com/bensonlee5/dagzoo/issues/12",
                github_state="OPEN",
                linear_id="linear-12",
                linear_identifier="DAG-12",
                linear_url="https://linear.app/dagzoo/issue/DAG-12",
                linear_state="In Progress",
                cutover_applied=True,
                cutover_applied_at="2026-03-08T01:00:00Z",
            )
        },
    )
    mapping_path.write_text(mapping.to_json())
    linear = _FakeLinear()

    migrated = MODULE.migrate_issues(
        gh=_FakeGH([_issue(12, state="CLOSED")]),
        linear=linear,
        repo="bensonlee5/dagzoo",
        project_slug="proj",
        mapping_path=mapping_path,
        issue_numbers=None,
    )

    assert linear.update_calls == [
        (
            "linear-12",
            {
                "teamId": "team-1",
                "projectId": "project-1",
                "title": "Issue 12",
                "description": MODULE.build_migrated_description(_issue(12, state="CLOSED")),
            },
        )
    ]
    assert migrated.entry_for(12).linear_state == "In Progress"


def test_migrate_issues_updates_state_before_cutover(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping = MODULE.MappingFile(
        repo="bensonlee5/dagzoo",
        project_slug="proj",
        migrated_at="2026-03-08T00:00:00Z",
        entries={
            14: MODULE.MappingEntry(
                github_number=14,
                github_url="https://github.com/bensonlee5/dagzoo/issues/14",
                github_state="OPEN",
                linear_id="linear-14",
                linear_identifier="DAG-14",
                linear_url="https://linear.app/dagzoo/issue/DAG-14",
                linear_state="Backlog",
                cutover_applied=False,
            )
        },
    )
    mapping_path.write_text(mapping.to_json())
    linear = _FakeLinear()

    migrated = MODULE.migrate_issues(
        gh=_FakeGH([_issue(14, state="CLOSED")]),
        linear=linear,
        repo="bensonlee5/dagzoo",
        project_slug="proj",
        mapping_path=mapping_path,
        issue_numbers=None,
    )

    assert linear.update_calls[0][1]["stateId"] == "state-done"
    assert migrated.entry_for(14).linear_state == "Done"
