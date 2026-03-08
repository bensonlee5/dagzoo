from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("github_to_linear", "scripts/linear/github_to_linear.py")
MODULE = _load_module("seed_harness_backlog", "scripts/linear/seed_harness_backlog.py")


class _RecordingLinear:
    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.calls: list[tuple[str, dict[str, object]]] = []

    def graphql(self, query: str, variables: dict[str, object]) -> dict[str, object]:
        self.calls.append((query, dict(variables)))
        if "issueLabelCreate" in query:
            input_payload = variables["input"]
            assert isinstance(input_payload, dict)
            label_name = str(input_payload["name"])
            return {
                "issueLabelCreate": {
                    "success": True,
                    "issueLabel": {
                        "id": f"label-{label_name}",
                        "name": label_name,
                        "color": input_payload.get("color"),
                    },
                }
            }
        if "issueCreate" in query:
            return {
                "issueCreate": {
                    "success": True,
                    "issue": {
                        "id": "issue-created",
                        "identifier": "DAG-1",
                        "url": "https://linear.app/dagzoo/issue/DAG-1",
                    },
                }
            }
        if "issueUpdate" in query:
            return {
                "issueUpdate": {
                    "success": True,
                    "issue": {
                        "id": str(variables["id"]),
                        "identifier": "DAG-1",
                        "url": "https://linear.app/dagzoo/issue/DAG-1",
                    },
                }
            }
        raise AssertionError(f"Unexpected GraphQL query: {query}")


class _BootstrapLinear:
    def __init__(self) -> None:
        self.dry_run = True
        self.ensure_states_calls: list[tuple[str, dict[str, object]]] = []

    def get_project(self, project_slug: str) -> tuple[str, str, str]:
        _ = project_slug
        return ("project-1", "team-1", "DAG")

    def get_team_metadata(self, team_id: str) -> tuple[dict[str, object], dict[str, object]]:
        _ = team_id
        return ({}, {})

    def ensure_workflow_states(
        self,
        team_id: str,
        states: dict[str, object],
    ) -> dict[str, object]:
        self.ensure_states_calls.append((team_id, dict(states)))
        return {
            "Backlog": MODULE.LinearState(id="state-backlog", name="Backlog", type="backlog"),
            "Todo": MODULE.LinearState(id="state-todo", name="Todo", type="unstarted"),
        }

    def graphql(self, query: str, variables: dict[str, object]) -> dict[str, object]:
        _ = variables
        if "query ProjectIssues" in query:
            return {"issues": {"nodes": []}}
        raise AssertionError(f"Unexpected GraphQL query: {query}")


def test_build_ticket_specs_contains_epic_and_audit_ticket() -> None:
    specs = MODULE.build_ticket_specs()
    titles = [spec.title for spec in specs]
    assert MODULE.HARNESS_EPIC_TITLE in titles
    assert MODULE.RECURRING_AUDIT_TITLE in titles
    assert (
        "process(harness): codify issue authoring and acceptance-criteria standards for Symphony"
        not in titles
    )
    assert (
        "docs(harness): write a repo-wide harness-engineering guide for dagzoo contributors"
        in titles
    )
    assert len(titles) == len(set(titles))


def test_harness_guide_ticket_no_longer_claims_missing_audit_template() -> None:
    specs = MODULE.build_ticket_specs()
    guide = next(
        spec
        for spec in specs
        if spec.title
        == "docs(harness): write a repo-wide harness-engineering guide for dagzoo contributors"
    )

    assert "weekly audit template" not in guide.title.lower()
    assert "weekly audit template" not in guide.description.lower()
    assert "weekly audit checklist is versioned in-repo" not in guide.description


def test_audit_ticket_defaults_to_todo_and_references_docs() -> None:
    specs = MODULE.build_ticket_specs()
    audit = next(spec for spec in specs if spec.title == MODULE.RECURRING_AUDIT_TITLE)
    assert audit.state_name == "Todo"
    assert "Friday" in audit.description
    assert "10:00 PM `America/Los_Angeles`" in audit.description
    assert "docs/development/harness_audit.md" in audit.description


def test_required_label_specs_cover_every_ticket_label() -> None:
    labels_from_specs = {label for spec in MODULE.build_ticket_specs() for label in spec.labels}

    assert MODULE.required_label_specs() == {
        "audit": "#7C3AED",
        "documentation": "#0EA5E9",
        "epic": "#DC2626",
        "harness": "#2563EB",
    }
    assert set(MODULE.required_label_specs()) == labels_from_specs


def test_ensure_labels_provisions_every_required_spec_label_in_dry_run() -> None:
    linear = _RecordingLinear(dry_run=True)

    labels = MODULE.ensure_labels(
        linear,
        team_id="team-1",
        existing_labels={},
    )

    assert set(labels) == {"audit", "documentation", "epic", "harness"}
    assert labels["epic"].id == "dry-run-epic"
    assert labels["documentation"].id == "dry-run-documentation"


def test_create_or_update_issue_create_keeps_state_and_labels() -> None:
    linear = _RecordingLinear()

    MODULE.create_or_update_issue(
        linear,
        existing=None,
        title="Issue A",
        description="body",
        team_id="team-1",
        project_id="project-1",
        state=MODULE.LinearState(id="state-backlog", name="Backlog", type="backlog"),
        label_ids=["label-harness", "label-epic"],
        parent_id="parent-1",
    )

    query, variables = linear.calls[0]
    assert "issueCreate" in query
    input_payload = variables["input"]
    assert isinstance(input_payload, dict)
    assert input_payload["stateId"] == "state-backlog"
    assert input_payload["labelIds"] == ["label-harness", "label-epic"]
    assert input_payload["parentId"] == "parent-1"
    assert input_payload["teamId"] == "team-1"


def test_create_or_update_issue_rerun_preserves_existing_state_and_labels() -> None:
    linear = _RecordingLinear()

    MODULE.create_or_update_issue(
        linear,
        existing={
            "id": "issue-1",
            "identifier": "DAG-1",
            "url": "https://linear.app/dagzoo/issue/DAG-1",
        },
        title="Issue A",
        description="updated body",
        team_id="team-1",
        project_id="project-1",
        state=MODULE.LinearState(id="state-backlog", name="Backlog", type="backlog"),
        label_ids=["label-harness", "label-epic"],
        parent_id="parent-1",
    )

    query, variables = linear.calls[0]
    assert "issueUpdate" in query
    assert variables["id"] == "issue-1"
    input_payload = variables["input"]
    assert isinstance(input_payload, dict)
    assert input_payload == {
        "title": "Issue A",
        "description": "updated body",
        "projectId": "project-1",
        "parentId": "parent-1",
    }


def test_main_bootstraps_workflow_states_before_seeding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    linear = _BootstrapLinear()
    monkeypatch.setattr(MODULE, "load_linear_api_key", lambda _path: "test-key")
    monkeypatch.setattr(MODULE, "LinearClient", lambda *args, **kwargs: linear)

    exit_code = MODULE.main(
        [
            "--linear-api-key-file",
            str(tmp_path / "linear.key"),
            "--project-slug",
            "proj",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert linear.ensure_states_calls == [("team-1", {})]
    assert "AUDIT_ISSUE_URL=https://linear.app/fake" in captured.out
