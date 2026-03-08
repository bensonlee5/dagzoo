#!/usr/bin/env python3
"""Migrate dagzoo GitHub issues to Linear and optionally cut GitHub over."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


GH_MIGRATION_MARKER = "<!-- dagzoo-linear-migration -->"
DEFAULT_LINEAR_ENDPOINT = "https://api.linear.app/graphql"
SYMPHONY_STATES: tuple[tuple[str, str, str, float], ...] = (
    ("Backlog", "backlog", "#6B7280", 0.0),
    ("Todo", "unstarted", "#94A3B8", 1.0),
    ("In Progress", "started", "#3B82F6", 2.0),
    ("Human Review", "started", "#8B5CF6", 1003.0),
    ("Rework", "started", "#F59E0B", 1004.0),
    ("Merging", "started", "#06B6D4", 1005.0),
    ("Done", "completed", "#22C55E", 3.0),
)
STATE_BY_GH_STATE = {"OPEN": "Backlog", "CLOSED": "Done"}
PRIORITY_FROM_LABEL = {"P0": 1, "P1": 2, "P2": 3}
RESERVED_LABELS = frozenset(PRIORITY_FROM_LABEL)
REFERENCE_RE = re.compile(r"(?<![A-Za-z0-9])#(?P<number>[1-9][0-9]*)\b")
MIGRATED_GH_NUMBER_RE = re.compile(r"Original GitHub issue:\s+`#(?P<number>[1-9][0-9]*)`")


class MigrationError(RuntimeError):
    """Raised when migration prerequisites or remote operations fail."""


@dataclass(slots=True)
class GitHubLabel:
    name: str
    color: str | None = None
    description: str | None = None


@dataclass(slots=True)
class GitHubIssue:
    number: int
    title: str
    body: str
    state: str
    labels: list[GitHubLabel]
    created_at: str
    updated_at: str
    url: str
    closed_at: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GitHubIssue":
        return cls(
            number=int(payload["number"]),
            title=str(payload["title"]),
            body=str(payload.get("body") or ""),
            state=str(payload["state"]).upper(),
            labels=[
                GitHubLabel(
                    name=str(label["name"]),
                    color=str(label.get("color")) if label.get("color") else None,
                    description=str(label.get("description")) if label.get("description") else None,
                )
                for label in payload.get("labels", [])
            ],
            created_at=str(payload["createdAt"]),
            updated_at=str(payload["updatedAt"]),
            closed_at=str(payload.get("closedAt")) if payload.get("closedAt") else None,
            url=str(payload["url"]),
        )


@dataclass(slots=True)
class LinearState:
    id: str
    name: str
    type: str
    position: float | None = None


@dataclass(slots=True)
class LinearLabel:
    id: str
    name: str
    color: str | None = None


@dataclass(slots=True)
class LinearIssueRef:
    id: str
    identifier: str
    url: str


@dataclass(slots=True)
class MappingEntry:
    github_number: int
    github_url: str
    github_state: str
    linear_id: str
    linear_identifier: str
    linear_url: str
    linear_state: str
    parent_github_number: int | None = None
    cutover_applied: bool = False
    cutover_applied_at: str | None = None


@dataclass(slots=True)
class MappingFile:
    repo: str
    project_slug: str
    migrated_at: str
    entries: dict[int, MappingEntry] = field(default_factory=dict)

    def entry_for(self, github_number: int) -> MappingEntry | None:
        return self.entries.get(github_number)

    def upsert(self, entry: MappingEntry) -> None:
        self.entries[entry.github_number] = entry

    def to_json(self) -> str:
        payload = {
            "repo": self.repo,
            "project_slug": self.project_slug,
            "migrated_at": self.migrated_at,
            "entries": [asdict(self.entries[number]) for number in sorted(self.entries)],
        }
        return json.dumps(payload, indent=2, sort_keys=False) + "\n"

    @classmethod
    def load(cls, path: Path, *, repo: str, project_slug: str) -> "MappingFile":
        if not path.exists():
            return cls(repo=repo, project_slug=project_slug, migrated_at=iso_now())
        payload = json.loads(path.read_text())
        payload_repo = str(payload.get("repo") or "").strip()
        payload_project_slug = str(payload.get("project_slug") or "").strip()
        if payload_repo and payload_repo != repo:
            raise MigrationError(
                f"Mapping file {path} targets repo {payload_repo!r}, expected {repo!r}."
            )
        if payload_project_slug and payload_project_slug != project_slug:
            raise MigrationError(
                "Mapping file "
                f"{path} targets project slug {payload_project_slug!r}, "
                f"expected {project_slug!r}."
            )
        entries = {
            int(entry["github_number"]): MappingEntry(**entry)
            for entry in payload.get("entries", [])
        }
        return cls(
            repo=payload_repo or repo,
            project_slug=payload_project_slug or project_slug,
            migrated_at=str(payload.get("migrated_at") or iso_now()),
            entries=entries,
        )


def iso_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def parse_issue_references(text: str) -> list[int]:
    seen: set[int] = set()
    refs: list[int] = []
    for match in REFERENCE_RE.finditer(text):
        number = int(match.group("number"))
        if number not in seen:
            seen.add(number)
            refs.append(number)
    return refs


def build_migrated_description(issue: GitHubIssue) -> str:
    labels = ", ".join(label.name for label in issue.labels) or "_none_"
    body = issue.body.strip() or "_No description provided in GitHub._"
    lines = [
        "## GitHub Migration Metadata",
        "",
        f"- Original GitHub issue: `#{issue.number}`",
        f"- GitHub URL: {issue.url}",
        f"- GitHub state: `{issue.state}`",
        f"- GitHub labels: {labels}",
        f"- GitHub created at: `{issue.created_at}`",
        f"- GitHub updated at: `{issue.updated_at}`",
    ]
    if issue.closed_at:
        lines.append(f"- GitHub closed at: `{issue.closed_at}`")
    lines.extend(("", "---", "", body))
    return "\n".join(lines)


def desired_linear_state_name(issue: GitHubIssue) -> str:
    try:
        return STATE_BY_GH_STATE[issue.state]
    except KeyError as exc:
        raise MigrationError(f"Unsupported GitHub issue state: {issue.state}") from exc


def desired_linear_priority(issue: GitHubIssue) -> int | None:
    for label in issue.labels:
        if label.name in PRIORITY_FROM_LABEL:
            return PRIORITY_FROM_LABEL[label.name]
    return None


def linear_label_specs(issues: list[GitHubIssue]) -> dict[str, dict[str, str | None]]:
    specs: dict[str, dict[str, str | None]] = {}
    for issue in issues:
        for label in issue.labels:
            if label.name in RESERVED_LABELS:
                continue
            specs.setdefault(
                label.name,
                {"color": normalize_color(label.color), "description": label.description},
            )
    return specs


def normalize_color(color: str | None) -> str | None:
    if not color:
        return None
    return color if color.startswith("#") else f"#{color}"


def epic_parent_map(issues: list[GitHubIssue]) -> dict[int, int]:
    issue_numbers = {issue.number for issue in issues}
    epic_numbers = {
        issue.number for issue in issues if any(label.name == "epic" for label in issue.labels)
    }
    parent_map: dict[int, int] = {}
    for issue in issues:
        if not any(label.name == "epic" for label in issue.labels):
            continue
        for ref in parse_issue_references(issue.body):
            if ref == issue.number or ref not in issue_numbers or ref in epic_numbers:
                continue
            parent_map.setdefault(ref, issue.number)
    return parent_map


def load_linear_api_key(path: Path) -> str:
    value = path.read_text().strip()
    if not value:
        raise MigrationError(f"Linear API key file is empty: {path}")
    return value


class GHCLI:
    def __init__(self, repo: str, *, dry_run: bool) -> None:
        self.repo = repo
        self.dry_run = dry_run

    def list_issues(self, issue_numbers: set[int] | None = None) -> list[GitHubIssue]:
        args = [
            "gh",
            "issue",
            "list",
            "--repo",
            self.repo,
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "number,title,body,state,labels,createdAt,updatedAt,closedAt,url",
        ]
        payload = run_json_command(args)
        issues = [GitHubIssue.from_payload(item) for item in payload]
        if issue_numbers:
            issues = [issue for issue in issues if issue.number in issue_numbers]
        issues.sort(key=lambda issue: issue.number)
        return issues

    def create_migration_comment(self, issue_number: int, linear_ref: LinearIssueRef) -> None:
        body = "\n".join(
            [
                GH_MIGRATION_MARKER,
                "Linear is now the source of truth for this issue.",
                "",
                f"- Linear successor: `{linear_ref.identifier}`",
                f"- Linear URL: {linear_ref.url}",
            ]
        )
        if self.dry_run:
            print(
                f"[dry-run] Would comment on GitHub issue #{issue_number}: {linear_ref.identifier}"
            )
            return
        run_command(
            [
                "gh",
                "api",
                f"repos/{self.repo}/issues/{issue_number}/comments",
                "-f",
                f"body={body}",
            ]
        )

    def close_issue(self, issue_number: int) -> None:
        if self.dry_run:
            print(f"[dry-run] Would close GitHub issue #{issue_number}")
            return
        run_command(
            [
                "gh",
                "api",
                "--method",
                "PATCH",
                f"repos/{self.repo}/issues/{issue_number}",
                "-f",
                "state=closed",
            ]
        )


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=True,
    )


def run_json_command(args: list[str]) -> Any:
    proc = run_command(args)
    return json.loads(proc.stdout)


class LinearClient:
    def __init__(
        self, api_key: str, *, endpoint: str = DEFAULT_LINEAR_ENDPOINT, dry_run: bool = False
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.dry_run = dry_run

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if (
            self.dry_run
            and "__type" not in query
            and "query" not in query.lstrip().split(maxsplit=1)[0]
        ):
            raise MigrationError("Dry-run mode cannot execute Linear mutations.")
        payload = json.dumps({"query": query, "variables": variables or {}}).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            raise MigrationError(f"Linear API error {exc.code}: {body}") from exc
        if data.get("errors"):
            raise MigrationError(f"Linear GraphQL error: {data['errors']}")
        return data["data"]

    def get_project(self, project_slug: str) -> tuple[str, str, str]:
        data = self.graphql(
            """
            query ProjectBySlug($slug: String!) {
              projects(filter: { slugId: { eq: $slug } }) {
                nodes {
                  id
                  name
                  teams(first: 5) {
                    nodes {
                      id
                      key
                      name
                    }
                  }
                }
              }
            }
            """,
            {"slug": project_slug},
        )
        nodes = data["projects"]["nodes"]
        if not nodes:
            raise MigrationError(f"No Linear project found for slug {project_slug}")
        project = nodes[0]
        teams = project["teams"]["nodes"]
        if not teams:
            raise MigrationError(f"Linear project {project_slug} has no owning team")
        team = teams[0]
        return str(project["id"]), str(team["id"]), str(team["key"])

    def get_team_metadata(
        self, team_id: str
    ) -> tuple[dict[str, LinearState], dict[str, LinearLabel]]:
        data = self.graphql(
            """
            query TeamMetadata($teamId: String!) {
              team(id: $teamId) {
                states(first: 50) {
                  nodes {
                    id
                    name
                    type
                    position
                  }
                }
                labels(first: 100) {
                  nodes {
                    id
                    name
                    color
                  }
                }
              }
            }
            """,
            {"teamId": team_id},
        )
        team = data["team"]
        states = {
            node["name"]: LinearState(
                id=str(node["id"]),
                name=str(node["name"]),
                type=str(node["type"]),
                position=float(node["position"]) if node.get("position") is not None else None,
            )
            for node in team["states"]["nodes"]
        }
        labels = {
            node["name"]: LinearLabel(
                id=str(node["id"]),
                name=str(node["name"]),
                color=str(node.get("color")) if node.get("color") else None,
            )
            for node in team["labels"]["nodes"]
        }
        return states, labels

    def existing_project_issue_map(self, project_id: str) -> dict[int, LinearIssueRef]:
        data = self.graphql(
            """
            query ProjectIssues($projectId: ID!) {
              issues(filter: { project: { id: { eq: $projectId } } }, first: 250) {
                nodes {
                  id
                  identifier
                  url
                  description
                }
              }
            }
            """,
            {"projectId": project_id},
        )
        mapping: dict[int, LinearIssueRef] = {}
        for node in data["issues"]["nodes"]:
            description = str(node.get("description") or "")
            match = MIGRATED_GH_NUMBER_RE.search(description)
            if not match:
                continue
            github_number = int(match.group("number"))
            mapping[github_number] = LinearIssueRef(
                id=str(node["id"]),
                identifier=str(node["identifier"]),
                url=str(node["url"]),
            )
        return mapping

    def ensure_workflow_states(
        self, team_id: str, existing_states: dict[str, LinearState]
    ) -> dict[str, LinearState]:
        states = dict(existing_states)
        for name, state_type, color, position in SYMPHONY_STATES:
            if name in states:
                continue
            if self.dry_run:
                print(f"[dry-run] Would create Linear workflow state {name!r}")
                states[name] = LinearState(
                    id=f"dry-run-{name}", name=name, type=state_type, position=position
                )
                continue
            data = self.graphql(
                """
                mutation CreateWorkflowState($input: WorkflowStateCreateInput!) {
                  workflowStateCreate(input: $input) {
                    success
                    workflowState {
                      id
                      name
                      type
                      position
                    }
                  }
                }
                """,
                {
                    "input": {
                        "teamId": team_id,
                        "name": name,
                        "type": state_type,
                        "color": color,
                        "position": position,
                        "description": f"Created for Symphony orchestration in dagzoo ({name})",
                    }
                },
            )
            payload = data["workflowStateCreate"]
            if not payload.get("success"):
                raise MigrationError(f"Failed to create workflow state {name!r}")
            node = payload["workflowState"]
            states[name] = LinearState(
                id=str(node["id"]),
                name=str(node["name"]),
                type=str(node["type"]),
                position=float(node["position"]) if node.get("position") is not None else None,
            )
        return states

    def ensure_labels(
        self,
        team_id: str,
        existing_labels: dict[str, LinearLabel],
        label_specs: dict[str, dict[str, str | None]],
    ) -> dict[str, LinearLabel]:
        labels = dict(existing_labels)
        for name, spec in sorted(label_specs.items()):
            if name in labels:
                continue
            if self.dry_run:
                print(f"[dry-run] Would create Linear label {name!r}")
                labels[name] = LinearLabel(id=f"dry-run-{name}", name=name, color=spec["color"])
                continue
            data = self.graphql(
                """
                mutation CreateIssueLabel($input: IssueLabelCreateInput!) {
                  issueLabelCreate(input: $input) {
                    success
                    issueLabel {
                      id
                      name
                      color
                    }
                  }
                }
                """,
                {
                    "input": {
                        "teamId": team_id,
                        "name": name,
                        "color": spec["color"],
                        "description": spec["description"],
                    }
                },
            )
            payload = data["issueLabelCreate"]
            if not payload.get("success"):
                raise MigrationError(f"Failed to create label {name!r}")
            node = payload["issueLabel"]
            labels[name] = LinearLabel(
                id=str(node["id"]),
                name=str(node["name"]),
                color=str(node.get("color")) if node.get("color") else None,
            )
        return labels

    def create_issue(self, input_payload: dict[str, Any]) -> LinearIssueRef:
        if self.dry_run:
            identifier = f"DRY-{input_payload['title'][:12].upper()}"
            print(f"[dry-run] Would create Linear issue {input_payload['title']!r}")
            return LinearIssueRef(
                id=f"dry-run-{identifier}",
                identifier=identifier,
                url=f"https://linear.app/fake/issue/{identifier}",
            )
        data = self.graphql(
            """
            mutation CreateIssue($input: IssueCreateInput!) {
              issueCreate(input: $input) {
                success
                issue {
                  id
                  identifier
                  url
                }
              }
            }
            """,
            {"input": input_payload},
        )
        payload = data["issueCreate"]
        if not payload.get("success"):
            raise MigrationError(f"Failed to create Linear issue {input_payload['title']!r}")
        issue = payload["issue"]
        return LinearIssueRef(
            id=str(issue["id"]), identifier=str(issue["identifier"]), url=str(issue["url"])
        )

    def update_issue(self, issue_id: str, input_payload: dict[str, Any]) -> LinearIssueRef:
        if self.dry_run:
            print(f"[dry-run] Would update Linear issue {issue_id}")
            return LinearIssueRef(
                id=issue_id, identifier=issue_id, url=f"https://linear.app/fake/issue/{issue_id}"
            )
        data = self.graphql(
            """
            mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
              issueUpdate(id: $id, input: $input) {
                success
                issue {
                  id
                  identifier
                  url
                }
              }
            }
            """,
            {"id": issue_id, "input": input_payload},
        )
        payload = data["issueUpdate"]
        if not payload.get("success"):
            raise MigrationError(f"Failed to update Linear issue {issue_id}")
        issue = payload["issue"]
        return LinearIssueRef(
            id=str(issue["id"]), identifier=str(issue["identifier"]), url=str(issue["url"])
        )


def build_issue_input(
    issue: GitHubIssue,
    *,
    team_id: str,
    project_id: str,
    state_id: str | None,
    label_ids: list[str],
    parent_id: str | None,
    include_historical_timestamps: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "teamId": team_id,
        "projectId": project_id,
        "title": issue.title,
        "description": build_migrated_description(issue),
    }
    if state_id is not None:
        payload["stateId"] = state_id
    if label_ids:
        payload["labelIds"] = label_ids
    priority = desired_linear_priority(issue)
    if priority is not None:
        payload["priority"] = priority
    if parent_id:
        payload["parentId"] = parent_id
    if include_historical_timestamps:
        payload["createdAt"] = issue.created_at
    if include_historical_timestamps and issue.closed_at:
        payload["completedAt"] = issue.closed_at
    return payload


def migrate_issues(
    *,
    gh: GHCLI,
    linear: LinearClient,
    repo: str,
    project_slug: str,
    mapping_path: Path,
    issue_numbers: set[int] | None,
) -> MappingFile:
    all_issues = gh.list_issues()
    issues = (
        [issue for issue in all_issues if issue.number in issue_numbers]
        if issue_numbers is not None
        else list(all_issues)
    )
    if not issues:
        raise MigrationError("No GitHub issues matched the requested scope.")

    mapping = MappingFile.load(mapping_path, repo=repo, project_slug=project_slug)
    mapping.migrated_at = iso_now()
    persist_mapping = not gh.dry_run
    project_id, team_id, _team_key = linear.get_project(project_slug)
    states, labels = linear.get_team_metadata(team_id)
    states = linear.ensure_workflow_states(team_id, states)
    labels = linear.ensure_labels(team_id, labels, linear_label_specs(issues))
    parent_map = epic_parent_map(all_issues)
    recovered_refs = linear.existing_project_issue_map(project_id)

    linear_refs_by_github_number: dict[int, LinearIssueRef] = {}
    for issue in all_issues:
        entry = mapping.entry_for(issue.number)
        if entry:
            linear_refs_by_github_number[issue.number] = LinearIssueRef(
                id=entry.linear_id,
                identifier=entry.linear_identifier,
                url=entry.linear_url,
            )
            continue
        recovered_ref = recovered_refs.get(issue.number)
        if recovered_ref:
            linear_refs_by_github_number[issue.number] = recovered_ref

    for issue in issues:
        entry = mapping.entry_for(issue.number)
        recovered_ref = linear_refs_by_github_number.get(issue.number)
        parent_github_number = parent_map.get(issue.number)
        parent_id = None
        if (
            parent_github_number is not None
            and parent_github_number in linear_refs_by_github_number
        ):
            parent_id = linear_refs_by_github_number[parent_github_number].id
        if entry and entry.cutover_applied:
            linear_ref = LinearIssueRef(
                id=entry.linear_id,
                identifier=entry.linear_identifier,
                url=entry.linear_url,
            )
            state_name = entry.linear_state
        else:
            label_ids = [labels[label.name].id for label in issue.labels if label.name in labels]
            state_name = desired_linear_state_name(issue)
            state = states.get(state_name)
            if state is None:
                raise MigrationError(f"Missing Linear workflow state {state_name!r}")
            payload = build_issue_input(
                issue,
                team_id=team_id,
                project_id=project_id,
                state_id=state.id,
                label_ids=label_ids,
                parent_id=parent_id,
                include_historical_timestamps=entry is None and recovered_ref is None,
            )
            if entry is None and recovered_ref is None:
                linear_ref = linear.create_issue(payload)
            else:
                issue_id = entry.linear_id if entry is not None else recovered_ref.id
                linear_ref = linear.update_issue(issue_id, payload)
        linear_refs_by_github_number[issue.number] = linear_ref
        mapping.upsert(
            MappingEntry(
                github_number=issue.number,
                github_url=issue.url,
                github_state=issue.state,
                linear_id=linear_ref.id,
                linear_identifier=linear_ref.identifier,
                linear_url=linear_ref.url,
                linear_state=state_name,
                parent_github_number=parent_github_number,
                cutover_applied=entry.cutover_applied if entry else False,
                cutover_applied_at=entry.cutover_applied_at if entry else None,
            )
        )

    if persist_mapping:
        mapping_path.write_text(mapping.to_json())

    selected_issue_numbers = {issue.number for issue in issues}
    for child_number, parent_number in parent_map.items():
        child_entry = mapping.entry_for(child_number)
        parent_entry = mapping.entry_for(parent_number)
        if not child_entry or not parent_entry:
            continue
        if (
            issue_numbers is not None
            and child_number not in selected_issue_numbers
            and parent_number not in selected_issue_numbers
        ):
            continue
        linear.update_issue(child_entry.linear_id, {"parentId": parent_entry.linear_id})
        child_entry.parent_github_number = parent_number
        mapping.upsert(child_entry)

    if persist_mapping:
        mapping_path.write_text(mapping.to_json())
    return mapping


def apply_cutover(
    *,
    gh: GHCLI,
    mapping: MappingFile,
    issue_numbers: set[int] | None,
    mapping_path: Path | None = None,
    persist_mapping: bool = False,
) -> None:
    def checkpoint_mapping() -> None:
        if not persist_mapping:
            return
        if mapping_path is None:
            raise MigrationError("Cutover checkpointing requires a mapping path.")
        mapping_path.write_text(mapping.to_json())

    numbers = sorted(mapping.entries)
    for number in numbers:
        if issue_numbers and number not in issue_numbers:
            continue
        entry = mapping.entries[number]
        if entry.cutover_applied:
            continue
        linear_ref = LinearIssueRef(
            id=entry.linear_id,
            identifier=entry.linear_identifier,
            url=entry.linear_url,
        )
        if entry.cutover_applied_at is None:
            gh.create_migration_comment(number, linear_ref)
            entry.cutover_applied_at = iso_now()
            checkpoint_mapping()
        if entry.github_state == "OPEN":
            gh.close_issue(number)
            entry.github_state = "CLOSED"
        entry.cutover_applied = True
        entry.cutover_applied_at = iso_now()
        checkpoint_mapping()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repo in owner/name form")
    parser.add_argument("--linear-api-key-file", required=True, type=Path)
    parser.add_argument("--project-slug", required=True, help="Linear project slug ID")
    parser.add_argument("--mapping-path", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("migrate", "cutover", "all"),
        default="migrate",
        help="Operation mode",
    )
    parser.add_argument(
        "--issue-number",
        dest="issue_numbers",
        type=int,
        action="append",
        help="Restrict to one or more GitHub issue numbers",
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_LINEAR_ENDPOINT, help="Linear GraphQL endpoint"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print intended changes without writes"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    mapping_path = args.mapping_path
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    linear_api_key = load_linear_api_key(args.linear_api_key_file)
    issue_numbers = set(args.issue_numbers or [])
    gh = GHCLI(args.repo, dry_run=args.dry_run)
    linear = LinearClient(linear_api_key, endpoint=args.endpoint, dry_run=args.dry_run)

    mapping = MappingFile.load(mapping_path, repo=args.repo, project_slug=args.project_slug)
    if args.mode in {"migrate", "all"}:
        mapping = migrate_issues(
            gh=gh,
            linear=linear,
            repo=args.repo,
            project_slug=args.project_slug,
            mapping_path=mapping_path,
            issue_numbers=issue_numbers or None,
        )
    if args.mode in {"cutover", "all"}:
        if not mapping.entries:
            raise MigrationError("Cutover requires a populated mapping file.")
        apply_cutover(
            gh=gh,
            mapping=mapping,
            issue_numbers=issue_numbers or None,
            mapping_path=mapping_path,
            persist_mapping=not args.dry_run,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
