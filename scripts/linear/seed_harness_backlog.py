#!/usr/bin/env python3
"""Seed recurring harness-audit tracker objects in Linear."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from github_to_linear import (
    DEFAULT_LINEAR_ENDPOINT,
    LinearClient,
    LinearLabel,
    LinearState,
    MigrationError,
    load_linear_api_key,
)


HARNESS_ARTICLE_URL = "https://openai.com/index/harness-engineering/"
RECURRING_AUDIT_TITLE = "ops(harness): weekly full-repo harness audit"
HARNESS_EPIC_TITLE = "epic: harness engineering adoption for autonomous dagzoo development"
LABEL_COLORS = {
    "audit": "#7C3AED",
    "documentation": "#0EA5E9",
    "epic": "#DC2626",
    "harness": "#2563EB",
}


@dataclass(frozen=True, slots=True)
class TicketSpec:
    title: str
    state_name: str
    labels: tuple[str, ...]
    description: str
    parent_title: str | None = None


def _markdown_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_ticket_specs() -> list[TicketSpec]:
    epic_description = (
        "## Summary\n\n"
        "Adopt Harness Engineering best practices for `dagzoo` so autonomous agents can work "
        "against a clearer repo contract, more enforceable architecture, and lower-entropy "
        "tooling/process surfaces.\n\n"
        "Primary reference:\n\n"
        f"- {HARNESS_ARTICLE_URL}\n\n"
        "## Themes\n\n"
        + _markdown_list(
            [
                "repo as system of record",
                "agent legibility",
                "agent-first interfaces",
                "architecture and taste enforcement",
                "entropy and garbage collection",
                "safe increases in autonomy",
            ]
        )
        + "\n\n## Acceptance Criteria\n\n"
        + _markdown_list(
            [
                "Child tickets exist for the concrete harness gaps already visible in the repo.",
                "The recurring weekly audit is configured and points at the repo-owned rubric.",
                "The adoption work is tracked separately from feature roadmap epics.",
            ]
        )
    )

    child_specs = [
        TicketSpec(
            title="docs(harness): expand AGENTS.md into a complete agent operating contract",
            state_name="Backlog",
            labels=("harness", "documentation"),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The current `AGENTS.md` is intentionally short and does not yet provide the level "
                "of repo legibility described in Harness Engineering.\n\n"
                "## Goal\n\n"
                "Turn `AGENTS.md` into a high-signal contract for autonomous contributors.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Canonical bootstrap, test, lint, and release commands are documented.",
                        "Public vs internal surfaces are called out where they matter.",
                        "Issue/PR expectations and user-facing break policy are explicit.",
                        "The document is specific to dagzoo rather than generic agent advice.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="ci(harness): add a pull request template and handoff evidence requirements",
            state_name="Backlog",
            labels=("harness",),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo has review expectations, but no checked-in PR template or standard handoff evidence surface.\n\n"
                "## Goal\n\n"
                "Force consistent intent/risk/validation reporting in PRs.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "A repo PR template exists.",
                        "The template requires change summary, validation evidence, and user-facing break callouts.",
                        "The template aligns with the existing `/review` expectation in `AGENTS.md`.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="ci(architecture): add structural dependency checks around src/dagzoo/core",
            state_name="Backlog",
            labels=("harness",),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo documents a preferred `src/dagzoo/core`-centric dependency direction, but there is no "
                "automated enforcement against drift.\n\n"
                "## Goal\n\n"
                "Add structural checks that keep architecture intent enforceable.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "A repeatable check exists for dependency-direction drift or obvious cycles.",
                        "The enforced rules match the repo's stated architecture conventions.",
                        "Violations fail locally and in CI with actionable output.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="ci(docs): enforce system-of-record docs updates for user-facing changes",
            state_name="Backlog",
            labels=("harness", "documentation"),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo has policy around user-facing breaks, but no explicit guardrail that required docs updates "
                "happen alongside CLI/output-contract changes.\n\n"
                "## Goal\n\n"
                "Add a docs/system-of-record guardrail for user-facing changes.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Changes to CLI flags, persisted metadata schema, or artifact contracts trigger docs expectations.",
                        "Version-bump/changelog policy is checked where applicable.",
                        "The guardrail is narrow enough to avoid excessive false positives.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="tooling(harness): add one-command doctor and verify entrypoints for agents",
            state_name="Backlog",
            labels=("harness",),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "Canonical commands are spread across CI configs, docs, and scripts. There is no minimal doctor/verify "
                "surface an unfamiliar agent can trust immediately.\n\n"
                "## Goal\n\n"
                "Provide a canonical bootstrap/verification entrypoint pair.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "One command verifies environment sanity and required tools.",
                        "One command runs the standard local validation path.",
                        "Docs and CI reference the same canonical surfaces.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="docs(harness): write a repo-wide harness-engineering guide for dagzoo contributors",
            state_name="Backlog",
            labels=("harness", "documentation"),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo has a weekly audit rubric, but it still lacks one broader dagzoo-specific guide that "
                "translates Harness Engineering into contributor-facing standards across docs, process, and tooling.\n\n"
                "## Goal\n\n"
                "Create the repo-owned harness-engineering guide for dagzoo contributors.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "The guide references the Harness Engineering article directly.",
                        "The guide explains dagzoo-specific expectations beyond the weekly audit rubric.",
                        "The guidance connects the repo contract, workflow, and verification surfaces in one place.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="cleanup(harness): eliminate stale generated-output/doc drift and codify garbage collection",
            state_name="Backlog",
            labels=("harness",),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo already documents at least one stale-output hazard (`public/` versus `site/public`) and does "
                "not yet have a fully explicit garbage-collection policy for stale generated surfaces.\n\n"
                "## Goal\n\n"
                "Reduce entropy by making stale outputs and cleanup rules explicit and enforceable.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Known stale-output/documentation drift is resolved or clearly fenced.",
                        "Cleanup rules for generated/local artifacts are documented.",
                        "Weekly audit checks can verify this area deterministically.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
    ]

    audit_description = (
        "## Summary\n\n"
        "Run the full weekly harness audit for `dagzoo`, using the repo-owned rubric to detect drift, "
        "create remediation issues, and keep the repo legible for unattended agent work.\n\n"
        "Primary reference:\n\n"
        f"- {HARNESS_ARTICLE_URL}\n"
        "- `docs/development/harness_audit.md`\n"
        "- `docs/development/issue_authoring.md`\n\n"
        "## Schedule\n\n"
        "- Weekly\n"
        "- Friday\n"
        "- 10:00 PM `America/Los_Angeles`\n\n"
        "## Required Flow\n\n"
        "1. Audit the repo using the checklist in `docs/development/harness_audit.md`.\n"
        "2. Reuse existing open issues when the same gap is already tracked.\n"
        "3. Create one remediation issue per net-new actionable gap.\n"
        "4. Put remediation issues in `Backlog` with label `harness`.\n"
        "5. Link remediation issues from this audit ticket.\n"
        "6. Close only after a concise summary is written.\n\n"
        "## Completion Rule\n\n"
        "This ticket is complete only when it records either:\n\n"
        "- linked remediation issues for every net-new actionable gap, or\n"
        "- a short explicit note that no action was required this week.\n"
    )

    return [
        TicketSpec(
            title=HARNESS_EPIC_TITLE,
            state_name="Backlog",
            labels=("epic", "harness"),
            description=epic_description,
        ),
        *child_specs,
        TicketSpec(
            title=RECURRING_AUDIT_TITLE,
            state_name="Todo",
            labels=("audit", "harness"),
            description=audit_description,
        ),
    ]


def required_label_specs() -> dict[str, str]:
    names = sorted({label for spec in build_ticket_specs() for label in spec.labels})
    missing = [name for name in names if name not in LABEL_COLORS]
    if missing:
        raise MigrationError(f"Missing configured colors for labels: {', '.join(missing)}")
    return {name: LABEL_COLORS[name] for name in names}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-api-key-file", required=True, type=Path)
    parser.add_argument("--project-slug", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_LINEAR_ENDPOINT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def find_existing_project_issues(
    linear: LinearClient, project_id: str
) -> dict[str, dict[str, Any]]:
    data = linear.graphql(
        """
        query ProjectIssues($projectId: ID!) {
          issues(filter: { project: { id: { eq: $projectId } } }, first: 200) {
            nodes {
              id
              identifier
              url
              title
            }
          }
        }
        """,
        {"projectId": project_id},
    )
    return {node["title"]: node for node in data["issues"]["nodes"]}


def ensure_labels(
    linear: LinearClient,
    *,
    team_id: str,
    existing_labels: dict[str, LinearLabel],
) -> dict[str, LinearLabel]:
    labels = dict(existing_labels)
    for name, color in required_label_specs().items():
        if name in labels:
            continue
        if linear.dry_run:
            print(f"[dry-run] Would create label {name!r}")
            labels[name] = LinearLabel(id=f"dry-run-{name}", name=name, color=color)
            continue
        data = linear.graphql(
            """
            mutation CreateLabel($input: IssueLabelCreateInput!) {
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
            {"input": {"teamId": team_id, "name": name, "color": color}},
        )
        payload = data["issueLabelCreate"]
        if not payload.get("success"):
            raise MigrationError(f"Failed to create label {name!r}")
        node = payload["issueLabel"]
        labels[name] = LinearLabel(id=node["id"], name=node["name"], color=node.get("color"))
    return labels


def create_or_update_issue(
    linear: LinearClient,
    *,
    existing: dict[str, Any] | None,
    title: str,
    description: str,
    team_id: str,
    project_id: str,
    state: LinearState,
    label_ids: list[str],
    parent_id: str | None,
) -> dict[str, str]:
    create_payload: dict[str, Any] = {
        "title": title,
        "description": description,
        "teamId": team_id,
        "projectId": project_id,
        "stateId": state.id,
        "labelIds": label_ids,
    }
    if parent_id:
        create_payload["parentId"] = parent_id

    if existing is None:
        if linear.dry_run:
            print(f"[dry-run] Would create issue {title!r}")
            return {
                "id": f"dry-run-{title}",
                "identifier": "DRY-1",
                "url": "https://linear.app/fake",
            }
        data = linear.graphql(
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
            {"input": create_payload},
        )
        response = data["issueCreate"]
        if not response.get("success"):
            raise MigrationError(f"Failed to create issue {title!r}")
        return response["issue"]

    update_payload: dict[str, Any] = {
        "title": title,
        "description": description,
        "projectId": project_id,
    }
    if parent_id:
        update_payload["parentId"] = parent_id

    if linear.dry_run:
        print(f"[dry-run] Would update issue {title!r} ({existing['identifier']})")
        return {"id": existing["id"], "identifier": existing["identifier"], "url": existing["url"]}
    data = linear.graphql(
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
        {"id": existing["id"], "input": update_payload},
    )
    response = data["issueUpdate"]
    if not response.get("success"):
        raise MigrationError(f"Failed to update issue {title!r}")
    return response["issue"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = load_linear_api_key(args.linear_api_key_file)
    linear = LinearClient(api_key, endpoint=args.endpoint, dry_run=args.dry_run)
    project_id, team_id, _team_key = linear.get_project(args.project_slug)
    states, labels = linear.get_team_metadata(team_id)
    states = linear.ensure_workflow_states(team_id, states)
    labels = ensure_labels(linear, team_id=team_id, existing_labels=labels)
    existing_issues = find_existing_project_issues(linear, project_id)

    created: dict[str, dict[str, str]] = {}
    for spec in build_ticket_specs():
        state = states.get(spec.state_name)
        if state is None:
            raise MigrationError(f"Missing Linear state {spec.state_name!r}")
        parent_id = None
        if spec.parent_title:
            parent = created.get(spec.parent_title) or existing_issues.get(spec.parent_title)
            if not parent:
                raise MigrationError(f"Parent issue missing: {spec.parent_title}")
            parent_id = parent["id"]
        label_ids = [labels[name].id for name in spec.labels]
        existing = existing_issues.get(spec.title)
        issue = create_or_update_issue(
            linear,
            existing=existing,
            title=spec.title,
            description=spec.description,
            team_id=team_id,
            project_id=project_id,
            state=state,
            label_ids=label_ids,
            parent_id=parent_id,
        )
        created[spec.title] = issue

    audit = created[RECURRING_AUDIT_TITLE]
    print(f"AUDIT_ISSUE_URL={audit['url']}")
    print(
        "NOTE=Configure this issue as a native recurring Linear issue: weekly, Friday, 10:00 PM America/Los_Angeles."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
