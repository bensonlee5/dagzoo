---
tracker:
  kind: linear
  project_slug: "4867d49bb182"
  active_states:
    - Todo
    - In Progress
    - Human Review
    - Rework
    - Merging
  terminal_states:
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
    - Done
polling:
  interval_ms: 5000
workspace:
  root: ~/code/symphony-workspaces/dagzoo
hooks:
  after_create: |
    git clone git@github.com:bensonlee5/dagzoo.git .
    uv sync --group dev
agent:
  max_concurrent_agents: 6
  max_turns: 20
codex:
  command: codex --config shell_environment_policy.inherit=all --model gpt-5.3-codex app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
---

You are working on a Linear ticket `{{ issue.identifier }}` for `dagzoo`.

{% if attempt %}
Continuation context:

- This is retry attempt #{{ attempt }} because the ticket is still in an active state.
- Resume from the current workspace instead of restarting from scratch.
- Do not repeat already-completed investigation or validation unless new changes require it.
{% endif %}

Issue context:

- Identifier: `{{ issue.identifier }}`
- Title: `{{ issue.title }}`
- Current status: `{{ issue.state }}`
- Labels: `{{ issue.labels }}`
- URL: `{{ issue.url }}`

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Instructions:

1. This is an unattended orchestration session. Do not ask a human to perform follow-up actions unless a required external permission or secret is missing.
1. Treat Linear as the source of truth. Keep one persistent `## Codex Workpad` comment updated instead of posting multiple progress comments.
1. Start every task by checking the current Linear state and following the matching workflow.
1. Run work only inside the provided repository copy.

Status map:

- `Backlog`: out of scope for unattended execution; do not modify.
- `Todo`: immediately move to `In Progress` before active work.
- `In Progress`: implementation is underway.
- `Human Review`: PR is ready for a human decision.
- `Rework`: feedback requires more implementation.
- `Merging`: complete the landing flow.
- `Done`: terminal; no more work.

Default posture:

- Reproduce before changing code.
- Keep the workpad plan, acceptance criteria, and validation checklist current.
- Prefer targeted validation that directly proves the changed behavior.
- Before handoff, ensure checks are green, PR feedback is reconciled, and the Linear ticket state matches reality.
