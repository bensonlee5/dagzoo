# Symphony and Linear

`dagzoo` uses Linear as the live issue tracker and can be driven by Symphony through the
repo-owned [`WORKFLOW.md`](../../WORKFLOW.md) contract.

## What lives in this repo

- `WORKFLOW.md`: tracker/runtime contract for Symphony.
- `scripts/linear/github_to_linear.py`: one-shot GitHub Issues -> Linear migration and GitHub cutover tool.

The Symphony runtime itself is not vendored here. Use the external
`openai/symphony/elixir` implementation and point it at this repo's `WORKFLOW.md`.

## Required environment

- `LINEAR_API_KEY`: Personal API key with access to the target Linear project.
- `uv`: used to bootstrap repo dependencies in per-issue workspaces.
- `codex`: available on `PATH` for Symphony to launch in app-server mode.

Optional:

- `SYMPHONY_WORKSPACE_ROOT`: override the workspace root instead of the default in `WORKFLOW.md`.
- `CODEX_BIN`: alternate Codex binary path when the runtime wrapper needs one.

## Target tracker

- Linear project URL: `https://linear.app/bl-personal/project/dagzoo-4867d49bb182/overview`
- Linear project slug ID: `4867d49bb182`
- Linear team key: `BL`

Symphony-native status model for this repo:

- `Backlog`
- `Todo`
- `In Progress`
- `Human Review`
- `Rework`
- `Merging`
- `Done`

The migration tool bootstraps any missing workflow states on the owning team before importing issues.

## Weekly Harness Audit

The repo-owned weekly audit rubric lives at
[`docs/development/harness_audit.md`](harness_audit.md).

Default recurring audit contract:

- Linear issue title: `ops(harness): weekly full-repo harness audit`
- Schedule: Friday, 10:00 PM `America/Los_Angeles`
- Creation state: `Todo`
- Remediation issues: `Backlog`, label `harness`

Use `scripts/linear/seed_harness_backlog.py` to seed the harness epic, child
tickets, and the weekly audit issue body. If Linear recurrence is not available
through the API, configure the final recurrence in the Linear UI after seeding.

## Running Symphony

Clone the external Symphony repo and run the Elixir reference implementation against this repository's workflow file:

```bash
git clone https://github.com/openai/symphony
cd symphony/elixir
mise trust
mise install
mise exec -- mix setup
mise exec -- mix build
mise exec -- ./bin/symphony /Users/bensonlee/dev/dagzoo/WORKFLOW.md
```

If you want the dashboard:

```bash
mise exec -- ./bin/symphony /Users/bensonlee/dev/dagzoo/WORKFLOW.md --port 4040
```

## Migrating GitHub issues to Linear

Dry-run a subset first:

```bash
uv run python scripts/linear/github_to_linear.py \
  --repo bensonlee5/dagzoo \
  --linear-api-key-file ~/.linear/linear_api_key.txt \
  --project-slug 4867d49bb182 \
  --mapping-path reference/linear_issue_map_2026-03-08.json \
  --issue-number 148 \
  --issue-number 146 \
  --issue-number 175 \
  --dry-run
```

Run the full migration and GitHub cutover:

```bash
uv run python scripts/linear/github_to_linear.py \
  --repo bensonlee5/dagzoo \
  --linear-api-key-file ~/.linear/linear_api_key.txt \
  --project-slug 4867d49bb182 \
  --mapping-path reference/linear_issue_map_2026-03-08.json \
  --mode all
```

## Migration defaults

- All open GitHub issues migrate to Linear `Backlog`.
- All closed GitHub issues migrate to Linear `Done`.
- GitHub labels `P0`, `P1`, and `P2` map to Linear priorities `Urgent`, `High`, and `Normal`.
- Non-priority GitHub labels are created or reused as Linear labels.
- GitHub issues labeled `epic` become parent issues when their bodies explicitly reference child issue numbers.
- After cutover, GitHub issues are commented with the Linear successor URL and remaining open issues are closed.
