# Docs Site (Hugo + Docsy)

This directory contains the Hugo docs app.

Source-of-truth docs remain in `../docs/` (single-source model).
Generated Hugo inputs are written to `site/.generated/` and are not tracked by git.
For the full rendered-docs/source-of-truth rationale, see
`docs/development/design-decisions.md` ("Single-source docs with Hugo-rendered
reference pages").
For internal docs links, use Hugo `relref`/`absURL` helpers; do not hardcode `/docs/...` paths.
Sync generated inputs with:

```bash
.venv/bin/python scripts/docs/sync_hugo_content.py
```

Validate synchronization and links:

```bash
.venv/bin/python scripts/docs/sync_hugo_content.py --check
.venv/bin/python scripts/docs/check_links.py
.venv/bin/python scripts/docs/check_built_output_links.py site/public
```

Build locally (requires `hugo` and `go`):

```bash
npm install --prefix site
.venv/bin/python scripts/docs/sync_hugo_content.py
hugo --source site --minify --gc --destination public
.venv/bin/python scripts/docs/check_built_output_links.py site/public
```

Canonical built output is `site/public/`. If you see a top-level `public/`
directory in the repo root, treat it as stale local output from the older build
flow.
