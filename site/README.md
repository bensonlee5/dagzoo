# Docs Site (Hugo + Docsy)

This directory contains the Hugo docs app.

Source-of-truth docs remain in `../docs/` (single-source model).
Generated Hugo inputs are written to `site/.generated/` and are not tracked by git.
Sync generated inputs with:

```bash
.venv/bin/python scripts/docs/sync_hugo_content.py
```

Validate synchronization and links:

```bash
.venv/bin/python scripts/docs/sync_hugo_content.py --check
.venv/bin/python scripts/docs/check_links.py
```

Build locally (requires `hugo` and `go`):

```bash
npm install --prefix site
.venv/bin/python scripts/docs/sync_hugo_content.py
hugo --source site --minify --gc --destination public
```
