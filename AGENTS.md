# Development Patterns

- Use `.venv/` for commands and tests in this repo.
- Prefer breaking dependency cycles and centralizing shared wiring in `src/dagzoo/core`; avoid long-lived shims.
- We optimize for iteration speed: internal Python APIs and internal config structure may change without backward-compat guarantees.
- If CLI flags, persisted metadata schema, or dataset artifact contract changes, treat it as a user-facing break and call it out explicitly.
- For behavior/schema changes under `src/dagzoo`, bump version in `pyproject.toml` (patch by default; minor for intentionally broad user-facing breaks). Docs/tests-only changes do not require a bump.
- On every version bump, update `CHANGELOG.md` in the same PR.
