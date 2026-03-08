from __future__ import annotations

from pathlib import Path


def test_workflow_has_parseable_yaml_frontmatter() -> None:
    workflow_path = Path(__file__).resolve().parents[1] / "WORKFLOW.md"
    text = workflow_path.read_text()
    lines = text.splitlines()

    assert lines[0] == "---"
    closing_index = lines.index("---", 1)
    frontmatter = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])

    assert "tracker:" in frontmatter
    assert 'project_slug: "4867d49bb182"' in frontmatter
    assert "workspace:" in frontmatter
    assert "codex:" in frontmatter
    assert ".codex/skills/" not in body
    assert "You are working on a Linear ticket" in body
