from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_build_ticket_specs_contains_epic_and_audit_ticket() -> None:
    specs = MODULE.build_ticket_specs()
    titles = [spec.title for spec in specs]
    assert MODULE.HARNESS_EPIC_TITLE in titles
    assert MODULE.RECURRING_AUDIT_TITLE in titles
    assert len(titles) == len(set(titles))


def test_audit_ticket_defaults_to_todo_and_references_docs() -> None:
    specs = MODULE.build_ticket_specs()
    audit = next(spec for spec in specs if spec.title == MODULE.RECURRING_AUDIT_TITLE)
    assert audit.state_name == "Todo"
    assert "Friday" in audit.description
    assert "10:00 PM `America/Los_Angeles`" in audit.description
    assert "docs/development/harness_audit.md" in audit.description
