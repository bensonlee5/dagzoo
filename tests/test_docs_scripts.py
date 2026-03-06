from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(module_name: str, rel_path: str):
    script_path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_effective_diversity_script_delegates_to_cli(monkeypatch) -> None:
    module = _load_module(
        "effective_diversity_audit_script",
        "scripts/effective_diversity_audit.py",
    )
    captured: dict[str, object] = {}

    def _stub_cli_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(module, "cli_main", _stub_cli_main)

    code = module.main(["--phase", "local", "--out-dir", "tmp/out"])

    assert code == 7
    assert captured["argv"] == [
        "diversity-audit",
        "--phase",
        "local",
        "--out-dir",
        "tmp/out",
    ]


def test_sync_docs_helpers_handle_route_aliases_and_heading_attrs() -> None:
    module = _load_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")

    stripped = module._strip_matching_h1(
        "# How dagzoo Works {#how-dagzoo-works}\n\nBody\n",
        title="How dagzoo Works",
    )
    route_map = module._build_route_map("/dagzoo")

    assert stripped == "Body\n"
    assert route_map["how-it-works.md"] == "/dagzoo/docs/how-it-works/"
    assert route_map["transforms.md"] == "/dagzoo/docs/transforms/"
    assert route_map["how-it-works.html"] == "/dagzoo/docs/how-it-works/"
    assert route_map["transforms.html"] == "/dagzoo/docs/transforms/"


def test_sync_docs_front_matter_includes_aliases_and_page_flags() -> None:
    module = _load_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")
    meta = module.PAGE_METADATA["how-it-works.md"]

    front_matter = module._front_matter("How It Works", meta)

    assert "title: How It Works" in front_matter
    assert "aliases:" in front_matter
    assert "/canonical/how-it-works.html" in front_matter
    assert "mermaid: true" in front_matter


def test_sync_docs_removes_legacy_generated_paths(tmp_path) -> None:
    module = _load_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")
    legacy_dir = tmp_path / "static" / "canonical"
    legacy_dir.mkdir(parents=True)

    changed: list[Path] = []
    module._remove_stale_path(legacy_dir, check=False, changed=changed)

    assert changed == [legacy_dir]
    assert not legacy_dir.exists()
