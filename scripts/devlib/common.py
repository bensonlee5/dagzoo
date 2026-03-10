from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "dagzoo"
DOCS_DEP_MAP_PATH = REPO_ROOT / "docs" / "development" / "module-dependency-map.md"


@dataclass(frozen=True)
class CommandSpec:
    label: str
    argv: tuple[str, ...]


class DevToolError(RuntimeError):
    """Raised for developer tooling failures with user-facing messages."""


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def python_tool(tool_name: str) -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / tool_name
    if candidate.exists():
        return str(candidate)
    return tool_name


def venv_python() -> Path:
    return REPO_ROOT / ".venv" / "bin" / "python"


def run_command(command: CommandSpec) -> None:
    result = subprocess.run(command.argv, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise DevToolError(f"{command.label} failed with exit code {result.returncode}.")


def format_command(argv: tuple[str, ...]) -> str:
    return " ".join(argv)


def tool_exists(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None
