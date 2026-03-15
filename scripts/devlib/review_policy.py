from __future__ import annotations

RELEASE_RISK_EXACT_PATHS = frozenset(
    {
        "src/dagzoo/core/config_resolution.py",
        "src/dagzoo/core/dataset.py",
        "src/dagzoo/core/fixed_layout/runtime.py",
        "src/dagzoo/core/metadata.py",
    }
)
RELEASE_RISK_PREFIXES = (
    "src/dagzoo/cli/",
    "src/dagzoo/config/",
    "src/dagzoo/io/",
    "configs/",
)

SUGGESTED_PYTEST_TARGETS = (
    (
        "src/dagzoo/cli/",
        (
            "tests/test_cli_validation.py",
            "tests/test_cli_outputs.py",
            "tests/test_benchmark_cli.py",
            "tests/test_generate_handoff.py",
        ),
    ),
    (
        "src/dagzoo/bench/",
        (
            "tests/test_benchmark_suite.py",
            "tests/test_benchmark_cli.py",
            "tests/test_benchmark_stage_metrics.py",
            "tests/test_benchmark_throughput.py",
        ),
    ),
    (
        "src/dagzoo/config/",
        (
            "tests/test_config.py",
            "tests/test_config_resolution.py",
            "tests/test_generate_handoff.py",
        ),
    ),
    (
        "src/dagzoo/core/config_resolution.py",
        (
            "tests/test_config.py",
            "tests/test_config_resolution.py",
            "tests/test_generate_handoff.py",
        ),
    ),
    (
        "configs/",
        (
            "tests/test_config.py",
            "tests/test_config_resolution.py",
            "tests/test_generate_handoff.py",
            "tests/test_benchmark_cli.py",
        ),
    ),
    ("scripts/devlib/", ("tests/test_dev_tooling.py",)),
    (".pre-commit-config.yaml", ("tests/test_dev_tooling.py",)),
    (
        "scripts/docs/",
        (
            "tests/test_docs_scripts.py",
            "tests/test_dev_tooling.py",
        ),
    ),
    ("README.md", ("tests/test_docs_scripts.py",)),
    ("docs/", ("tests/test_docs_scripts.py",)),
    ("site/", ("tests/test_docs_scripts.py",)),
)


def is_release_risk_path(path: str) -> bool:
    return path in RELEASE_RISK_EXACT_PATHS or path.startswith(RELEASE_RISK_PREFIXES)


def suggested_pytest_targets(changed_files: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    suggested: list[str] = []
    for changed_file in changed_files:
        for path_prefix, targets in SUGGESTED_PYTEST_TARGETS:
            if changed_file == path_prefix or changed_file.startswith(path_prefix):
                for target in targets:
                    if target in seen:
                        continue
                    seen.add(target)
                    suggested.append(target)
    return tuple(suggested)
