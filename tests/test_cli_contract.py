import pytest

from dagzoo.cli.parser import build_parser
from dagzoo.cli.parsing import DEVICE_CHOICES, HARDWARE_POLICY_CHOICES

_COMMAND_BASE_ARGS = {
    "generate": ["generate", "--config", "configs/default.yaml"],
    "benchmark": ["benchmark"],
}


@pytest.mark.parametrize("command", sorted(_COMMAND_BASE_ARGS))
@pytest.mark.parametrize("device", DEVICE_CHOICES)
def test_cli_parser_accepts_supported_device_choices(command: str, device: str) -> None:
    parser = build_parser()

    args = parser.parse_args([*_COMMAND_BASE_ARGS[command], "--device", device])

    assert args.device == device


@pytest.mark.parametrize("command", sorted(_COMMAND_BASE_ARGS))
def test_cli_parser_rejects_invalid_device_choice(command: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([*_COMMAND_BASE_ARGS[command], "--device", "invalid-device"])

    assert int(exc_info.value.code) == 2


@pytest.mark.parametrize("command", sorted(_COMMAND_BASE_ARGS))
@pytest.mark.parametrize("hardware_policy", HARDWARE_POLICY_CHOICES)
def test_cli_parser_accepts_supported_hardware_policies(command: str, hardware_policy: str) -> None:
    parser = build_parser()

    args = parser.parse_args([*_COMMAND_BASE_ARGS[command], "--hardware-policy", hardware_policy])

    assert args.hardware_policy == hardware_policy


@pytest.mark.parametrize("command", sorted(_COMMAND_BASE_ARGS))
def test_cli_parser_rejects_invalid_hardware_policy_choice(command: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([*_COMMAND_BASE_ARGS[command], "--hardware-policy", "missing-policy"])

    assert int(exc_info.value.code) == 2


def test_cli_parser_accepts_generate_handoff_root() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            "handoffs/smoke",
        ]
    )

    assert args.handoff_root == "handoffs/smoke"
