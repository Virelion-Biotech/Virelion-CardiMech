import json

from cardimech.hearttwin_cli import dispatch, main


def test_hearttwin_dispatch_health_and_reference_validation() -> None:
    assert dispatch("mechanics.health", {})["service"] == "CardiMech"
    assert dispatch("mechanics.validate.reference", {})["passed"] is True


def test_hearttwin_cli_reads_standard_env(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HEARTTWIN_CAPABILITY", "mechanics.health")
    monkeypatch.setenv("HEARTTWIN_PAYLOAD", "{}")
    monkeypatch.delenv("HEARTTWIN_PAYLOAD_STDIN", raising=False)
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["service"] == "CardiMech"


def test_hearttwin_cli_rejects_unknown_capability(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HEARTTWIN_CAPABILITY", "mechanics.nope")
    monkeypatch.setenv("HEARTTWIN_PAYLOAD", "{}")
    assert main() == 1
    assert "does not support" in capsys.readouterr().err
