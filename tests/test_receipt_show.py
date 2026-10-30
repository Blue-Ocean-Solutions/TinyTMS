"""scripts/receipt_show.py: one screen per receipt, signature checked first; a tampered receipt exits 1."""

import importlib.util
import json
from pathlib import Path

import pytest

from tinytms.receipt import MACHINE_PATH_PREFIXES, load_or_create_key, sign_receipt

ROOT = Path(__file__).resolve().parent.parent
RECEIPTS = sorted(p for p in (ROOT / "receipts").glob("*.json") if p.name != "llamadas.json")
LOCAL_HOLD = ROOT / "receipts" / "20261028T165434299357Z-HOLD.json"
API_HOLD = ROOT / "receipts" / "20261028T153041947742Z-HOLD.json"
SEAT_KEY = "50be0cedb34ab0b93023708d5059408480839afad77d8ef7889ef3b5f53fbb9c"


def _script():
    spec = importlib.util.spec_from_file_location("receipt_show", ROOT / "scripts" / "receipt_show.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", RECEIPTS, ids=lambda p: p.name)
def test_every_committed_receipt_prints_and_exits_0(path, capsys):
    m = _script()
    assert m.main([str(path)]) == 0
    out = capsys.readouterr().out
    body = json.loads(path.read_text(encoding="utf-8"))
    assert "signature OK" in out and body["kind"] in out
    if body["kind"] == "checkin":
        assert f"verdict: {body['verdict']['decision']}" in out


def test_checkin_view_names_provider_model_seconds_and_reasons(capsys):
    m = _script()
    assert m.main([str(API_HOLD)]) == 0
    out = capsys.readouterr().out
    calls = json.loads(API_HOLD.read_text(encoding="utf-8"))["calls"]
    assert f"vision  OK  meta-api  sam-3.1  {calls['vision']['s']} s  count=1" in out
    assert "doc     OK  meta-api  muse-spark-1.3" in out and "pieces=12" in out and "seal=MX-48213" in out
    assert "reason: COUNT_MISMATCH doc=12 vision=1" in out
    assert "sha256 651abf360ba0" in out  # inputs by sha


def test_local_view_takes_seconds_only_from_the_receipt(capsys):
    m = _script()
    assert m.main([str(LOCAL_HOLD)]) == 0
    out = capsys.readouterr().out
    assert "vision  OK  local-mlx  muse-glimmer-30b-4bit@local  9.924 s  count=1" in out
    assert "doc     OK  local-mlx  muse-glimmer-30b-4bit@local  - s" in out  # doc_local records no seconds
    assert "voice   NO_MODEL  local-mlx" in out and "reason: NO_MODEL:voice" in out


def test_tampered_copy_exits_1(tmp_path, capsys):
    m = _script()
    body = json.loads(API_HOLD.read_text(encoding="utf-8"))
    body["verdict"]["decision"] = "ACCEPT"
    bad = tmp_path / API_HOLD.name
    bad.write_text(json.dumps(body), encoding="utf-8")
    assert m.main([str(bad)]) == 1
    out = capsys.readouterr().out
    assert "signature FAILED" in out and "signature OK" not in out


def test_trusted_key_pins_origin(capsys):
    m = _script()
    assert m.main([str(LOCAL_HOLD), "--trusted-key", SEAT_KEY]) == 0
    assert m.main([str(API_HOLD), "--trusted-key", SEAT_KEY]) == 1  # signed by the API lane, not the seat
    assert "signature FAILED" in capsys.readouterr().out


def test_redacted_machine_paths_is_shown(tmp_path, capsys):
    m = _script()
    sk = load_or_create_key(tmp_path / "keys")
    body = {"kind": "checkin", "utc": "FIXTURE", "mode": "fixture", "command": MACHINE_PATH_PREFIXES[0] + "someone/x",
            "verdict": {"decision": "HOLD", "reasons": ["FIXTURE"]}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(sign_receipt(body, sk)), encoding="utf-8")
    assert m.main([str(p)]) == 0
    assert "redacted_machine_paths: 1" in capsys.readouterr().out


def test_missing_or_unreadable_file_is_an_operational_error(tmp_path, capsys):
    m = _script()
    assert m.main([str(tmp_path / "nope.json")]) == 3
    garbage = tmp_path / "g.json"
    garbage.write_text("not json", encoding="utf-8")
    assert m.main([str(garbage)]) == 1  # unreadable is unverifiable: never shown as a receipt
    assert "signature FAILED" in capsys.readouterr().out
