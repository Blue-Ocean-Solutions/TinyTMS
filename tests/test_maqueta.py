"""scripts/maqueta.py: one command for the dock scene -> signed receipt + a human-readable ACTA.md.

No camera, no model, no network: dry runs use fixtures/<case>; camera:N goes through a monkeypatched
tinytms.capture.capture_frame; api mode runs without MODEL_API_KEY (NO_MODEL, never a call)."""

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from tinytms import capture
from tinytms.receipt import find_machine_paths, verify_receipt
from tinytms.witness import sha256_bytes, sha256_file

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "fixtures"
SCRIPT = REPO / "scripts" / "maqueta.py"

_spec = importlib.util.spec_from_file_location("maqueta", SCRIPT)
assert _spec is not None and _spec.loader is not None
maqueta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(maqueta)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)
    monkeypatch.setattr(socket.socket, "bind", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


def run(tmp_path, *args):
    out = tmp_path / "out"
    code = maqueta.main(["--out", str(out), "--receipts-dir", str(tmp_path / "receipts"),
                         "--keys-dir", str(tmp_path / "keys"), *args])
    return code, out


def receipts(tmp_path):
    return sorted((tmp_path / "receipts").glob("*.json")) if (tmp_path / "receipts").exists() else []


def test_dry_run_accept(tmp_path):
    code, out = run(tmp_path, "--dry-run", "accept")
    assert code == 0
    acta = (out / "ACTA.md").read_text(encoding="utf-8")
    assert find_machine_paths(acta) == []
    # the three inputs are copied next to the record and listed with their sha256
    for name in ("load.FIXTURE.png", "doc.FIXTURE.png", "voice.FIXTURE.wav"):
        sha = sha256_file(FIX / "accept" / name)
        assert sha256_file(out / name) == sha
        assert sha in acta
    [rec] = receipts(tmp_path)
    assert rec.name.endswith("-ACCEPT-FIXTURE.json")
    assert rec.name in acta and sha256_file(rec) in acta
    r = json.loads(rec.read_text())
    assert verify_receipt(r) and r["verdict"]["decision"] == "ACCEPT"
    assert {v["sha256"] for v in r["inputs"].values()} == {sha256_file(FIX / "accept" / n) for n in
                                                          ("load.FIXTURE.png", "doc.FIXTURE.png", "voice.FIXTURE.wav")}
    assert "ACCEPT" in acta and "fixture" in acta and "FIXTURE" in acta
    for w in ("vision", "doc", "voice"):
        assert f"### {w}" in acta
    assert "pieces" in acta and "seal" in acta  # witness fields, not just statuses
    # the command line is recorded without the tmp paths of this machine
    assert "python scripts/maqueta.py" in acta and "--dry-run accept" in acta


def test_dry_run_never_writes_the_repo_receipts(tmp_path):
    # house rule (PR #1): FIXTURE receipts stay out of the public tree; receipts/ holds only real runs
    before = set((REPO / "receipts").iterdir())
    out = tmp_path / "out"
    assert maqueta.main(["--dry-run", "accept", "--out", str(out), "--keys-dir", str(tmp_path / "keys")]) == 0
    assert set((REPO / "receipts").iterdir()) == before
    [rec] = (out / "receipts").glob("*.json")
    assert rec.name.endswith("-ACCEPT-FIXTURE.json") and verify_receipt(json.loads(rec.read_text()))
    acta = (out / "ACTA.md").read_text(encoding="utf-8")
    assert f"receipts/{rec.name}" in acta and find_machine_paths(acta) == []


def test_maqueta_records_are_git_ignored():
    assert "maqueta/" in (REPO / ".gitignore").read_text().splitlines()


def test_dry_run_hold_lists_reasons(tmp_path):
    code, out = run(tmp_path, "--dry-run", "count_mismatch")
    assert code == 2
    acta = (out / "ACTA.md").read_text(encoding="utf-8")
    [rec] = receipts(tmp_path)
    reasons = json.loads(rec.read_text())["verdict"]["reasons"]
    assert reasons and any(x.startswith("COUNT_MISMATCH") for x in reasons)
    for x in reasons:
        assert x in acta
    assert "HOLD" in acta and find_machine_paths(acta) == []


def test_dry_run_refuses_camera_and_models(tmp_path, monkeypatch):
    def never(*a, **k):
        raise AssertionError("dry run touched the camera")

    monkeypatch.setattr(capture, "capture_frame", never)
    assert run(tmp_path, "--dry-run", "accept", "--load", "camera:0")[0] == 3
    assert run(tmp_path, "--dry-run", "accept", "--explain", "api")[0] == 3
    assert run(tmp_path, "--dry-run", "accept", "--mode", "api")[0] == 3
    assert receipts(tmp_path) == []


def _fake_jpeg() -> bytes:
    import cv2
    import numpy as np

    ok, enc = cv2.imencode(".jpg", np.full((48, 64, 3), 128, dtype=np.uint8))
    assert ok
    return enc.tobytes()


def test_camera_frame_is_saved_and_hashed(tmp_path, monkeypatch):
    data = _fake_jpeg()
    seen = []

    def fake(index=0, path=None):
        seen.append(index)
        return capture.Image(data=data, sha256=sha256_bytes(data), width=64, height=48, mime="image/jpeg",
                             source=f"camera:{index}")

    monkeypatch.setattr(capture, "capture_frame", fake)
    a = FIX / "accept"
    code, out = run(tmp_path, "--mode", "api", "--load", "camera:0", "--doc", str(a / "doc.FIXTURE.png"),
                    "--voice", str(a / "voice.FIXTURE.wav"))
    assert seen == [0]
    assert code == 2  # api mode without MODEL_API_KEY: NO_MODEL x3, never a call
    frame = out / "load-camera0.jpg"
    assert frame.read_bytes() == data
    acta = (out / "ACTA.md").read_text(encoding="utf-8")
    assert sha256_bytes(data) in acta and "camera:0" in acta and "load-camera0.jpg" in acta
    assert "NO_MODEL:vision" in acta and find_machine_paths(acta) == []
    [rec] = receipts(tmp_path)
    r = json.loads(rec.read_text())
    assert r["inputs"]["load"] == {"name": "load-camera0.jpg", "sha256": sha256_bytes(data)}


def test_capture_error_is_exit_3_and_said_in_acta(tmp_path, monkeypatch):
    def broken(index=0, path=None):
        raise capture.CaptureError(f"camera {index} cannot be opened (no camera or no permission)")

    monkeypatch.setattr(capture, "capture_frame", broken)
    a = FIX / "accept"
    code, out = run(tmp_path, "--mode", "api", "--load", str(a / "load.FIXTURE.png"), "--doc", "camera:1",
                    "--voice", str(a / "voice.FIXTURE.wav"))
    assert code == 3
    acta = (out / "ACTA.md").read_text(encoding="utf-8")
    assert "CaptureError" in acta and "camera 1 cannot be opened" in acta
    assert not list(out.glob("*.jpg"))  # never a placeholder image
    assert receipts(tmp_path) == []
    assert find_machine_paths(acta) == []


def test_missing_input_is_exit_3(tmp_path):
    code, out = run(tmp_path, "--mode", "api", "--load", "nope.png", "--doc", "nope.png", "--voice", "nope.wav")
    assert code == 3
    assert "ERROR" in (out / "ACTA.md").read_text(encoding="utf-8")
    assert receipts(tmp_path) == []


def test_real_run_needs_mode(tmp_path):
    a = FIX / "accept"
    with pytest.raises(SystemExit) as e:
        run(tmp_path, "--load", str(a / "load.FIXTURE.png"), "--doc", str(a / "doc.FIXTURE.png"),
            "--voice", str(a / "voice.FIXTURE.wav"))
    assert e.value.code != 0


def test_never_overwrites_an_acta(tmp_path):
    assert run(tmp_path, "--dry-run", "accept")[0] == 0
    before = (tmp_path / "out" / "ACTA.md").read_bytes()
    assert run(tmp_path, "--dry-run", "accept")[0] == 3
    assert (tmp_path / "out" / "ACTA.md").read_bytes() == before


def test_script_runs_as_a_command(tmp_path):
    # the Saturday dry run, exactly as CI and the README call it (plus tmp dirs)
    p = subprocess.run([sys.executable, "scripts/maqueta.py", "--dry-run", "--out", str(tmp_path / "out"),
                        "--receipts-dir", str(tmp_path / "receipts"), "--keys-dir", str(tmp_path / "keys")],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.startswith("ACCEPT")
    assert (tmp_path / "out" / "ACTA.md").is_file()
