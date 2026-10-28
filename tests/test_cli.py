import json
import subprocess
import sys
from pathlib import Path

import pytest

from tinytms.receipt import verify_receipt

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "fixtures"


def run(tmp_path, *args, env_extra=None, drop_key=True):
    import os

    env = {k: v for k, v in os.environ.items() if not (drop_key and k == "MODEL_API_KEY")}
    env.update(env_extra or {})
    cmd = [sys.executable, "-m", "tinytms.cli", "checkin", "--receipts-dir", str(tmp_path / "receipts"),
           "--keys-dir", str(tmp_path / "keys"), *args]
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def only_receipt(tmp_path):
    files = list((tmp_path / "receipts").glob("*.json"))
    assert len(files) == 1, files
    return files[0], json.loads(files[0].read_text())


def test_fixture_accept(tmp_path):
    p = run(tmp_path, "--mode", "fixture", "--fixture-dir", str(FIX / "accept"))
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.startswith("ACCEPT")
    path, r = only_receipt(tmp_path)
    assert path.name.endswith("-ACCEPT-FIXTURE.json")
    assert verify_receipt(r)
    pub = (tmp_path / "keys" / "ed25519.pub").read_text().strip()
    assert verify_receipt(path.read_bytes(), trusted_public_key=pub)
    assert r["verdict"]["decision"] == "ACCEPT"
    assert {m["provider"] for m in r["models"].values()} == {"FIXTURE"}
    assert "FIXTURE" in r["provenance"]
    for k, name in (("load", "load.FIXTURE.png"), ("doc", "doc.FIXTURE.png"), ("voice", "voice.FIXTURE.wav")):
        assert r["inputs"][k]["name"] == name and len(r["inputs"][k]["sha256"]) == 64
    priv = (tmp_path / "keys" / "ed25519.key").read_text().strip()
    assert priv not in p.stdout + p.stderr + path.read_text()


@pytest.mark.parametrize("case,reason", [
    ("count_mismatch", "COUNT_MISMATCH doc=12 vision=11 voice=12"),
    ("seal_mismatch", "SEAL_MISMATCH"),
    ("overweight", "OVERWEIGHT"),
])
def test_fixture_hold_cases(tmp_path, case, reason):
    p = run(tmp_path, "--mode", "fixture", "--fixture-dir", str(FIX / case))
    assert p.returncode == 2, p.stdout + p.stderr
    _, r = only_receipt(tmp_path)
    assert r["verdict"]["decision"] == "HOLD"
    assert any(x.startswith(reason) for x in r["verdict"]["reasons"])
    assert verify_receipt(r)


def test_max_weight_flag(tmp_path):
    p = run(tmp_path, "--mode", "fixture", "--fixture-dir", str(FIX / "accept"), "--max-weight-kg", "8000")
    assert p.returncode == 2 and "OVERWEIGHT" in p.stdout


def test_api_without_key_holds_no_model(tmp_path):
    a = FIX / "accept"
    p = run(tmp_path, "--mode", "api", "--load", str(a / "load.FIXTURE.png"), "--doc", str(a / "doc.FIXTURE.png"),
            "--voice", str(a / "voice.FIXTURE.wav"))
    assert p.returncode == 2, p.stdout + p.stderr
    assert "NO_MODEL:vision" in p.stdout
    _, r = only_receipt(tmp_path)
    assert r["verdict"]["reasons"] == ["NO_MODEL:vision", "NO_MODEL:doc", "NO_MODEL:voice"]
    assert {m["provider"] for m in r["models"].values()} == {"meta-api"}
    assert verify_receipt(r)


def test_missing_input_is_exit_3(tmp_path):
    p = run(tmp_path, "--mode", "api", "--load", "nope.png", "--doc", "nope.png", "--voice", "nope.wav")
    assert p.returncode == 3
    assert not (tmp_path / "receipts").exists()


def test_fixture_mode_without_dir_is_exit_3(tmp_path):
    assert run(tmp_path, "--mode", "fixture").returncode == 3


def test_unlabelled_fixture_is_refused(tmp_path):
    d = tmp_path / "fx"
    d.mkdir()
    for f in (FIX / "accept").iterdir():
        (d / f.name).write_bytes(f.read_bytes())
    data = json.loads((d / "vision.FIXTURE.json").read_text())
    data["provenance"] = "recorded from SAM"  # not labelled FIXTURE -> refused
    (d / "vision.FIXTURE.json").write_text(json.dumps(data))
    p = run(tmp_path, "--mode", "fixture", "--fixture-dir", str(d))
    assert p.returncode == 2 and "ERROR:vision" in p.stdout


def test_local_mode_without_cached_model_holds(tmp_path):
    # Empty HF cache -> NO_MODEL; never downloads, never loads a 30B model in tests.
    a = FIX / "accept"
    p = run(tmp_path, "--mode", "local", "--load", str(a / "load.FIXTURE.png"), "--doc", str(a / "doc.FIXTURE.png"),
            "--voice", str(a / "voice.FIXTURE.wav"),
            env_extra={"HF_HUB_CACHE": str(tmp_path / "empty-hf"), "TINYTMS_GLIMMER_PATH": ""})
    assert p.returncode == 2, p.stdout + p.stderr
    _, r = only_receipt(tmp_path)
    assert r["verdict"]["reasons"] == ["NO_MODEL:vision", "NO_MODEL:doc", "NO_MODEL:voice"]
    assert "not in local cache" in r["witnesses"]["vision"]["detail"]
