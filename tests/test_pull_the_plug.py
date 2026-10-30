"""scripts/pull_the_plug.py: the same check-in twice, --mode api then --mode auto with the API unreachable.

Each run is the real CLI in a child process; the summary is read back from the two signed receipts. Local answers
come from the FIXTURE fake worker of tests/test_local_modes.py. No key, so nothing leaves the machine, except in the
one test that points both runs at closed ports on 127.0.0.1 to exercise the transport-failure fallback.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.test_local_modes import DOC_FIELDS, FAKE_WORKER

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures" / "accept"
INPUTS = ["--load", str(FIX / "load.FIXTURE.png"), "--doc", str(FIX / "doc.FIXTURE.png"),
          "--voice", str(FIX / "voice.FIXTURE.wav")]


def _script():
    spec = importlib.util.spec_from_file_location("pull_the_plug", ROOT / "scripts" / "pull_the_plug.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    worker = tmp_path / "fake-python"
    worker.write_text(FAKE_WORKER.format(python=sys.executable))
    worker.chmod(0o755)
    model = tmp_path / "fake-glimmer"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "FIXTURE"}')
    for k, v in {"TINYTMS_LOCAL_PYTHON": str(worker), "TINYTMS_GLIMMER_PATH": str(model),
                 "FAKE_LOG": str(tmp_path / "worker.log"), "FAKE_DOC": json.dumps(DOC_FIELDS)}.items():
        monkeypatch.setenv(k, v)
    for k in ("MODEL_API_KEY", "FAKE_FAIL_LOAD", "TINYTMS_META_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def _run(m, tmp: Path, *extra: str) -> int:
    return m.main([*INPUTS, "--receipts-dir", str(tmp / "r"), "--keys-dir", str(tmp / "k"),
                   "--unreachable-url", "http://127.0.0.1:9/v1", *extra])


def _receipts(tmp: Path) -> dict[str, dict]:
    out = {}
    for p in sorted((tmp / "r").glob("*.json")):
        if p.name != "llamadas.json":
            body = json.loads(p.read_text(encoding="utf-8"))
            out[body["mode"]] = body
    return out


def _row(out: str, label: str) -> str:
    return next(line for line in out.splitlines() if line.startswith(label))


def test_two_runs_second_falls_back_for_vision_and_doc(local_env, capsys):
    m = _script()
    assert _run(m, local_env) == 0
    out = capsys.readouterr().out
    assert "run 1 · --mode api" in out and "run 2 · --mode auto, API unreachable" in out

    r = _receipts(local_env)
    assert set(r) == {"api", "auto"}
    assert all(w["status"] == "NO_MODEL" for w in r["api"]["witnesses"].values())  # no key in the cloud
    assert {"vision", "doc"} <= set(r["auto"]["fallback"])
    assert r["auto"]["witnesses"]["vision"]["provider"] == "local-mlx"
    assert r["auto"]["witnesses"]["vision"]["count"] == 12 and r["auto"]["witnesses"]["doc"]["pieces"] == 12
    assert r["auto"]["witnesses"]["voice"]["status"] == "NO_MODEL"

    vision, doc, voice = _row(out, "vision"), _row(out, "doc"), _row(out, "voice")
    run2 = lambda line: line.split(" | ")[1]  # noqa: E731
    assert "local-mlx" in run2(vision) and "fallback" in run2(vision) and "count=12" in run2(vision)
    assert "local-mlx" in run2(doc) and "fallback" in run2(doc) and "pieces=12" in run2(doc)
    assert "NO_MODEL" in run2(voice)
    assert _row(out, "verdict").count("HOLD") == 2
    assert _row(out, "receipt").count(".json") == 2


def test_local_worker_that_cannot_load_shows_error_never_a_number(local_env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_FAIL_LOAD", "1")
    m = _script()
    assert _run(m, local_env) == 0
    out = capsys.readouterr().out
    for label in ("vision", "doc"):
        cell = _row(out, label).split(" | ")[1]
        assert "ERROR" in cell and "count=" not in cell and "pieces=" not in cell and "- s" in cell
    assert "the Mac did not answer" in out


def test_no_local_model_at_all_shows_no_model_and_says_so(local_env, monkeypatch, capsys):
    monkeypatch.setenv("TINYTMS_GLIMMER_PATH", str(local_env / "nothing-here"))
    m = _script()
    assert _run(m, local_env) == 0
    out = capsys.readouterr().out
    cell = _row(out, "vision").split(" | ")[1]
    assert "NO_MODEL" in cell and "count=" not in cell and "- s" in cell
    assert "the Mac did not answer" in out


def test_unreachable_api_falls_back_on_transport_failure(local_env, monkeypatch, capsys):
    m = _script()
    monkeypatch.setenv("MODEL_API_KEY", "FIXTURE-not-a-key")
    monkeypatch.setenv("TINYTMS_META_BASE_URL", m.closed_local_url())  # run 1 must not reach the real API either
    assert m.main([*INPUTS, "--receipts-dir", str(local_env / "r"), "--keys-dir", str(local_env / "k")]) == 0
    r = _receipts(local_env)
    assert r["api"]["calls"]["vision"]["detail"].startswith("transport:")
    assert all(f["why"].startswith("transport:") for f in r["auto"]["fallback"].values())
    assert r["auto"]["witnesses"]["vision"]["provider"] == "local-mlx"
    assert "FIXTURE-not-a-key" not in capsys.readouterr().out


def test_missing_input_is_an_operational_error(local_env, capsys):
    m = _script()
    assert m.main(["--load", str(local_env / "nope.jpg"), *INPUTS[2:], "--receipts-dir", str(local_env / "r"),
                   "--keys-dir", str(local_env / "k"), "--unreachable-url", "http://127.0.0.1:9/v1"]) == 3
    assert "run 1" in capsys.readouterr().err


def test_closed_local_url_is_loopback():
    m = _script()
    url = m.closed_local_url()
    assert url.startswith("http://127.0.0.1:") and url.endswith("/v1") and int(url.split(":")[2].split("/")[0]) > 0
