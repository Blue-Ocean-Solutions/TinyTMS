"""--mode local and --mode auto, end to end through the REAL Worker plumbing, with a FAKE worker.

No local model runs here (MLX has no Metal device in CI). TINYTMS_LOCAL_PYTHON points at a tiny executable
written below that speaks _mlx_worker.py's JSON-lines protocol and answers with canned FIXTURE text. Every
model answer in this file is a FIXTURE; API answers come from httpx.MockTransport. No network.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from tinytms.local_vlm import COUNT_PROMPT, parse_single_integer
from tinytms.meta_api import DOC_PROMPT, ApiResult, MetaClient, is_transport_failure
from tinytms.pipeline import run_checkin
from tinytms.receipt import find_machine_paths, verify_receipt

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "fixtures"
DOC_FIELDS = {"pieces": 12, "weight_kg": 4800, "seal": "MX-48213", "shipper": "FIXTURE", "consignee": "FIXTURE",
              "hazmat": False}

FAKE_WORKER = '''#!{python}
"""FIXTURE fake MLX worker: the _mlx_worker.py JSON-lines protocol, no model, canned FIXTURE answers."""
import json, os, sys
def emit(**kw):
    sys.stdout.write(json.dumps(kw) + "\\n"); sys.stdout.flush()
emit(event="hello", python="fake", executable="fake-worker", package="mlx-vlm", package_version="FIXTURE")
if os.environ.get("FAKE_FAIL_LOAD"):
    emit(event="error", stage="load", error="FIXTURE load failure"); sys.exit(1)
emit(event="loaded", load_s=0.0)
for line in sys.stdin:
    req = json.loads(line)
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({{"argv": sys.argv[1:], "op": req.get("op"), "prompt": req.get("prompt"),
                            "image": os.path.basename(req.get("image", "")), "max_tokens": req.get("max_tokens")}}) + "\\n")
    p = req.get("prompt", "")
    if "single integer" in p:
        text = os.environ.get("FAKE_COUNT", "12")
    elif "freight document" in p:
        text = os.environ["FAKE_DOC"]
    else:
        emit(event="error", stage="request", error="FIXTURE: unexpected prompt"); continue
    emit(event="result", text=text, prompt_tokens=None, generation_tokens=None, generation_tps=None,
         peak_memory_gb=None, wall_s=0.0)
'''


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to open a real HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)


@pytest.fixture
def fake_local(tmp_path, monkeypatch):
    worker = tmp_path / "fake-python"
    worker.write_text(FAKE_WORKER.format(python=sys.executable))
    worker.chmod(0o755)
    model = tmp_path / "fake-glimmer"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "FIXTURE"}')
    log = tmp_path / "worker.log"
    env = {"TINYTMS_LOCAL_PYTHON": str(worker), "TINYTMS_GLIMMER_PATH": str(model), "FAKE_LOG": str(log),
           "FAKE_DOC": json.dumps(DOC_FIELDS)}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("FAKE_FAIL_LOAD", raising=False)
    return env, (lambda: [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else [])


@pytest.fixture
def media(tmp_path):
    """Decodable images drawn here (FIXTURE media) + the fixture WAV."""
    out = {}
    for k in ("load", "doc"):
        p = tmp_path / f"{k}.png"
        p.write_bytes(cv2.imencode(".png", np.full((6, 8, 3), 127, dtype=np.uint8))[1].tobytes())
        out[k] = p
    out["voice"] = FIX / "accept" / "voice.FIXTURE.wav"
    return out


def run(mode, tmp_path, media, **kw):
    return run_checkin(mode, receipts_dir=tmp_path / "r", keys_dir=tmp_path / "k", **media, **kw)


def test_parse_single_integer_is_strict():
    assert [parse_single_integer(t) for t in ("12", " 12\n", "12.", "0")] == [12, 12, 12, 0]
    for t in ("twelve", "I count 12 pallets", "12 pallets", "12, maybe 13", "-3", "unknown", "", "1.5"):
        assert parse_single_integer(t) is None, t
    assert "single integer" in COUNT_PROMPT.format(concept="pallet")


def test_local_mode_glimmer_for_paper_and_view_voice_no_model(tmp_path, media, fake_local):
    _, calls = fake_local
    res = run("local", tmp_path, media)
    r = res.receipt
    assert res.verdict.decision == "HOLD" and r["verdict"]["reasons"] == ["NO_MODEL:voice"]
    assert r["models"] == {"vision": {"model": "muse-glimmer-30b-4bit@local", "provider": "local-mlx"},
                           "doc": {"model": "muse-glimmer-30b-4bit@local", "provider": "local-mlx"},
                           "voice": {"model": "", "provider": "local-mlx"}}
    assert r["witnesses"]["vision"]["count"] == 12
    w = r["witnesses"]["doc"]
    assert (w["pieces"], w["weight_kg"], w["seal"], w["hazmat"]) == (12, 4800.0, "MX-48213", False)
    assert "speech-to-text" in r["witnesses"]["voice"]["detail"]
    got = calls()
    assert [c["prompt"] for c in got] == [COUNT_PROMPT.format(concept="pallet"), DOC_PROMPT]
    assert [c["image"] for c in got] == ["load.png", "doc.png"] and got[0]["argv"][2:4] == ["--kind", "vlm"]
    assert verify_receipt(r) and find_machine_paths(r) == []


def test_local_vision_prose_is_no_count_not_a_guess(tmp_path, media, fake_local, monkeypatch):
    monkeypatch.setenv("FAKE_COUNT", "I can see about 12 pallets")
    r = run("local", tmp_path, media).receipt
    assert r["witnesses"]["vision"]["status"] == "OK" and r["witnesses"]["vision"]["count"] is None
    assert "VISION_NO_COUNT" not in r["verdict"]["reasons"]  # voice NO_MODEL stops the judge first
    assert "not a single integer" in r["witnesses"]["vision"]["detail"]


def test_local_load_failure_is_error_witnesses(tmp_path, media, fake_local, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_LOAD", "1")
    r = run("local", tmp_path, media).receipt
    assert r["verdict"]["reasons"] == ["ERROR:vision", "ERROR:doc", "NO_MODEL:voice"]
    assert "FIXTURE load failure" in r["witnesses"]["doc"]["detail"]


# --- auto --------------------------------------------------------------------------------------

SPARK_OK = {"status": "completed", "output": [{"type": "message", "content": [
    {"type": "output_text", "text": json.dumps(DOC_FIELDS)}]}], "usage": {"output_tokens": 1}}


def api(handler):
    return lambda **kw: MetaClient(api_key="FIXTURE-key", transport=httpx.MockTransport(handler), **kw)


def test_transport_failure_classifier():
    def e(**kw):
        return ApiResult("ERROR", "m", **kw)

    assert is_transport_failure(e(http_status=503)) and is_transport_failure(e(http_status=401))
    assert is_transport_failure(e(detail="timeout: ReadTimeout")) and is_transport_failure(e(detail="transport: x"))
    assert not is_transport_failure(e(http_status=200, detail="response is not JSON"))
    assert not is_transport_failure(e(detail="budget cap reached for m"))
    assert not is_transport_failure(e(detail="x.png: not a JPEG or PNG"))
    assert not is_transport_failure(ApiResult("OK", "m", http_status=200))


def test_auto_falls_back_per_witness(tmp_path, media, fake_local):
    _, calls = fake_local

    def h(req):
        if req.url.path.endswith("/asr/transcribe"):
            raise httpx.ConnectError("FIXTURE no route")
        if b'"stream": true' in req.content or b'"stream":true' in req.content:
            return httpx.Response(503, text="FIXTURE SAM down")
        return httpx.Response(200, json=SPARK_OK)

    res = run("auto", tmp_path, media, client_factory=api(h))
    r = res.receipt
    assert r["mode"] == "auto"
    assert {k: m["provider"] for k, m in r["models"].items()} == \
        {"vision": "local-mlx", "doc": "meta-api", "voice": "local-mlx"}
    assert r["models"]["vision"]["model"] == "muse-glimmer-30b-4bit@local"
    assert r["models"]["doc"]["model"] == "muse-spark-1.3"
    assert set(r["fallback"]) == {"vision", "voice"}
    assert "HTTP 503" in r["fallback"]["vision"]["why"] and "ConnectError" in r["fallback"]["voice"]["why"]
    assert set(r["calls"]) == {"vision", "doc", "voice"}  # the failed API calls stay on the record
    assert r["witnesses"]["vision"]["count"] == 12 and r["verdict"]["reasons"] == ["NO_MODEL:voice"]
    assert [c["prompt"] for c in calls()] == [COUNT_PROMPT.format(concept="pallet")]  # only vision went local
    assert verify_receipt(r)


def test_auto_without_key_is_all_local(tmp_path, media, fake_local):
    r = run("auto", tmp_path, media).receipt
    assert {m["provider"] for m in r["models"].values()} == {"local-mlx"}
    assert all("NoKey" in f["why"] for f in r["fallback"].values()) and set(r["fallback"]) == {"vision", "doc", "voice"}
    assert "calls" not in r


def test_auto_keeps_api_errors_that_are_answers(tmp_path, media, fake_local):
    def h(req):
        if req.url.path.endswith("/asr/transcribe"):
            return httpx.Response(200, json={"transcript": "FIXTURE doce pallets, sello MX-48213."})
        if b'"stream": true' in req.content or b'"stream":true' in req.content:
            return httpx.Response(200, text="data: {\"type\":\"response.output_text.done\",\"text\":\"\"}\n")
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "FIXTURE not json"}]}]})

    _, calls = fake_local
    r = run("auto", tmp_path, media, client_factory=api(h)).receipt
    assert "fallback" not in r and calls() == []
    assert r["witnesses"]["doc"]["status"] == "ERROR" and r["models"]["doc"]["provider"] == "meta-api"
    assert r["verdict"]["reasons"] == ["ERROR:doc"]


def test_cli_local_and_auto_with_fake_worker(tmp_path, media, fake_local):
    env_extra, _ = fake_local
    env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
    for mode in ("local", "auto"):
        rd = tmp_path / f"r-{mode}"
        p = subprocess.run([sys.executable, "-m", "tinytms.cli", "checkin", "--mode", mode, "--load", str(media["load"]),
                            "--doc", str(media["doc"]), "--voice", str(media["voice"]), "--receipts-dir", str(rd),
                            "--keys-dir", str(tmp_path / "k")], cwd=REPO, env=env, capture_output=True, text=True,
                           timeout=120)
        assert p.returncode == 2, p.stdout + p.stderr
        assert "reason: NO_MODEL:voice" in p.stdout and "vision: OK model=muse-glimmer-30b-4bit@local" in p.stdout
        (f,) = rd.glob("*-HOLD.json")
        assert verify_receipt(f.read_bytes())
        if mode == "auto":
            assert "fallback: vision -> local-mlx (NoKey" in p.stdout
