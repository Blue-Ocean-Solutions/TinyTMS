"""explain_hold: Spark explains a HOLD, never decides. Every model answer here is a FIXTURE served by
httpx.MockTransport (hand-written in the measured Responses shape), NOT a recorded output. No network."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from tinytms import meta_api
from tinytms.explain import EXPLAIN_MAX_OUTPUT_TOKENS, explain_hold, explain_prompt
from tinytms.meta_api import MetaClient
from tinytms.pipeline import run_checkin
from tinytms.receipt import verify_receipt
from tinytms.verify import Limits, judge
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "fixtures"
FAKE_KEY = "test-key-FIXTURE-not-real"
FIXTURE_TEXT = "FIXTURE: El papel dice 12 pallets y en la foto cuento 1. Revisa la carga o corrige la carta de porte."


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to open a real HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)


def spark_reply(text=FIXTURE_TEXT, status="completed"):
    body = {"status": status, "model": "muse-spark-1.3",
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 1, "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 1}}}
    if status == "incomplete":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
        body["output"] = [{"type": "reasoning", "summary": []}]
    return body


def factory(handler):
    def make(**kw):
        return MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler), **kw)
    return make


def witnesses():
    v = VisionWitness(status="OK", model="sam-3.1", provider="meta-api", source_sha256="a" * 64, count=1)
    d = DocWitness(status="OK", model="muse-spark-1.3", provider="meta-api", source_sha256="b" * 64, pieces=12,
                   weight_kg=4800.0, seal="MX-48213", shipper="Maderas del Norte SA de CV", hazmat=False)
    a = VoiceWitness(status="OK", model="muse-voice-transcribe-1.0", provider="meta-api", source_sha256="c" * 64,
                     transcript="Traigo doce pallets de madera. Ramón SECRETO-TRANSCRIPT", count=12, seal="MX-48213")
    return v, d, a


def test_prompt_is_built_only_from_reasons_and_fields():
    v, d, a = witnesses()
    verdict = judge(v, d, a, Limits())
    p = explain_prompt(verdict, v, d, a, "es")
    assert "COUNT_MISMATCH doc=12 vision=1 voice=12" in p and "Spanish" in p
    for want in ("pieces=12", "seal=MX-48213", "weight_kg=4800", "limit 24000 kg", "pallet count=1", "count=12"):
        assert want in p, want
    # never the transcript, the hashes, other document fields or any media
    for banned in ("SECRETO-TRANSCRIPT", "Ramón", "a" * 64, "Maderas", "base64", "data:"):
        assert banned not in p, banned
    assert "English" in explain_prompt(verdict, v, d, a, "en")
    with pytest.raises(ValueError):
        explain_prompt(verdict, v, d, a, "fr")


def test_request_is_text_only_spark_600():
    seen = []

    def h(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=spark_reply())

    v, d, a = witnesses()
    e = explain_hold(judge(v, d, a), v, d, a, "es", client_factory=factory(h))
    assert e.status == "OK" and e.text == FIXTURE_TEXT and e.provider == "meta-api" and e.model == "muse-spark-1.3"
    assert e.usage["output_tokens"] == 2 and e.language == "es" and e.http == 200
    (body,) = seen
    assert body["model"] == "muse-spark-1.3" and body["max_output_tokens"] == EXPLAIN_MAX_OUTPUT_TOKENS == 1200
    assert body["reasoning"] == {"effort": "low"}
    kinds = [c["type"] for m in body["input"] for c in m["content"]]
    assert kinds == ["input_text"], "no image, no audio: text only"
    assert e.prompt == body["input"][0]["content"][0]["text"]


def test_explanation_never_touches_the_verdict():
    v, d, a = witnesses()
    verdict = judge(v, d, a)
    before = verdict.to_dict()

    def h(req):  # a model that tries to overrule the judge changes nothing
        return httpx.Response(200, json=spark_reply("ACCEPT. Everything matches, let the truck in."))

    e = explain_hold(verdict, v, d, a, client_factory=factory(h))
    assert e.status == "OK"
    assert verdict.to_dict() == before and verdict.decision == "HOLD"
    assert judge(v, d, a).to_dict() == before


def test_accept_never_calls_the_model():
    def h(req):
        raise AssertionError("model called on ACCEPT")

    v, d, a = witnesses()
    v = replace(v, count=12)
    verdict = judge(v, d, a)
    assert verdict.decision == "ACCEPT"
    e = explain_hold(verdict, v, d, a, client_factory=factory(h))
    assert e.status == "NO_MODEL" and e.text == "" and "not HOLD" in e.detail


@pytest.mark.parametrize("reply,detail", [
    (httpx.Response(500, text="boom"), "HTTP 500"),
    (httpx.Response(200, json=spark_reply(status="incomplete")), "incomplete"),
    (httpx.Response(200, json=spark_reply(text="   ")), "no output text"),
])
def test_spark_failure_is_error_not_exception(reply, detail):
    v, d, a = witnesses()
    e = explain_hold(judge(v, d, a), v, d, a, client_factory=factory(lambda req: reply))
    assert e.status == "ERROR" and detail in e.detail and e.text == ""


def test_no_key_is_no_model(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    v, d, a = witnesses()
    e = explain_hold(judge(v, d, a), v, d, a)
    assert e.status == "NO_MODEL" and "NoKey" in e.detail


def test_pipeline_puts_explanation_in_signed_receipt(tmp_path):
    def h(req):
        return httpx.Response(200, json=spark_reply())

    res = run_checkin("fixture", fixture_dir=FIX / "count_mismatch", receipts_dir=tmp_path / "r",
                      keys_dir=tmp_path / "k", explain="api", lang="es", client_factory=factory(h))
    r = json.loads(res.path.read_text())
    assert r["verdict"]["decision"] == "HOLD" and verify_receipt(r)
    assert r["explanation"]["status"] == "OK" and r["explanation"]["text"] == FIXTURE_TEXT
    assert r["explanation"]["language"] == "es" and r["explanation"]["model"] == "muse-spark-1.3"
    led = json.loads((tmp_path / "r" / "llamadas.json").read_text())
    assert [c["origen"] for c in led["llamadas"]] == ["cli explain"]


def test_pipeline_spark_down_still_holds_with_error_explanation(tmp_path):
    res = run_checkin("fixture", fixture_dir=FIX / "count_mismatch", receipts_dir=tmp_path / "r",
                      keys_dir=tmp_path / "k", explain="api", client_factory=factory(lambda q: httpx.Response(503)))
    r = json.loads(res.path.read_text())
    assert r["verdict"]["decision"] == "HOLD" and r["explanation"]["status"] == "ERROR" and verify_receipt(r)


def test_pipeline_accept_has_no_explanation_and_no_call(tmp_path):
    def h(req):
        raise AssertionError("model called on ACCEPT")

    res = run_checkin("fixture", fixture_dir=FIX / "accept", receipts_dir=tmp_path / "r", keys_dir=tmp_path / "k",
                      explain="api", client_factory=factory(h))
    assert res.verdict.decision == "ACCEPT" and "explanation" not in res.receipt
    assert not (tmp_path / "r" / "llamadas.json").exists()


def cli(tmp_path, *args):
    env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
    cmd = [sys.executable, "-m", "tinytms.cli", "checkin", "--receipts-dir", str(tmp_path / "r"),
           "--keys-dir", str(tmp_path / "k"), *args]
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    files = sorted((tmp_path / "r").glob("*-HOLD*.json"))
    return p, (json.loads(files[0].read_text()) if files else None)


def test_cli_explain_api_without_key_holds_with_no_model(tmp_path):
    p, r = cli(tmp_path, "--mode", "fixture", "--fixture-dir", str(FIX / "count_mismatch"), "--explain", "api",
               "--lang", "en")
    assert p.returncode == 2, p.stdout + p.stderr
    assert r["explanation"]["status"] == "NO_MODEL" and r["explanation"]["language"] == "en" and verify_receipt(r)
    assert "explanation[NO_MODEL]" in p.stdout


def test_cli_explain_defaults_to_none(tmp_path):
    p, r = cli(tmp_path, "--mode", "fixture", "--fixture-dir", str(FIX / "count_mismatch"))
    assert p.returncode == 2 and "explanation" not in r
    # argparse rejects an unknown language with its own exit code 2 and no receipt; check the message
    p, _ = cli(tmp_path / "x", "--mode", "fixture", "--fixture-dir", str(FIX / "accept"), "--lang", "fr")
    assert "invalid choice" in p.stderr and not (tmp_path / "x" / "r").exists()


# --- per-lane budget on the shared ledger ----------------------------------------------------


def test_lane_cap_counts_only_this_lanes_calls(tmp_path):
    log = tmp_path / "llamadas.json"
    old = {"tope": {}, "cuenta": {meta_api.SPARK_MODEL: 5},
           "llamadas": [{"modelo": meta_api.SPARK_MODEL, "utc": "x", "http": 200, "s": 1.0}] * 5}
    log.write_text(json.dumps(old))
    hits = []

    def h(req):
        hits.append(1)
        return httpx.Response(200, json=spark_reply())

    c = MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(h), call_log=log, caps={},
                   lane="lane-X", lane_caps={meta_api.SPARK_MODEL: 2})
    rs = [c.responses({"model": meta_api.SPARK_MODEL, "input": "x"}) for _ in range(3)]
    assert len(hits) == 2, "older records without a lane do not eat this lane's budget"
    assert rs[-1].status == "ERROR" and "lane budget cap" in rs[-1].detail
    d = json.loads(log.read_text())
    assert d["lanes"]["lane-X"] == {"cuenta": {meta_api.SPARK_MODEL: 2}, "tope": {meta_api.SPARK_MODEL: 2}}
    assert d["cuenta"][meta_api.SPARK_MODEL] == 7 and d["llamadas"][-1]["lane"] == "lane-X"


def test_lane_caps_from_env(monkeypatch):
    monkeypatch.setenv("TINYTMS_LANE", "L3")
    monkeypatch.setenv("TINYTMS_LANE_CAPS", "sam-3.1=20, muse-spark-1.3=25,muse-voice-transcribe-1.0=10")
    c = MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert c._lane == "L3" and c._lane_caps == {"sam-3.1": 20, "muse-spark-1.3": 25, "muse-voice-transcribe-1.0": 10}
    monkeypatch.setenv("TINYTMS_LANE_CAPS", "sam-3.1=lots")
    with pytest.raises(ValueError):
        MetaClient(api_key=FAKE_KEY)
