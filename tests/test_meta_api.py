"""Meta API client tests. Every response below is a FIXTURE served by httpx.MockTransport:
hand-written (in the shapes measured on 2026-10-28) to exercise our code, NOT a recorded Meta API output.
Recorded REAL outputs live in tests/data/ and are tested in test_real_data.py. No test touches the network."""

import base64
import hashlib
import json
import wave

import cv2
import httpx
import numpy as np
import pytest

from tinytms import meta_api
from tinytms.adapters import doc_from_spark, vision_from_sam, voice_from_transcript
from tinytms.meta_api import MetaClient, NoKey

FAKE_KEY = "test-key-FIXTURE-not-real"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to open a real HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)


@pytest.fixture
def img(tmp_path):
    """A real (decodable) 8x6 PNG drawn here: FIXTURE media, not a photo."""
    p = tmp_path / "load.png"
    ok, enc = cv2.imencode(".png", np.full((6, 8, 3), 127, dtype=np.uint8))
    p.write_bytes(enc.tobytes())
    return p


def client(handler, **kw):
    return MetaClient(api_key=FAKE_KEY, base_url="https://api.meta.ai/v1", transport=httpx.MockTransport(handler), **kw)


def test_nokey_without_env(monkeypatch, img):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(NoKey):
        MetaClient()
    with pytest.raises(NoKey):
        meta_api.sam_segment(img, "pallet")
    with pytest.raises(NoKey):
        meta_api.spark_extract_document(img)
    with pytest.raises(NoKey):
        meta_api.voice_transcribe(img)


def test_blank_key_is_nokey(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "   ")
    with pytest.raises(NoKey):
        MetaClient()


# FIXTURE SSE: hand-written in the measured event order; mask data is a short dummy string.
FIXTURE_SSE = "\n".join([
    "event: response.created",
    'data: {"type":"response.created","response":{"status":"in_progress","output":[]}}',
    "",
    'data: {"type":"response.output_text.delta","delta":"<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=8;h=6;c=0.9|>"}',
    "",
    'data: {"type":"response.output_text.done","text":"<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=8;h=6;c=0.9|>'
    '<|mask;x=0;y=0;c=0.9;data=FIXTUREFIXTURE|><|box;x1=4;y1=1;x2=7;y2=5;w=8;h=6;c=0.5|>'
    '<|mask;x=0;y=0;c=0.5;data=FIXTURE|>"}',
    "",
    'data: {"type":"response.completed","response":{"status":"completed","output":[]}}',
    "",
])


def test_sam_request_shape(img):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"], seen["url"] = req.method, str(req.url)
        seen["auth"], seen["accept"] = req.headers["authorization"], req.headers.get("accept")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=FIXTURE_SSE, headers={"content-type": "text/event-stream"})

    with client(handler) as c:
        r = c.sam_segment(img, "pallet")
    assert seen["method"] == "POST"
    assert seen["url"] == "https://api.meta.ai/v1/responses"
    assert seen["auth"] == f"Bearer {FAKE_KEY}"
    assert seen["accept"] == "text/event-stream"
    body = seen["body"]
    assert body["model"] == meta_api.SAM_MODEL == "sam-3.1"
    assert body["stream"] is True and body["metadata"] == {"include_confidence": "true"}
    assert body["input"][0]["type"] == "message" and body["input"][0]["role"] == "user"
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "pallet"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode()
    assert r.status == "OK" and r.provider == "meta-api" and r.data is None and r.raw_text
    assert "base64," not in json.dumps(r.request)  # the receipt copy carries no image payload
    assert r.media["original_sha256"] == hashlib.sha256(img.read_bytes()).hexdigest()
    w = vision_from_sam(r.raw_text, model=r.model, provider=r.provider, source_sha256="x")
    assert w.status == "OK" and w.count == 2
    assert w.boxes == ((1, 2, 3, 4, 0.9), (4, 1, 7, 5, 0.5)) and (w.image_w, w.image_h) == (8, 6)


def test_spark_doc_parsing(img):
    out = {"output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thinking {not json}"}]},
                      {"type": "message", "content": [{"type": "output_text", "text":
           '```json\n{"pieces": 12, "weight_kg": 8450.5, "seal": "MX-48213", "hazmat": false}\n```'}]}]}
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=out)

    with client(handler) as c:
        r = c.spark_extract_document(img)
    assert seen["body"]["max_output_tokens"] == meta_api.SPARK_MAX_OUTPUT_TOKENS >= 800
    assert "stream" not in seen["body"]
    assert [p["type"] for p in seen["body"]["input"][0]["content"]] == ["input_text", "input_image"]
    w = doc_from_spark(r.data, model=r.model, provider=r.provider, source_sha256="x")
    assert (w.status, w.pieces, w.weight_kg, w.seal, w.hazmat) == ("OK", 12, 8450.5, "MX-48213", False)


def test_spark_bad_types_is_error():
    w = doc_from_spark({"output_text": '{"pieces": "doce"}'}, model="m", provider="meta-api", source_sha256="x")
    assert w.status == "ERROR" and "pieces" in w.detail


def _wav(path, rate=16000, ch=1, sw=2, n=1600):
    with wave.open(str(path), "wb") as o:
        o.setnchannels(ch)
        o.setsampwidth(sw)
        o.setframerate(rate)
        o.writeframes(b"\x00" * n * ch * sw)
    return path


def test_voice_request_and_transcript(tmp_path):
    audio = _wav(tmp_path / "note.wav")
    seen = {}

    def handler(req):
        seen["url"], seen["ctype"], seen["raw"] = str(req.url), req.headers["content-type"], req.read()
        return httpx.Response(200, json={"sessionId": "FIXTURE", "audioDurationMs": 100, "turns": [],
                                         "transcript": "Traigo doce tarimas, el sello es MX-48213."})

    with client(handler) as c:
        r = c.voice_transcribe(audio)
    assert seen["url"] == "https://api.meta.ai/v1/asr/transcribe"
    assert seen["ctype"].startswith("multipart/form-data")
    assert b'name="request"' in seen["raw"] and b'name="audio"' in seen["raw"]
    assert b'"model": "muse-voice-transcribe-1.0"' in seen["raw"] and b'"mode": "PUSH_TO_TALK"' in seen["raw"]
    assert audio.read_bytes() in seen["raw"]  # already mono/16k: sent untouched
    w = voice_from_transcript(r.data, model=r.model, provider=r.provider, source_sha256="x")
    assert (w.count, w.seal, w.audio_duration_ms) == (12, "MX-48213", 100)


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_http_error_is_error_and_redacts_key(img, status):
    with client(lambda req: httpx.Response(status, text=f"nope {FAKE_KEY}")) as c:
        r = c.sam_segment(img)
    assert r.status == "ERROR" and r.http_status == status and r.data is None
    assert f"HTTP {status}" in r.detail
    assert FAKE_KEY not in r.detail


def test_timeout_is_error(img):
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    with client(handler) as c:
        r = c.spark_extract_document(img)
    assert r.status == "ERROR" and r.detail.startswith("timeout")


def test_transport_error_is_error(img):
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    with client(handler) as c:
        r = c.sam_segment(img)
    assert r.status == "ERROR" and "ConnectError" in r.detail


def test_non_json_is_error(img):
    with client(lambda req: httpx.Response(200, text="<html>")) as c:
        r = c.spark_extract_document(img)
    assert r.status == "ERROR" and "not JSON" in r.detail


def test_unknown_sam_shape_is_error_not_guess():
    w = vision_from_sam({"weird": 1}, model="m", provider="meta-api", source_sha256="x")
    assert w.status == "ERROR" and w.count is None
    w = vision_from_sam("data: {\"type\":\"response.created\"}\n", model="m", provider="meta-api", source_sha256="x")
    assert w.status == "ERROR" and "no output_text.done" in w.detail


def test_timeouts_are_explicit():
    t = meta_api.DEFAULT_TIMEOUT
    assert None not in (t.connect, t.read, t.write, t.pool)


def test_repr_hides_key():
    c = MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert FAKE_KEY not in repr(c)
