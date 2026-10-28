"""Meta API client tests. Every response below is a FIXTURE served by httpx.MockTransport:
hand-written to exercise our parsing, NOT a recorded Meta API output. No test touches the network."""

import base64
import json

import httpx
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
    p = tmp_path / "load.png"
    p.write_bytes(b"\x89PNG-FIXTURE")
    return p


def client(handler):
    return MetaClient(api_key=FAKE_KEY, base_url="https://api.meta.ai/v1", transport=httpx.MockTransport(handler))


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


def test_sam_request_shape(img):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"], seen["url"] = req.method, str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"detections": [{"bbox": [0, 0, 1, 1]}, {"bbox": [1, 1, 2, 2]}]})

    with client(handler) as c:
        r = c.sam_segment(img, "pallet")
    assert seen["method"] == "POST"
    assert seen["url"] == "https://api.meta.ai/v1/responses"
    assert seen["auth"] == f"Bearer {FAKE_KEY}"
    body = seen["body"]
    assert body["model"] == meta_api.SAM_MODEL
    content = body["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"] == "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode()
    assert content[1] == {"type": "input_text", "text": "pallet"}
    assert r.status == "OK" and r.provider == "meta-api"
    w = vision_from_sam(r.data, model=r.model, provider=r.provider, source_sha256="x")
    assert w.status == "OK" and w.count == 2 and w.boxes == ((0, 0, 1, 1), (1, 1, 2, 2))


def test_spark_doc_parsing(img):
    out = {"output": [{"type": "message", "content": [{"type": "output_text", "text":
           '```json\n{"pieces": 12, "weight_kg": 8450.5, "seal": "MX-48213", "hazmat": false}\n```'}]}]}

    with client(lambda req: httpx.Response(200, json=out)) as c:
        r = c.spark_extract_document(img)
    w = doc_from_spark(r.data, model=r.model, provider=r.provider, source_sha256="x")
    assert (w.status, w.pieces, w.weight_kg, w.seal, w.hazmat) == ("OK", 12, 8450.5, "MX-48213", False)


def test_spark_bad_types_is_error():
    w = doc_from_spark({"output_text": '{"pieces": "doce"}'}, model="m", provider="meta-api", source_sha256="x")
    assert w.status == "ERROR" and "pieces" in w.detail


def test_voice_request_and_transcript(tmp_path):
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"RIFF-FIXTURE")
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"text": "Traigo doce tarimas, el sello es MX-48213."})

    with client(handler) as c:
        r = c.voice_transcribe(audio)
    part = seen["body"]["input"][0]["content"][0]
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
    w = voice_from_transcript(r.data, model=r.model, provider=r.provider, source_sha256="x")
    assert (w.count, w.seal) == (12, "MX-48213")


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
        r = c.sam_segment(img)
    assert r.status == "ERROR" and "not JSON" in r.detail


def test_unknown_sam_shape_is_error_not_guess():
    w = vision_from_sam({"weird": 1}, model="m", provider="meta-api", source_sha256="x")
    assert w.status == "ERROR" and w.count is None


def test_timeouts_are_explicit():
    t = meta_api.DEFAULT_TIMEOUT
    assert None not in (t.connect, t.read, t.write, t.pool)


def test_repr_hides_key():
    c = MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert FAKE_KEY not in repr(c)
