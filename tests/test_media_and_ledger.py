"""Image/WAV preparation, SAM text helpers, call ledger + budget cap, capture. All inputs are FIXTURE
media drawn or synthesised here; HTTP is httpx.MockTransport. No test touches the network or a camera."""

import hashlib
import json
import wave

import cv2
import httpx
import numpy as np
import pytest

from tinytms import capture, meta_api
from tinytms.adapters import call_record, sam_boxes, strip_masks, strip_sse, vision_from_sam
from tinytms.meta_api import MediaError, MetaClient, prepare_image, prepare_wav, redact_request

FAKE_KEY = "test-key-FIXTURE-not-real"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to open a real HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)


def _img(path, w, h, ext=".jpg"):
    ok, enc = cv2.imencode(ext, np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))
    path.write_bytes(enc.tobytes())
    return path


def _wav(path, rate=16000, ch=1, sw=2, n=1600):
    with wave.open(str(path), "wb") as o:
        o.setnchannels(ch)
        o.setsampwidth(sw)
        o.setframerate(rate)
        o.writeframes((np.arange(n * ch, dtype=np.int64) % 200).astype(f"<i{sw}" if sw > 1 else "u1").tobytes()
                      if sw in (1, 2, 4) else b"\x00" * n * ch * sw)
    return path


def test_large_image_is_downscaled_and_original_sha_kept(tmp_path):
    p = _img(tmp_path / "big.jpg", 2000, 1000)
    data, mime, info = prepare_image(p)
    assert mime == "image/jpeg" and info["resized"] is True
    assert info["original_wh"] == [2000, 1000] and info["sent_wh"] == [1280, 640]
    assert info["original_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest() != info["sent_sha256"]
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (640, 1280)


def test_small_png_is_sent_untouched(tmp_path):
    p = _img(tmp_path / "s.png", 900, 600, ".png")
    data, mime, info = prepare_image(p)
    assert (mime, info["resized"], data) == ("image/png", False, p.read_bytes())


def test_non_image_is_media_error(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"GIF89a-FIXTURE")
    with pytest.raises(MediaError):
        prepare_image(p)


def test_wav_16k_mono_passthrough_and_24k_kept(tmp_path):
    p = _wav(tmp_path / "a.wav")
    assert prepare_wav(p)[0] == p.read_bytes()
    q = _wav(tmp_path / "b.wav", rate=24000)
    out, info = prepare_wav(q)
    assert out == q.read_bytes() and info["converted"] is False


@pytest.mark.parametrize("rate,ch,sw", [(44100, 2, 2), (8000, 1, 1), (48000, 1, 4), (22050, 2, 3)])
def test_wav_is_converted_to_mono_16k_pcm16(tmp_path, rate, ch, sw):
    p = _wav(tmp_path / "c.wav", rate=rate, ch=ch, sw=sw, n=rate // 10)
    out, info = prepare_wav(p)
    assert info["converted"] is True and info["original"]["rate"] == rate
    (tmp_path / "out.wav").write_bytes(out)
    with wave.open(str(tmp_path / "out.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert abs(w.getnframes() - 1600) <= 1


def test_non_riff_audio_is_error_and_nothing_is_sent(tmp_path):
    p = tmp_path / "note.m4a"
    p.write_bytes(b"\x00\x00\x00\x20ftypM4A FIXTURE")
    called = []
    c = MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(lambda r: called.append(r) or httpx.Response(200)))
    r = c.voice_transcribe(p)
    assert r.status == "ERROR" and "RIFF" in r.detail and called == []


def test_redact_request_drops_base64():
    body = {"input": [{"content": [{"type": "input_image", "image_url": "data:image/png;base64,QUJDRA=="}]}]}
    red = redact_request(body)
    s = json.dumps(red)
    assert "QUJDRA" not in s and "data:image/png;base64 " in s
    assert body["input"][0]["content"][0]["image_url"].endswith("QUJDRA==")  # original untouched


def test_strip_masks_and_boxes():
    t = ("<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10;c=0.5|><|mask;x=0;y=0;c=0.5;data=AAAA;BBBB|>"
         ",1<|box;x1=5;y1=5;x2=9;y2=9;w=10;h=10;c=0.75|><|mask;x=0;y=0;c=0.75;data=CC|>")
    s = strip_masks(t)
    assert "AAAA" not in s and "CC|" not in s and s.count("data=<omitted") == 2
    assert [b["c"] for b in sam_boxes(s)] == [0.5, 0.75]


def test_strip_sse_drops_deltas_and_masks():
    sse = "\n".join([
        'data: {"type":"response.output_text.delta","delta":"<|mask;x=0;data=SECRETMASK"}',
        'data: {"type":"response.output_text.done","text":"<|box;x1=1;y1=1;x2=2;y2=2;w=3;h=3;c=1|>'
        '<|mask;x=0;y=0;c=1;data=SECRETMASK|>"}',
    ])
    s = strip_sse(sse)
    assert "SECRETMASK" not in s
    assert vision_from_sam(s, model="m", provider="meta-api", source_sha256="x").count == 1


def test_sam_stream_error_event_is_error():
    sse = 'data: {"type":"error","error":{"message":"FIXTURE boom"}}\n'
    w = vision_from_sam(sse, model="m", provider="meta-api", source_sha256="x")
    assert w.status == "ERROR" and "FIXTURE boom" in w.detail


def test_sam_empty_text_is_ok_zero():
    sse = 'data: {"type":"response.output_text.done","text":""}\ndata: {"type":"response.completed","response":{}}\n'
    w = vision_from_sam(sse, model="m", provider="meta-api", source_sha256="x")
    assert (w.status, w.count) == ("OK", 0)


def _png(tmp_path):
    return _img(tmp_path / "i.png", 8, 6, ".png")


def test_ledger_records_every_call_and_cap_blocks(tmp_path):
    log = tmp_path / "llamadas.json"
    hits = []

    def handler(req):
        hits.append(1)
        return httpx.Response(429, text=f"slow down {FAKE_KEY}")

    c = MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler), call_log=log,
                   caps={meta_api.SPARK_MODEL: 2}, origin="test")
    img = _png(tmp_path)
    for _ in range(3):
        r = c.spark_extract_document(img)
    assert len(hits) == 2  # third call refused before any request
    assert r.status == "ERROR" and "budget cap" in r.detail
    d = json.loads(log.read_text())
    assert d["cuenta"] == {meta_api.SPARK_MODEL: 2}
    assert [x["http"] for x in d["llamadas"]] == [429, 429]
    assert all(set(x) >= {"modelo", "utc", "http", "s"} for x in d["llamadas"])
    assert FAKE_KEY not in log.read_text()


def test_call_record_has_no_base64_and_truncates(tmp_path):
    big = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "x" * 5000}]}],
           "usage": {"output_tokens": 1}}
    c = MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=big)))
    rec = call_record(c.spark_extract_document(_png(tmp_path)))
    assert len(rec["raw"]) == 2000 and rec["usage"] == {"output_tokens": 1}
    assert "base64," not in json.dumps(rec) and rec["http"] == 200 and rec["s"] is not None


def test_load_image(tmp_path):
    p = _img(tmp_path / "a.jpg", 30, 20)
    im = capture.load_image(p)
    assert (im.width, im.height, im.mime) == (30, 20, "image/jpeg")
    assert im.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    bad = tmp_path / "b.png"
    bad.write_bytes(b"nope")
    with pytest.raises(capture.CaptureError):
        capture.load_image(bad)


def test_capture_without_camera_is_explicit_error(monkeypatch):
    class NoCam:  # FIXTURE stand-in for cv2.VideoCapture: never opens a real device
        def __init__(self, index):
            self.index = index

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", NoCam)
    with pytest.raises(capture.CaptureError, match="cannot be opened"):
        capture.capture_frame(0)
