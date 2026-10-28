"""Web page tests: call the WSGI app directly (wsgiref.util.setup_testing_defaults). No port, no network.
Model answers, where any, are FIXTURE responses from httpx.MockTransport."""

import io
import json
import socket
import uuid
from pathlib import Path
from wsgiref.util import setup_testing_defaults

import httpx
import pytest

from tinytms.meta_api import MetaClient
from tinytms.receipt import find_machine_paths, verify_receipt
from tinytms.web import make_app, parse_multipart

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "fixtures"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)
    monkeypatch.setattr(socket.socket, "bind", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fname, data, ctype) in files.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\n"
                  f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    out.write(f"--{b}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={b}"


def call(app, method="GET", path="/", body=b"", ctype="", extra=None):
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "wsgi.input": io.BytesIO(body),
           "CONTENT_LENGTH": str(len(body)), "CONTENT_TYPE": ctype}
    setup_testing_defaults(env)  # HTTP_HOST / SERVER_NAME = 127.0.0.1
    env.update(extra or {})
    got = {}

    def start(status, headers):
        got["status"], got["headers"] = status, dict(headers)

    data = b"".join(app(env, start))
    return got["status"], got["headers"], data


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path / "receipts", tmp_path / "keys", FIX)


@pytest.fixture
def app_api(tmp_path):
    """A server started with --allow-api: the page may spend Meta API credits."""
    return make_app(tmp_path / "receipts", tmp_path / "keys", FIX, allow_api=True)


def test_negative_content_length_is_400_and_body_is_not_read(app):
    stream = io.BytesIO(b"x" * 64)
    status, _, _ = call(app, "POST", "/checkin", b"", "multipart/form-data; boundary=x",
                        extra={"wsgi.input": stream, "CONTENT_LENGTH": "-1"})
    assert status == "400 Bad Request"
    assert stream.tell() == 0  # nothing was read past the cap


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "10.0.0.5:8765", "127.0.0.1.evil.example"])
def test_foreign_host_header_is_403_dns_rebinding(app, tmp_path, host):
    assert call(app, "GET", "/", extra={"HTTP_HOST": host})[0] == "403 Forbidden"
    body, ctype = multipart({"mode": "fixture", "fixture": "accept"}, {})
    assert call(app, "POST", "/checkin", body, ctype, extra={"HTTP_HOST": host})[0] == "403 Forbidden"
    assert not list((tmp_path / "receipts").glob("*.json"))


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:8765", "localhost:8765", "[::1]:8765"])
def test_loopback_hosts_are_served(app, host):
    assert call(app, "GET", "/", extra={"HTTP_HOST": host})[0] == "200 OK"


@pytest.mark.parametrize("hdr", [{"HTTP_ORIGIN": "http://evil.example"}, {"HTTP_ORIGIN": "http://127.0.0.1.evil.example:8765"},
                                 {"HTTP_REFERER": "http://evil.example/attack.html"}, {"HTTP_ORIGIN": "null"}])
def test_cross_site_post_is_403_csrf(app, tmp_path, hdr):
    body, ctype = multipart({"mode": "fixture", "fixture": "accept"}, {})
    status, _, page = call(app, "POST", "/checkin", body, ctype, extra=hdr)
    assert status == "403 Forbidden", page
    assert not list((tmp_path / "receipts").glob("*.json"))


@pytest.mark.parametrize("hdr", [{"HTTP_ORIGIN": "http://127.0.0.1"}, {"HTTP_ORIGIN": "http://127.0.0.1:8765"},
                                 {"HTTP_REFERER": "http://localhost:8765/"}, {}])
def test_same_origin_or_headerless_post_is_served(app, hdr):
    body, ctype = multipart({"mode": "fixture", "fixture": "accept"}, {})
    assert call(app, "POST", "/checkin", body, ctype, extra=hdr)[0] == "200 OK"


@pytest.mark.parametrize("fields", [{"mode": "api"}, {"mode": "auto"}, {"mode": "fixture", "fixture": "accept", "explain": "api"}])
def test_paid_modes_need_explicit_opt_in(app, tmp_path, fields):
    body, ctype = multipart(fields, {})
    status, _, page = call(app, "POST", "/checkin", body, ctype)
    assert status == "403 Forbidden" and b"--allow-api" in page
    assert not list((tmp_path / "receipts").glob("*.json"))


def test_allow_api_lets_the_page_use_paid_modes(app_api):
    body, ctype = multipart({"mode": "fixture", "fixture": "count_mismatch", "explain": "api"}, {})
    status, _, page = call(app_api, "POST", "/checkin", body, ctype)
    assert status == "200 OK" and b"NO_MODEL" in page  # no key in tests: explanation NO_MODEL, HOLD stands


def test_get_form(app):
    status, h, body = call(app)
    page = body.decode()
    assert status == "200 OK" and h["Content-Type"].startswith("text/html")
    for name in ('name="load"', 'name="doc"', 'name="voice"', 'name="mode"', 'name="explain"', 'name="lang"',
                 'enctype="multipart/form-data"', 'action="/checkin"'):
        assert name in page, name
    for mode in ("fixture", "api", "local", "auto"):
        assert f'<option value="{mode}"' in page
    assert "<style>" in page and "<script" not in page  # CSS inline, no external JS
    assert '<option value="accept" selected>' in page


def test_post_fixture_accept(app, tmp_path):
    body, ctype = multipart({"mode": "fixture", "fixture": "accept", "explain": "none", "lang": "es"}, {})
    status, _, page = call(app, "POST", "/checkin", body, ctype)
    page = page.decode()
    assert status == "200 OK", page
    assert 'class="v ACCEPT"' in page and "FIXTURE" in page
    (rp,) = (tmp_path / "receipts").glob("*-ACCEPT-FIXTURE.json")
    assert f'href="/receipts/{rp.name}"' in page
    r = json.loads(rp.read_text())
    assert verify_receipt(r) and r["origin"] == "web" and r["verdict"]["decision"] == "ACCEPT"
    # the receipt link serves the signed receipt byte for byte
    s, h, served = call(app, "GET", f"/receipts/{rp.name}")
    assert s == "200 OK" and h["Content-Type"].startswith("application/json") and served == rp.read_bytes()


def test_post_fixture_hold_shows_reasons_and_explanation_status(app_api):
    body, ctype = multipart({"mode": "fixture", "fixture": "count_mismatch", "explain": "api", "lang": "en"}, {})
    status, _, page = call(app_api, "POST", "/checkin", body, ctype)
    page = page.decode()
    assert status == "200 OK" and 'class="v HOLD"' in page
    assert "COUNT_MISMATCH doc=12 vision=11 voice=12" in page
    assert "Explicación al chofer" in page and "NO_MODEL" in page  # no key in tests: explanation NO_MODEL


def test_uploaded_files_are_hashed_and_named_safely(app, tmp_path):
    wav = (FIX / "accept" / "voice.FIXTURE.wav").read_bytes()
    png = (FIX / "accept" / "load.FIXTURE.png").read_bytes()
    body, ctype = multipart({"mode": "fixture", "fixture": "accept"},
                            {"load": ("../../etc/mi foto<1>.png", png, "image/png"),
                             "voice": ("nota.wav", wav, "audio/wav"), "doc": ("", b"", "application/octet-stream")})
    status, _, page = call(app, "POST", "/checkin", body, ctype)
    assert status == "200 OK", page
    (rp,) = (tmp_path / "receipts").glob("*.json")
    r = json.loads(rp.read_text())
    import hashlib

    assert r["inputs"]["load"] == {"name": "mi_foto_1_.png", "sha256": hashlib.sha256(png).hexdigest()}
    assert r["inputs"]["voice"]["sha256"] == hashlib.sha256(wav).hexdigest()
    assert r["inputs"]["doc"]["name"] == "doc.FIXTURE.png"  # empty file input -> fixture media
    assert find_machine_paths(r) == []


def test_api_mode_through_web_uses_the_same_pipeline(tmp_path):
    def h(req):  # every API call fails: HOLD with ERROR witnesses, never an ACCEPT
        return httpx.Response(503, text="FIXTURE unavailable")

    app = make_app(tmp_path / "receipts", tmp_path / "keys", FIX, allow_api=True,
                   client_factory=lambda **kw: MetaClient(api_key="FIXTURE-key", transport=httpx.MockTransport(h), **kw))
    a = FIX / "accept"
    files = {"load": ("l.png", (a / "load.FIXTURE.png").read_bytes(), "image/png"),
             "doc": ("d.png", (a / "doc.FIXTURE.png").read_bytes(), "image/png"),
             "voice": ("v.wav", (a / "voice.FIXTURE.wav").read_bytes(), "audio/wav")}
    body, ctype = multipart({"mode": "api"}, files)
    status, _, page = call(app, "POST", "/checkin", body, ctype)
    assert status == "200 OK" and 'class="v HOLD"' in page.decode()
    (rp,) = (tmp_path / "receipts").glob("*-HOLD.json")
    r = json.loads(rp.read_text())
    assert r["verdict"]["reasons"][0] == "ERROR:vision" and r["mode"] == "api"
    assert [c["origen"] for c in json.loads((tmp_path / "receipts" / "llamadas.json").read_text())["llamadas"]] == \
        ["web checkin"] * 3


def test_api_mode_without_files_is_400(app_api):
    body, ctype = multipart({"mode": "api"}, {})
    status, _, page = call(app_api, "POST", "/checkin", body, ctype)
    assert status == "400 Bad Request" and b"--load is required" in page


@pytest.mark.parametrize("fields", [{"mode": "nope"}, {"mode": "fixture", "fixture": "../fixtures/accept"},
                                    {"mode": "fixture", "fixture": "accept", "lang": "fr"},
                                    {"mode": "fixture", "fixture": "accept", "explain": "maybe"}])
def test_bad_form_values_are_400(app, fields):
    body, ctype = multipart(fields, {})
    assert call(app, "POST", "/checkin", body, ctype)[0] == "400 Bad Request"


def test_not_multipart_and_too_large(tmp_path):
    app = make_app(tmp_path / "r", tmp_path / "k", FIX, max_upload_bytes=100)
    assert call(app, "POST", "/checkin", b"mode=fixture", "application/x-www-form-urlencoded")[0] == "400 Bad Request"
    body, ctype = multipart({"mode": "fixture", "fixture": "accept", "pad": "x" * 200}, {})
    assert call(app, "POST", "/checkin", body, ctype)[0] == "413 Payload Too Large"
    assert call(app, "GET", "/checkin")[0] == "405 Method Not Allowed"


@pytest.mark.parametrize("name", ["..%2Fkeys%2Fed25519.key", "../keys/ed25519.key", "..json", "a..b.json",
                                  "x.key", "sub/x.json", "nope.json", "%2e%2e.json", ""])
def test_receipt_route_refuses_unsafe_or_missing_names(app, tmp_path, name):
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "ed25519.key").write_text("FIXTURE-secret")
    status, _, body = call(app, "GET", "/receipts/" + name)
    assert status == "404 Not Found" and b"FIXTURE-secret" not in body


def test_unknown_path_404(app):
    assert call(app, "GET", "/etc/passwd")[0] == "404 Not Found"


def test_parse_multipart_keeps_binary_bytes():
    data = bytes(range(256)) * 4 + b"\r\n--not-a-boundary\r\n" + b"\x00\xff"
    body, ctype = multipart({"mode": "api", "lang": "es"}, {"voice": ("v.wav", data, "audio/wav")})
    fields, files = parse_multipart(body, ctype)
    assert fields == {"mode": "api", "lang": "es"} and files["voice"] == ("v.wav", data)


def test_html_escapes_model_text(tmp_path):
    def h(req):
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "FIXTURE <script>alert(1)</script>"}]}]})

    app = make_app(tmp_path / "r", tmp_path / "k", FIX, allow_api=True,
                   client_factory=lambda **kw: MetaClient(api_key="FIXTURE-key", transport=httpx.MockTransport(h), **kw))
    body, ctype = multipart({"mode": "fixture", "fixture": "count_mismatch", "explain": "api"}, {})
    page = call(app, "POST", "/checkin", body, ctype)[2].decode()
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
