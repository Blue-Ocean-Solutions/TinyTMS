"""colleagues: Meta models asked about a check-in the judge already decided, behind a firewall.
They answer; they never decide. Every model answer here is a FIXTURE served by httpx.MockTransport. No network."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from tinytms.colleagues import COLLEAGUES, FirewallMissing, consult, firewall, load_firewall, receipt_body
from tinytms.meta_api import DEFAULT_CAPS, MetaClient
from tinytms.verify import judge
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness

REPO = Path(__file__).resolve().parent.parent
PHOTO = REPO / "evals" / "public" / "boxes-moving.jpg"
FAKE_KEY = "test-key-FIXTURE-not-real"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to open a real HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)


def reply(text, status="completed"):
    body = {"status": status, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 1, "output_tokens": 2}}
    if status == "incomplete":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
        body["output"] = []
    return body


def factory(text="CONSISTENT", seen=None, status="completed"):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply(text, status))

    def make(**kw):
        return MetaClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler), **kw)

    return make


def witnesses(seal_voice="MX-48213"):
    v = VisionWitness(status="OK", model="sam-3.1", provider="meta-api", source_sha256="a" * 64, count=0,
                      concept="pallet")
    d = DocWitness(status="OK", model="muse-spark-1.3", provider="meta-api", source_sha256="b" * 64, pieces=12,
                   weight_kg=4800.0, seal="MX-48213", hazmat=False)
    a = VoiceWitness(status="OK", model="muse-voice-transcribe-1.0", provider="meta-api", source_sha256="c" * 64,
                     count=12, seal=seal_voice)
    return v, d, a


@pytest.fixture
def words(tmp_path):
    p = tmp_path / "firewall.txt"
    p.write_text("# local list, never committed\nZorro-Azul\nproyecto x\n", encoding="utf-8")
    return load_firewall(p)


def test_the_firewall_list_is_local_and_its_absence_stops_everything(tmp_path, words):
    assert words == ["zorro-azul", "proyecto x"]
    with pytest.raises(FirewallMissing):
        load_firewall(tmp_path / "nope.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(FirewallMissing):
        load_firewall(empty)


def test_without_a_list_no_colleague_is_called(tmp_path):
    seen = []
    v, d, a = witnesses()
    c = consult("wonder", judge(v, d, a), v, d, a, firewall_words=None, client_factory=factory(seen=seen))
    assert c.status == "REFUSED" and "no firewall list" in c.detail and seen == []


def test_a_listed_word_stops_the_call_and_is_never_echoed(words):
    seen = []
    v, d, a = witnesses(seal_voice="ZORRO-AZUL-48213")  # a voice field that carries a listed word
    c = consult("wonder", judge(v, d, a), v, d, a, firewall_words=words, client_factory=factory(seen=seen))
    assert c.status == "REFUSED" and "firewall list entry 1" in c.detail and seen == []
    record = json.dumps(c.to_dict()).lower()
    assert "zorro" not in record and c.prompt == ""  # the refused prompt is kept only as its sha256
    assert len(c.prompt_sha256) == 64


@pytest.mark.parametrize("text, why", [
    ("token sk-abcdefgh12345678 here", "key-shaped"),
    ("a hf_abcdefghijklmnop token", "key-shaped"),
    ("see /Users/someone/notes.txt", "machine path"),
])
def test_key_shapes_and_machine_paths_never_go_out(text, why):
    assert any(why in r for r in firewall(text, []))


def test_a_clean_audit_goes_out_once_and_its_answer_is_only_advice(words):
    seen = []
    v, d, a = witnesses()
    verdict = judge(v, d, a)
    c = consult("wonder", verdict, v, d, a, firewall_words=words,
                client_factory=factory("ACCEPT. Everything matches.", seen))
    assert c.status == "OK" and c.model == "muse-wonder-1.0" and len(seen) == 1
    assert seen[0]["model"] == "muse-wonder-1.0"
    sent = json.dumps(seen[0])
    assert "COUNT_MISMATCH" in sent and "MX-48213" in sent  # the judge's reasons and the witnesses' fields
    body = receipt_body("colleague", source_name="x-HOLD.json", source_raw=b"{}", source_key="k", verdict=verdict,
                        consult=c, command="python scripts/ask_colleagues.py")
    assert body["verdict"]["decision"] == "HOLD"  # a colleague saying ACCEPT changes nothing
    assert body["colleague"]["answer"] == "ACCEPT. Everything matches."


def test_spark_1_1_sees_only_the_receipts_own_photo(words, tmp_path):
    seen = []
    v, d, a = witnesses()
    verdict = judge(v, d, a)
    other = tmp_path / "other.jpg"
    other.write_bytes(PHOTO.read_bytes() + b"\x00")
    c = consult("spark-1.1", verdict, v, d, a, firewall_words=words, photo=other,
                photo_sha256=hashlib.sha256(PHOTO.read_bytes()).hexdigest(), client_factory=factory("0", seen))
    assert c.status == "REFUSED" and "photo" in c.detail and seen == []
    c = consult("spark-1.1", verdict, v, d, a, firewall_words=words, photo=PHOTO,
                photo_sha256=hashlib.sha256(PHOTO.read_bytes()).hexdigest(), client_factory=factory("0", seen))
    assert c.status == "OK" and c.count == 0 and len(seen) == 1
    content = seen[0]["input"][0]["content"]
    assert any(p.get("type") == "input_image" for p in content)


@pytest.mark.parametrize("text, count", [("3", 3), ("There are 2 pallets.", 2), ("none that I can see", None)])
def test_a_count_is_read_from_the_answer_or_left_none(words, text, count):
    v, d, a = witnesses()
    c = consult("spark-1.1", judge(v, d, a), v, d, a, firewall_words=words, photo=PHOTO,
                photo_sha256=hashlib.sha256(PHOTO.read_bytes()).hexdigest(), client_factory=factory(text))
    assert c.status == "OK" and c.count == count
    if count is None:
        assert "no integer" in c.detail


def test_an_incomplete_answer_is_an_error_and_still_only_advice(words):
    v, d, a = witnesses()
    c = consult("wonder", judge(v, d, a), v, d, a, firewall_words=words,
                client_factory=factory(status="incomplete"))
    assert c.status == "ERROR" and "incomplete" in c.detail


def test_no_key_means_no_model(words, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    v, d, a = witnesses()
    c = consult("wonder", judge(v, d, a), v, d, a, firewall_words=words)
    assert c.status == "NO_MODEL"


def test_every_colleague_has_a_call_cap():
    assert set(COLLEAGUES) == {"wonder", "spark-1.1"}
    for col in COLLEAGUES.values():
        assert DEFAULT_CAPS.get(col.model, 0) > 0, col.model


def test_an_unknown_colleague_is_an_error(words):
    v, d, a = witnesses()
    with pytest.raises(ValueError):
        consult("someone", judge(v, d, a), v, d, a, firewall_words=words)
