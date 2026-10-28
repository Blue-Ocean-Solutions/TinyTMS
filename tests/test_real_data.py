"""Parsers against REAL Meta API answers recorded by scripts/live_eval.py (tests/data/real-*.json).

Each file carries "provenance": "REAL meta-api <utc>" and points at its signed receipt in receipts/.
Mask data and base64 were removed before writing; no key is stored. No test touches the network.
"""

import json
import os
import re
from pathlib import Path

import pytest

from tinytms.adapters import doc_from_spark, sam_text, parse_sse, vision_from_sam, voice_from_transcript
from tinytms.receipt import verify_receipt
from tinytms.verify import Limits, judge

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "tests" / "data"
FILES = sorted(DATA.glob("real-*.json"))
META = dict(provider="meta-api", source_sha256="x")


def load(kind):
    out = [json.loads(p.read_text(encoding="utf-8")) for p in FILES if p.name.startswith(f"real-{kind}-")]
    assert out, f"no tests/data/real-{kind}-*.json recorded"
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_real_files_are_labelled_and_clean(path):
    s = path.read_text(encoding="utf-8")
    d = json.loads(s)
    assert d["provenance"] == f"REAL meta-api {d['utc']}" and d["http"] == 200
    assert d["model"] in ("sam-3.1", "muse-spark-1.3", "muse-voice-transcribe-1.0")
    assert "base64," not in s
    assert not re.search(r"data=(?!<omitted)[^|]{50,}", s), "mask data left in test file"
    key = os.environ.get("MODEL_API_KEY", "")
    assert not key or key not in s
    receipt = REPO / "receipts" / d["receipt"]
    assert receipt.is_file() and verify_receipt(receipt.read_bytes())


def test_real_sam_parses_to_recorded_counts():
    for d in load("sam"):
        w = vision_from_sam(d["response_sse"], model=d["model"], concept=d["concept"], **META)
        text = sam_text(parse_sse(d["response_sse"]))
        assert w.status == "OK"
        assert w.count == text.count("<|box;") == d["witness_at_record_time"]["count"]
        assert [list(b) for b in w.boxes] == d["witness_at_record_time"]["boxes"]
        if w.count:
            assert (w.image_w, w.image_h) == tuple(d["media"]["sent_wh"])
            for x1, y1, x2, y2, c in w.boxes:
                assert 0 <= x1 < x2 <= w.image_w and 0 <= y1 < y2 <= w.image_h and 0 < c <= 1


def test_real_sam_zero_records_is_ok_count_zero():
    zero = [d for d in load("sam") if d["witness_at_record_time"]["count"] == 0]
    assert zero, "expected at least one real zero-detection answer"
    for d in zero:
        w = vision_from_sam(d["response_sse"], model=d["model"], concept=d["concept"], **META)
        assert (w.status, w.count, w.boxes) == ("OK", 0, ())


def test_real_spark_reads_the_synthetic_carta():
    ok = [d for d in load("spark") if d["response_json"].get("status") == "completed"]
    assert ok
    for d in ok:
        w = doc_from_spark(d["response_json"], model=d["model"], **META)
        assert (w.status, w.pieces, w.weight_kg, w.seal, w.hazmat) == ("OK", 12, 4800.0, "MX-48213", False)
        assert d["response_json"]["usage"]["output_tokens_details"]["reasoning_tokens"] > 0


def test_real_spark_incomplete_is_error():
    inc = [d for d in load("spark") if d["response_json"].get("status") == "incomplete"]
    for d in inc:  # recorded with max_output_tokens 800: reasoning ate the budget, JSON cut
        w = doc_from_spark(d["response_json"], model=d["model"], **META)
        assert w.status == "ERROR" and "incomplete" in w.detail and w.pieces is None


def test_real_voice_silence_is_empty_and_holds():
    for d in (x for x in load("voice") if x["input"]["name"] == "silencio_16k.wav"):
        assert set(d["response_json"]) >= {"sessionId", "transcript", "audioDurationMs", "turns"}
        w = voice_from_transcript(d["response_json"], model=d["model"], **META)
        assert (w.status, w.transcript, w.count, w.seal) == ("OK", "", None, None)
        assert w.detail == "empty transcript" and w.audio_duration_ms == d["response_json"]["audioDurationMs"]


def real_voice_es():
    out = [d for d in load("voice") if d["input"]["name"] == "es-doce-pallets-16k.wav"]
    assert out, "no real transcript of evals/voice/es-doce-pallets-16k.wav recorded"
    return out


# SYNTHETIC-TTS note (evals/PROCEDENCIA.md), said: "Traigo doce pallets de madera. El sello es M X guión cuatro
# ocho dos uno tres." Transcribed three times on 2026-10-28, it came back three ways (the ASR is not deterministic;
# this is what it WROTE). Expected parse per transcript: units "payets"/"pajets"/"payettes", "guión" is the seal's
# dash, a letter group repeated back to back is one stutter. "SMX" (the ASR fused "es MX") is NOT guessed back to
# "MX": that seal reads SMX-48213 and the judge HOLDs on SEAL_MISMATCH. Fail closed. The weight sentence after the
# period never leaks into the seal.
VOICE_ES_EXPECTED = {
    "Traigo doce payets de madera. El sello es MX guión cuatro ocho dos uno tres.": (12, "MX-48213"),
    "Traigo doce pajets de madera. El sello es MX MX guión cuatro ocho dos uno tres.": (12, "MX-48213"),
    "Traigo doce payettes de madera. El sello SMX SMX guión cuatro ocho dos uno tres.": (12, "SMX-48213"),
}


def test_real_voice_es_parses_count_and_seal():
    seen = set()
    for d in real_voice_es():
        t = d["response_json"]["transcript"]
        (key,) = [k for k in VOICE_ES_EXPECTED if k in t]
        seen.add(key)
        w = voice_from_transcript(d["response_json"], model=d["model"], **META)
        assert (w.status, w.count, w.seal) == ("OK", *VOICE_ES_EXPECTED[key]), t
        assert w.audio_duration_ms == d["response_json"]["audioDurationMs"]
    assert seen == set(VOICE_ES_EXPECTED), "every recorded variant is covered by a real file"


def test_seal_stutter_collapse_is_letters_only():
    from tinytms.adapters import parse_seal

    assert parse_seal("el sello es MX MX guión cuatro ocho dos uno tres.") == "MX-48213"
    assert parse_seal("el sello es MX cuatro cuatro ocho dos.") == "MX4482"  # repeated digits are digits
    assert parse_seal("el sello es MX guión.") is None  # no digits: no seal


def test_judge_on_real_es_voice_and_spark_agree_on_count_and_seal():
    spark = next(d for d in load("spark") if d["response_json"].get("status") == "completed")
    sam = next(d for d in load("sam") if d["input"]["name"] == "boxes-pallet.jpg" and d["concept"] == "pallet")
    v = vision_from_sam(sam["response_sse"], model=sam["model"], concept="pallet", **META)
    doc = doc_from_spark(spark["response_json"], model=spark["model"], **META)
    for d in real_voice_es():
        a = voice_from_transcript(d["response_json"], model=d["model"], **META)
        verdict = judge(v, doc, a, Limits())
        if a.seal == "SMX-48213":
            assert verdict.reasons == ("COUNT_MISMATCH doc=12 vision=1 voice=12", "SEAL_MISMATCH doc=MX48213 voice=SMX48213")
        else:  # paper and voice agree on count and seal; only the photo disagrees
            assert verdict.reasons == ("COUNT_MISMATCH doc=12 vision=1 voice=12",) and verdict.checks["seal"] == "pass"


def test_judge_on_real_witnesses_holds():
    sam = next(d for d in load("sam") if d["input"]["name"] == "boxes-pallet.jpg" and d["concept"] == "pallet")
    spark = next(d for d in load("spark") if d["response_json"].get("status") == "completed")
    voice = next(d for d in load("voice") if d["input"]["name"] == "silencio_16k.wav")
    v = vision_from_sam(sam["response_sse"], model=sam["model"], concept="pallet", **META)
    doc = doc_from_spark(spark["response_json"], model=spark["model"], **META)
    a = voice_from_transcript(voice["response_json"], model=voice["model"], **META)
    verdict = judge(v, doc, a, Limits())
    assert verdict.decision == "HOLD"
    assert "VOICE_NO_COUNT" in " ".join(verdict.reasons)
