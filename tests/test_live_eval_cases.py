"""scripts/live_eval.py cases: SAM runs on boxes-moving.jpg only; the voice-es case takes a Spanish note in
evals/voice/es-*.wav, and when there is none (the SYNTHETIC-TTS note was withdrawn for its license) it is skipped and
the script says so. No call is made here: the only key is a placeholder and no job ever reaches the API."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("live_eval", ROOT / "scripts" / "live_eval.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_cases_use_boxes_moving_only():
    jobs, skipped = _script().plan(ROOT / "evals", {"sam", "spark", "voice"})
    assert {p.name for kind, p, _, _ in jobs if kind == "sam"} == {"boxes-moving.jpg"}
    assert [p.name for kind, p, _, _ in jobs if kind == "voice"] == ["silencio_16k.wav"]
    assert skipped == []


def test_voice_es_without_a_spanish_note_is_skipped_and_said(tmp_path):
    (tmp_path / "voice").mkdir()
    jobs, skipped = _script().plan(tmp_path, {"voice-es"})
    assert jobs == []
    assert len(skipped) == 1 and skipped[0].startswith("voice-es: skipped") and "evals/PROCEDENCIA.md" in skipped[0]


def test_voice_es_takes_a_spanish_note_when_there_is_one(tmp_path):
    (tmp_path / "voice").mkdir()
    (tmp_path / "voice" / "es-chofer-16k.wav").write_bytes(b"RIFF")
    jobs, skipped = _script().plan(tmp_path, {"voice-es"})
    assert [(kind, p.name, tag) for kind, p, _, tag in jobs] == [("voice", "es-chofer-16k.wav", "es-chofer-16k-")]
    assert skipped == []


def test_the_tree_carries_no_spanish_tts_note():
    assert not list((ROOT / "evals" / "voice").glob("es-*.wav"))


def test_sam_own_without_a_photo_is_skipped_and_said(tmp_path):
    jobs, skipped = _script().plan(tmp_path, {"sam-own"})
    assert jobs == []
    assert len(skipped) == 1 and skipped[0].startswith("sam-own: skipped")


def test_sam_own_takes_each_own_photo_with_each_concept(tmp_path):
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "screens.jpg").write_bytes(b"\xff\xd8\xff\xdb")
    jobs, skipped = _script().plan(tmp_path, {"sam-own"}, own_concepts=("television", "tablet"))
    assert [(kind, p.name, c, tag) for kind, p, c, tag in jobs] == [
        ("sam", "screens.jpg", "television", "screens-television-"),
        ("sam", "screens.jpg", "tablet", "screens-tablet-"),
    ]
    assert skipped == []


def test_own_photos_carry_no_exif():
    # A phone photo carries EXIF, location included. Receipts and the repo are public: own photos go in clean.
    for photo in sorted((ROOT / "evals" / "own").glob("*.jpg")):
        assert b"Exif\x00\x00" not in photo.read_bytes(), f"{photo.name} still carries EXIF"


def test_cli_prints_the_skip_and_calls_nothing(tmp_path):
    (tmp_path / "evals" / "voice").mkdir(parents=True)
    env = {**os.environ, "MODEL_API_KEY": "FIXTURE-not-a-key"}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "live_eval.py"), "--evals", str(tmp_path / "evals"),
                        "--only", "voice-es", "--receipts-dir", str(tmp_path / "r"), "--keys-dir", str(tmp_path / "k")],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert "voice-es: skipped" in p.stdout and "0/0 calls OK" in p.stdout
    assert not (tmp_path / "r" / "llamadas.json").exists()  # nothing reached the ledger
