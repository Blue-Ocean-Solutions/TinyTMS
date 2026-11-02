"""Lowercase seals: the voice parser against what the local Meta ASR writes (ORDEN Nube-7, 2026-11-02).

The local ASR (Omnilingual CTC) writes lowercase Spanish with no punctuation and numbers as words:
"el sello es mx guión cuatro ocho dos uno tres". The parser took a seal's letter group only when it was UPPERCASE, so
that seal came back None with an empty detail: lost, and nothing said why. The rule now, in seal position only: a
lowercase group of up to 3 ASCII letters is a seal's letters when a dash or digits follow it, and it comes out
uppercased. Function words ("es", "de", "y", ...) never are. A seal said with no id read always says why, and the
judge HOLDs on it.
"""

from __future__ import annotations

import random
import re

import pytest

from tests.test_real_data import VOICE_ES_EXPECTED
from tests.test_voice_variantes import CASES, ES_0_29, es_words
from tinytms.adapters import parse_seal, voice_from_transcript
from tinytms.verify import Limits, judge, normalize_seal
from tinytms.witness import DocWitness, VisionWitness

# The paper side as the real Spark read of the synthetic Carta de Porte pins it (tests/test_real_data.py).
V = VisionWitness(status="OK", model="FIXTURE", provider="FIXTURE", count=12)
D = DocWitness(status="OK", model="FIXTURE", provider="FIXTURE", pieces=12, weight_kg=4800.0, seal="MX-48213",
               hazmat=False)
L = Limits(max_weight_kg=24000)


def witness(text: str):
    return voice_from_transcript({"transcript": text}, model="prueba", provider="none", source_sha256="")


def asr(text: str) -> str:
    """A sentence as the local ASR writes it: lowercase, no punctuation."""
    return " ".join(re.sub(r"[.,;:!?…]", " ", text.lower()).split())


def test_the_reproduced_transcript_reads_the_seal():
    w = witness("traigo doce tarimas de madera el sello es mx guión cuatro ocho dos uno tres")
    assert (w.count, w.seal, w.detail) == (12, "MX-48213", "")
    assert judge(V, D, w, L).decision == "ACCEPT"


@pytest.mark.parametrize("text", VOICE_ES_EXPECTED, ids=["payets", "pajets-mx-mx", "payettes-smx"])
def test_the_three_real_transcripts_as_the_local_asr_writes_them(text):
    # The same three real transcripts, lowercased and without punctuation: same count, same seal. "smx smx" stays
    # SMX-48213 (not guessed back to MX) and the judge HOLDs on it, as in the pinned uppercase original.
    w = witness(asr(text))
    assert (w.count, w.seal) == VOICE_ES_EXPECTED[text], (asr(text), w.detail)


# --- (a) lowercase gives what uppercase gives ----------------------------------------------------------------------

ORDER_A = "doce bultos sello {mx} {dash} cuatro ocho dos uno tres cuatro mil ochocientos kilos sin material peligroso"


@pytest.mark.parametrize("dash", ["guión", "guion"])
def test_lowercase_sentence_gives_what_the_uppercase_one_gives(dash):
    low, up = witness(ORDER_A.format(mx="mx", dash=dash)), witness(ORDER_A.format(mx="MX", dash=dash))
    assert (low.count, low.seal, low.detail) == (up.count, up.seal, up.detail)
    assert low.count == 12
    vl, vu = judge(V, D, low, L), judge(V, D, up, L)
    assert vl == vu
    assert (vl.checks["count"], vl.checks["weight"], vl.checks["hazmat"]) == ("pass", "pass", "pass")
    # Without a stop between the seal and the weight neither form reads a seal: "tres cuatro mil ochocientos kilos"
    # is ...3 and 4800 kg, or ...34 and 1800 kg ("mil ochocientos"); the parser does not guess. It says so and HOLDs.
    if low.seal is None:
        assert "seal" in low.detail and vl.decision == "HOLD" and "SEAL_MISSING voice" in vl.reasons


@pytest.mark.parametrize("dash", ["guión", "guion"])
def test_lowercase_sentence_with_a_stop_after_the_seal_accepts(dash):
    w = witness(f"doce bultos. sello mx {dash} cuatro ocho dos uno tres. cuatro mil ochocientos kilos, sin material "
                "peligroso.")
    assert (w.count, w.seal) == (12, "MX-48213"), w.detail
    v = judge(V, D, w, L)
    assert v.decision == "ACCEPT" and v.reasons == ()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_every_corpus_case_lowercased_reads_the_same_seal(case):
    up, low = witness(case["text"]), witness(case["text"].lower())
    assert normalize_seal(low.seal) == normalize_seal(up.seal), (case["text"].lower(), low.detail)


def test_generated_seals_in_lowercase():
    rng = random.Random("nube-7:seal")
    for n in [0, 7, 21, 100, 1000, 48213, 999999] + [rng.randint(0, 999999) for _ in range(500)]:
        text = f"el sello es mx guión {es_words(n)} peso cuatro mil ochocientos kilos"
        assert parse_seal(text) == f"MX-{n}", text
    for _ in range(500):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(3, 8)))
        text = f"el sello es mx raya {' '.join(ES_0_29[int(d)] for d in digits)} traigo doce tarimas"
        assert parse_seal(text) == f"MX-{digits}", text


# --- (b) the forms a seal comes in ---------------------------------------------------------------------------------


@pytest.mark.parametrize("said", ["mx-48213", "mx 48213", "MX guión 4 8 2 1 3", "mx guión 4 8 2 1 3",
                                  "mx guion cuatro ocho dos uno tres", "m x guión cuatro ocho dos uno tres"])
def test_seal_forms_normalise_to_the_paper(said):
    for text in (f"traigo doce tarimas el sello es {said}", f"el sello es {said} traigo doce tarimas"):
        w = witness(text)
        assert normalize_seal(w.seal) == "MX48213", (text, w.seal, w.detail)
        assert w.count == 12, (text, w.detail)


# --- (c) function words and look-alikes ----------------------------------------------------------------------------

STOP = "es el la los las de del un una y o no en con sin que se su mi tu al lo".split()


@pytest.mark.parametrize("word", STOP)
def test_a_function_word_is_never_seal_letters(word):
    for text in (f"el sello {word} cuatro ocho dos uno tres", f"el sello {word} guión cuatro ocho dos uno tres",
                 f"el sello es mx guión cuatro ocho {word} dos uno tres traigo doce tarimas"):
        w = witness(text)
        letters = re.sub(r"[^A-Z]", "", normalize_seal(w.seal) or "")
        assert letters in ("", "MX"), (text, w.seal)
        assert w.seal is not None or "seal" in w.detail, text


def test_el_sello_es_de_plastico_is_no_seal():
    w = witness("el sello es de plástico")
    assert w.seal is None and "seal" in w.detail


@pytest.mark.parametrize("said,seal", [
    ("smx guión cuatro ocho dos uno tres", "SMX-48213"),  # read as said, never guessed back to MX: HOLD
    ("mxñ guión cuatro ocho dos uno tres", None),  # Ñ is not an ASCII seal letter
    ("мх guión cuatro ocho dos uno tres", None),  # Cyrillic em + ha
])
def test_look_alikes_are_not_read_as_mx(said, seal):
    w = witness(f"traigo doce tarimas el sello es {said}")
    assert w.seal == seal
    v = judge(V, D, w, L)
    assert v.decision == "HOLD" and v.checks["seal"] == "fail"
    assert w.seal is not None or "seal" in w.detail


# --- never silent --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "el sello es de plástico",
    "el sello está roto traigo doce tarimas",
    "doce tarimas sello mx",
    "doce tarimas el sello es mxñ guión cuatro ocho dos uno tres",
])
def test_a_seal_said_without_an_id_says_why_and_holds(text):
    w = witness(text)
    assert w.seal is None and "seal" in w.detail, w.detail
    v = judge(V, D, w, L)
    assert v.decision == "HOLD" and "SEAL_MISSING voice" in v.reasons


def test_a_mention_without_an_id_does_not_hide_one_with_an_id():
    w = witness("el sello es de plástico y el sello es mx guión cuatro ocho dos uno tres")
    assert w.seal == "MX-48213"


def test_a_correction_in_lowercase_is_not_resolved_to_the_first_seal():
    w = witness("el sello es mx guión cuatro ocho dos uno tres perdón mx guión cuatro ocho dos tres uno")
    assert w.seal is None and "seal" in w.detail


LOW_VOCAB = ("sello seal es el de la y o no un una mx smx mxñ m x guión guion raya dash cuatro ocho dos uno tres "
             "doce mil ochocientos kilos tarimas bultos 48213 mx-48213 perdón sin con madera plástico roto eh").split()


def test_fuzz_a_seal_keyword_followed_by_words_is_never_silent():
    rng = random.Random("nube-7:fuzz")
    for _ in range(4000):
        words = [rng.choice(LOW_VOCAB) for _ in range(rng.randint(1, 14))]
        text = " ".join(words)
        w = witness(text)
        assert w.status == "OK"
        assert w.seal is None or any(ch.isdigit() for ch in w.seal), text
        said = any(x in ("sello", "seal") for x in words[:-1])
        if said and w.seal is None:
            assert "seal" in w.detail, text
