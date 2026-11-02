"""The voice witness fails closed, never guesses (ORDER Nube-8, 2026-11-02).

House rule: the voice witness returns the right value, or None with a non-empty reason. It never picks one of two
readings and never returns a wrong value with an empty reason. Every phrase of the order is also a case of the
SYNTHETIC corpus (tests/data/voice-es-variantes.json, ids "*-n8-<fix>-*") with its exact expected count and seal; this
file adds what the corpus cannot say: the judge's decision against the paper, and variants generated around them.
"""

from __future__ import annotations

import pytest

from tests.test_voice_variantes import CASES
from tinytms.verify import Limits, judge
from tinytms.adapters import voice_from_transcript
from tinytms.witness import DocWitness, VisionWitness

# The paper side as the real Spark read of the synthetic Carta de Porte pins it (tests/test_real_data.py).
V = VisionWitness(status="OK", model="FIXTURE", provider="FIXTURE", count=12)
D = DocWitness(status="OK", model="FIXTURE", provider="FIXTURE", pieces=12, weight_kg=4800.0, seal="MX-48213",
               hazmat=False)
L = Limits(max_weight_kg=24000)
N8 = {c["id"]: c for c in CASES if "-n8-" in c["id"]}


def witness(text: str):
    return voice_from_transcript({"transcript": text}, model="prueba", provider="none", source_sha256="")


def fix(n: int) -> list[dict]:
    return [c for c in N8.values() if c["id"].split("-")[2] == str(n)]


# --- 1. no weight split: a seal running into a weight with no stop HOLDs, with a stop it ACCEPTs ---------------------


def test_fix_1_cases_are_in_the_corpus():
    assert {c["id"] for c in fix(1)} == {"es-n8-1-peso-sin-punto", "es-n8-1-peso-con-punto"}


def test_fix_1_unpunctuated_holds_and_punctuated_accepts():
    w = witness(N8["es-n8-1-peso-sin-punto"]["text"])
    v = judge(V, D, w, L)
    assert (w.count, w.seal) == (12, None) and v.decision == "HOLD" and v.reasons == ("SEAL_MISSING voice",)
    w = witness(N8["es-n8-1-peso-con-punto"]["text"])
    v = judge(V, D, w, L)
    assert (w.count, w.seal, w.detail) == (12, "MX-48213", "") and v.decision == "ACCEPT" and v.reasons == ()


@pytest.mark.parametrize("kg", ["cuatro mil ochocientos", "mil ochocientos", "tres mil", "4800", "ocho mil"])
def test_fix_1_no_weight_after_seal_digits_ever_splits(kg):
    w = witness(f"doce bultos sello mx guion cuatro ocho dos uno tres {kg} kilos")
    assert w.seal is None and w.detail.startswith("seal digits run into a weight: "), w.detail


# --- 2. lowercase letters only before the first digit; after the id a word ends the seal ----------------------------

AFTER_ID = ["got", "hmm", "ah", "pos", "eh", "mmm", "bye", "pa", "ok", "uh", "so", "the", "I", "i", "and"]


def test_fix_2_cases_are_in_the_corpus():
    assert {c["id"] for c in fix(2)} >= {"en-n8-2-got", "en-n8-2-hmm", "en-n8-2-ah", "es-n8-2-pos", "en-n8-2-seal-is"}


@pytest.mark.parametrize("word", AFTER_ID)
@pytest.mark.parametrize("seal", ["MX-48213", "mx-48213", "mx 48213", "mx guión cuatro ocho dos uno tres"])
def test_fix_2_a_word_after_the_id_ends_the_seal(seal, word):
    for text in (f"Seal {seal}, {word}, twelve pallets.", f"seal {seal} {word} twelve pallets"):
        w = witness(text)
        assert w.count == 12, (text, w.detail)
        if w.seal is not None:  # read: the paper's seal, never one with the word's letters glued on
            assert judge(V, D, w, L).decision == "ACCEPT", (text, w.seal)
        else:
            assert "seal" in w.detail, text
    w = witness(f"Seal {seal}, {word}, twelve pallets.")
    assert judge(V, D, w, L).decision == "ACCEPT", (w.seal, w.detail)


@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_fix_2_seals_is_not_smx(apostrophe):
    for text in (f"The seal{apostrophe}s MX-48213.", f"the seal{apostrophe}s mx 48213", f"The seal{apostrophe}s M X dash "
                 "four eight two one three, twelve pallets."):
        w = witness(text)
        assert w.seal is not None and w.seal.replace("-", "").upper() == "MX48213", (text, w.seal, w.detail)


@pytest.mark.parametrize("text", ["the seal s MX-48213", "el sello hoy mx 48213", "el sello es ab mx guión 48213"])
def test_fix_2_letters_that_do_not_read_one_way_give_a_reason(text):
    w = witness(text)
    assert w.seal is None and "unclear seal letters" in w.detail, (w.seal, w.detail)


@pytest.mark.parametrize("said,seal", [("m x guión 48213", "MX-48213"), ("mx mx guión 48213", "MX-48213"),
                                       ("smx smx guión 48213", "SMX-48213"), ("mx guión 48213", "MX-48213")])
def test_fix_2_prefixes_that_read_one_way_still_read(said, seal):
    assert witness(f"el sello es {said}").seal == seal
