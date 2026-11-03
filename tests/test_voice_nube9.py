"""The voice witness reads less and never guesses (ORDER Nube-9, 2026-11-03).

House rule: the voice witness returns the right value, or None with a non-empty reason. It never picks one of two
readings and never returns a wrong value, with or without a reason. Every phrase of the order is pinned here with its
exact count, seal and detail, and the judge's decision against the paper (12 pieces, MX-48213, 4800 kg, no hazmat).
tests/test_voice_words.py has one case per word of every word list; tests/voice_notes.py is the seeded generator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.voice_notes import CLASSES, accepts, classify, forms, notes
from tinytms.adapters import _parse_voice, from_fixture, voice_from_transcript
from tinytms.verify import Limits, judge
from tinytms.witness import DocWitness, VisionWitness

V = VisionWitness(status="OK", model="FIXTURE", provider="FIXTURE", count=12)
D = DocWitness(status="OK", model="FIXTURE", provider="FIXTURE", pieces=12, weight_kg=4800.0, seal="MX-48213",
               hazmat=False)
L = Limits(max_weight_kg=24000)
ROOT = Path(__file__).resolve().parent.parent


def witness(text: str):
    return voice_from_transcript({"transcript": text}, model="prueba", provider="none", source_sha256="")


def check(text: str, count, seal, detail: str, decision: str, doc: DocWitness = D) -> None:
    w = witness(text)
    assert (w.count, w.seal, w.detail) == (count, seal, detail), text
    assert judge(V, doc, w, L).decision == decision, text


WORDS = "words between the seal and its id"

# --- 1. no read-past: any word between the keyword and the id but a lead-in gives None and the reason ----------------

READ_PAST = [
    ("Traigo doce tarimas. El sello, según la carta, es MX-48213.",
     12, None, f"{WORDS}: 'sello , según la carta , es mx-48213'"),
    ("Traigo doce tarimas. El sello de la carta es MX-48213.", 12, None, f"{WORDS}: 'sello de la carta es mx-48213'"),
    ("Twelve pallets. The seal on the paperwork is MX-48213.",
     12, None, f"{WORDS}: 'seal on the paperwork is mx-48213'"),
    ("Doce tarimas. El sello de ayer era MX-48213.", 12, None, f"{WORDS}: 'sello de ayer era mx-48213'"),
    ("Traigo doce tarimas con sello, la factura es 48213.", 12, None, f"{WORDS}: 'sello , la factura es 48213'"),
    ("Twelve pallets, seal is fine, trailer number is 53.",
     12, None, f"{WORDS}: 'seal is fine , trailer number is 53'"),
    ("The seal on trailer number four is intact.", None, None,
     f"no count found in transcript; {WORDS}: 'seal on trailer number four'; "
     "seal digits run into the next words: 'four is'"),
    ("Son doce tarimas. El sello lo vi en la uno.", 12, None, f"{WORDS}: 'sello lo vi en la uno'"),
    # ORDER Nube-8's fix-7 examples: the house rule outranks that order
    ("Traigo doce tarimas. El sello de la caja es MX-48213.", 12, None, f"{WORDS}: 'sello de la caja es mx-48213'"),
    ("Traigo doce tarimas. El sello que traigo es el MX-48213.",
     12, None, f"{WORDS}: 'sello que traigo es el mx-48213'"),
]
# what stays from fix 7: "doble u" is W; a plural, and a keyword said last, give a reason
FIX_7_KEPT = [
    ("Traigo doce tarimas. El sello doble u equis cuatro ocho dos uno tres.", 12, "WX48213", ""),
    ("Traigo doce tarimas, los sellos son MX-48213 y MX-48214.",
     12, None, "several seals said in transcript: 'sellos'"),
    ("Traigo doce tarimas y el sello.", 12, None, "seal said but no id read: 'sello .'"),
]


@pytest.mark.parametrize("text,count,seal,detail", READ_PAST)
def test_1_an_id_after_other_words_is_never_read(text, count, seal, detail):
    check(text, count, seal, detail, "HOLD")


@pytest.mark.parametrize("text,count,seal,detail", FIX_7_KEPT)
def test_1_what_stays_from_fix_7(text, count, seal, detail):
    check(text, count, seal, detail, "HOLD")
    if seal:
        check(text, count, seal, detail, "ACCEPT", doc=DocWitness(**{**D.to_dict(), "seal": "WX-48213"}))


@pytest.mark.parametrize("said", ["El sello es", "Sello", "Sello número", "El sello es el", "Sello No.", "Sello:",
                                  "El sello de seguridad es", "Seal number", "The seal is", "Seal is the"])
def test_1_lead_ins_still_read(said):
    check(f"Traigo doce tarimas. {said} MX-48213.", 12, "MX-48213", "", "ACCEPT")


# --- 2. hedged seals ------------------------------------------------------------------------------------------------

HEDGED_SEALS = [
    ("Traigo doce tarimas. El sello creo que es MX-48213.",
     12, None, "hedged seal in transcript: 'sello creo que es mx-48213'"),
    ("Twelve pallets. I think the seal is MX-48213.", 12, None, "hedged seal in transcript: 'i think the seal'"),
    ("Traigo doce tarimas. Sello, creo, MX-48213.", 12, None, "hedged seal in transcript: 'sello , creo , mx-48213'"),
    ("Traigo doce tarimas. El sello es MX-48213, creo.",
     12, None, "hedged seal in transcript: 'sello es mx-48213 , creo'"),
    ("Twelve pallets. The seal is MX-48213, I think.",
     12, None, "hedged seal in transcript: 'seal is mx-48213 , i think'"),
]


@pytest.mark.parametrize("text,count,seal,detail", HEDGED_SEALS)
def test_2_a_hedged_seal_is_none_like_a_hedged_count(text, count, seal, detail):
    check(text, count, seal, detail, "HOLD")


# --- 3. two values for the same field anywhere in the note ---------------------------------------------------------

TWO_VALUES = [
    ("Twelve pallets, nope, uh, thirteen total. Seal MX-48213.",
     None, "MX-48213", "counts disagree in transcript: [12, 13] (self-correction: 'nope')"),
    ("Twelve pallets, uh, thirteen total. Seal MX-48213.", None, "MX-48213", "counts disagree in transcript: [12, 13]"),
    ("Twelve pallets. The seal is MX-48213. Sorry, MX-48231.",
     12, None, "seals disagree in transcript: ['MX48213', 'MX48231']"),
    ("Twelve pallets. The seal is MX-48213. MX-48231.",
     12, None, "seals disagree in transcript: ['MX48213', 'MX48231']"),
    # a count correction right after the seal is never read as a second seal
    ("Twelve pallets, seal MX-48213, no, thirteen.", None, None,
     "counts disagree in transcript: [12, 13] (self-correction: 'no'); "
     "correction right after the seal: 'no , thirteen'"),
    ("Traigo doce tarimas, sello MX-48213, perdón, trece.", None, None,
     "counts disagree in transcript: [12, 13] (self-correction: 'perdón'); "
     "correction right after the seal: 'perdón , trece'"),
    ("Twelve pallets, seal MX-48213, sorry, thirteen pallets.",
     None, "MX-48213", "counts disagree in transcript: [12, 13]"),
    ("twelve pallets seal mx-48213 no thirteen", None, None,
     "counts disagree in transcript: [12, 13] (self-correction: 'no'); correction right after the seal: 'no thirteen'"),
]


@pytest.mark.parametrize("text,count,seal,detail", TWO_VALUES)
def test_3_two_values_disagree(text, count, seal, detail):
    check(text, count, seal, detail, "HOLD")


# --- 4. seal digits followed by a number word ----------------------------------------------------------------------

DIGITS_THEN_A_NUMBER = [
    ("doce bultos sello mx guion cuatro ocho dos uno tres cuatro mil ochocientos sin material peligroso",
     12, None, "seal digits run into a number: 'mx guion cuatro ocho dos uno tres cuatro mil ochocientos sin'"),
    ("doce bultos sello mx guion cuatro ocho dos uno tres cuatro mil ochocientos es el peso",
     12, None, "seal digits run into the next words: 'cuatro ocho dos uno tres cuatro mil ochocientos es'"),
    ("doce tarimas sello cuatro ocho dos uno tres cuatro es el andén",
     12, None, "seal digits run into the next words: 'cuatro ocho dos uno tres cuatro es'"),
]


@pytest.mark.parametrize("text,count,seal,detail", DIGITS_THEN_A_NUMBER)
def test_4_digits_then_a_number_word_never_give_a_longer_id(text, count, seal, detail):
    check(text, count, seal, detail, "HOLD")


# --- 5. never drop or invent a letter ------------------------------------------------------------------------------


def test_5_the_letter_after_the_dash_is_kept():
    text = "got twelve pallets seal m x dash four eight two one three dash b"
    check(text, 12, "MX-48213-B", "", "HOLD")  # the paper says MX-48213: a different seal, not a guess
    check(text, 12, "MX-48213-B", "", "ACCEPT", doc=DocWitness(**{**D.to_dict(), "seal": "MX-48213-B"}))


@pytest.mark.parametrize("text,seal,detail", [
    ("Twelve pallets, seal M X dash four eight two one three dash.", None,
     "seal ends in a dash: 'm x dash four eight two one three dash .'"),
    ("Twelve pallets, seal mx 48213 b.", None, "unclear seal letter after the digits: 'mx 48213 b'"),
    ("Doce tarimas, sello MX guión cuatro ocho dos uno tres guión be.", "MX-48213-B", ""),
    ("Doce tarimas, sello ce a guión cuatro ocho dos uno tres.", "CA-48213", ""),
])
def test_5_a_letter_is_read_or_the_seal_is_none_with_the_reason(text, seal, detail):
    w = witness(text)
    assert (w.count, w.seal, w.detail) == (12, seal, detail)


# --- 6. no false HOLDs: each inside an otherwise good note ----------------------------------------------------------

ES = "Traigo doce tarimas. El sello es MX-48213."
ES_REST = "Pesa cuatro mil ochocientos kilos, sin material peligroso."
EN = "Twelve pallets. The seal is MX-48213."
EN_REST = "Four thousand eight hundred kilos, no hazmat."
LOW = "traigo doce tarimas pesa cuatro mil ochocientos kilos sin material peligroso el sello es mx guión " \
      "cuatro ocho dos uno tres"
LOW_SEAL = "traigo doce tarimas el sello es mx guión cuatro ocho dos uno tres"
FALSE_HOLDS = [
    f"{ES} {ES_REST} Bueno, un saludo.",
    f"Seal MX-48213, got all twelve pallets. {EN_REST}",
    f"Seal MX-48213, we got twelve pallets. {EN_REST}",
    f"{ES} {ES_REST} Todo bien, no falta nada.",
    f"Traigo doce tarimas, el sello es MX-48213, todo bien, no falta nada. {ES_REST}",
    f"{EN} It matches the paperwork. {EN_REST}",
    f"Twelve pallets, seal MX-48213, it matches the paperwork. {EN_REST}",
    f"{ES} El sello coincide. {ES_REST}",
    f"Traigo doce tarimas, el sello es MX-48213, el sello coincide. {ES_REST}",
    f"{EN} The seal is not broken. {EN_REST}",
    f"Twelve pallets, the seal is MX-48213, the seal is not broken. {EN_REST}",
    f"{LOW_SEAL} pesa cuatro mil ochocientos kilos sin material peligroso",
    f"{LOW_SEAL} son cuatro mil ochocientos kilos sin material peligroso",
    f"{LOW_SEAL} ahí van cuatro mil ochocientos kilos sin material peligroso",
    f"{LOW} buenas tardes",
    f"{LOW} muchas gracias",
    f"{LOW} nada más",
    f"{LOW} saludos",
    "twelve pallets seal m x dash four eight two one three weighs four thousand eight hundred kilos no hazmat",
]


@pytest.mark.parametrize("text", FALSE_HOLDS)
def test_6_a_good_note_accepts(text):
    check(text, 12, "MX-48213", "", "ACCEPT")


@pytest.mark.parametrize("denial,detail", [
    ("El sello no coincide.", "seal reported broken, missing or not matching in transcript: 'sello no coincide'"),
    ("The seal is broken.", "seal reported broken, missing or not matching in transcript: 'seal is broken'"),
    ("Falta el sello.", "seal reported broken, missing or not matching in transcript: 'falta el sello'"),
    ("The seal doesn't match.",
     "seal reported broken, missing or not matching in transcript: 'seal doesn t match'"),
])
def test_6_a_denial_still_holds(denial, detail):
    check(f"{ES} {denial}", 12, None, detail, "HOLD")


# --- 8. the real transcripts and the demo lines give what they gave on db6c9cf --------------------------------------

REAL = {  # tests/test_real_data.py pins them too
    "Traigo doce payets de madera. El sello es MX guión cuatro ocho dos uno tres.": (12, "MX-48213", ""),
    "Traigo doce pajets de madera. El sello es MX MX guión cuatro ocho dos uno tres.": (12, "MX-48213", ""),
    "Traigo doce payettes de madera. El sello SMX SMX guión cuatro ocho dos uno tres.": (12, "SMX-48213", ""),
}
DEMO = {  # fixtures/*/voice.FIXTURE.json (scripts/make_fixtures.py): (count, seal) as db6c9cf read them
    "accept": (12, "MX-48213"), "count_mismatch": (12, "MX-48213"), "seal_mismatch": (12, "MX-48231"),
    "overweight": (12, "MX-48213"),
}


def test_8_the_real_transcripts_read_as_on_db6c9cf():
    seen = set()
    for f in sorted((ROOT / "tests" / "data").glob("real-voice-es-*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))["response_json"]["transcript"]
        (key,) = [k for k in REAL if k in t]
        seen.add(key)
        w = witness(t)
        assert (w.count, w.seal, w.detail) == REAL[key], t
    assert seen == set(REAL)


@pytest.mark.parametrize("name", sorted(DEMO))
def test_8_the_demo_lines_read_as_on_db6c9cf(name):
    w = from_fixture(ROOT / "fixtures" / name / "voice.FIXTURE.json")
    assert (w.status, w.count, w.seal) == ("OK", *DEMO[name])
    assert w.detail.startswith("fixture voice.FIXTURE.json sha256=") and ";" not in w.detail, w.detail


# --- 9. the seeded generator: no wrong value, no pick, no silent None ----------------------------------------------


def test_9_generated_notes_never_get_a_wrong_value():
    all_notes = forms(notes())
    assert len(all_notes) >= 6000 and sum(n.good for n in all_notes) >= 600
    by_form = {False: dict.fromkeys(CLASSES, 0), True: dict.fromkeys(CLASSES, 0)}
    accepted = {False: 0, True: 0}
    for nt in all_notes:
        parsed = _parse_voice(nt.text)
        for cls in classify(nt, parsed):
            by_form[nt.low][cls] += 1
        if nt.good:
            accepted[nt.low] += accepts(nt, parsed)
    for low, c in by_form.items():
        assert (c["wrong"], c["picked"], c["none_empty"]) == (0, 0, 0), (low, c)
    good = sum(n.good for n in all_notes) // 2
    assert accepted[False] / good >= 0.97, accepted
