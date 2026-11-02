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


# --- 3. self-corrections with words in between ----------------------------------------------------------------------

BETWEEN = ["", ", eh,", ", este,", " eh", ", uh,", " espérame", " a ver", ", a ver,", " hmm", ", pues,", " oye"]


def test_fix_3_cases_are_in_the_corpus():
    assert len(fix(3)) == 11


@pytest.mark.parametrize("between", BETWEEN)
@pytest.mark.parametrize("marker", ["no", "perdón", "digo", "mejor", "me equivoqué", "sorry", "make it", "I mean"])
def test_fix_3_a_correction_with_words_between_never_keeps_either_count(marker, between):
    for a, b in ((12, 13), (13, 12), (4, 40)):
        from tests.test_voice_variantes import en_words, es_words
        import random
        wa, wb = es_words(a, True), es_words(b, True)
        for text in (f"Traigo {wa} tarimas, {marker}{between} {wb}.", f"Traigo {wa} tarimas {marker}{between} {wb}",
                     f"Traigo {wa} tarimas, {marker}{between} {wb} en total.",
                     f"{en_words(a, random.Random(a))} pallets, {marker}{between} {en_words(b, random.Random(b))}."):
            w = witness(text)
            assert w.count is None, (text, w.detail)
            assert f"ambiguous counts in transcript: {sorted({a, b})}" in w.detail and "self-correction" in w.detail, (
                text, w.detail)


@pytest.mark.parametrize("chain", ["no, perdón,", "no, eh, perdón,", "me equivoqué, es", "perdón, a ver,", "sorry, it's",
                                   "no, no, perdón, es el", "mejor dicho,", "o sea, eh,"])
@pytest.mark.parametrize("second", ["MX cuatro ocho dos tres uno", "cuatro ocho dos tres uno", "MX-48231"])
def test_fix_3_a_corrected_seal_is_never_resolved(chain, second):
    for first in ("MX cuatro ocho dos uno tres", "MX-48213", "mx guión cuatro ocho dos uno tres"):
        text = f"Traigo doce tarimas. El sello es {first}, {chain} {second}."
        w = witness(text)
        assert w.seal is None and "seals disagree in transcript" in w.detail, (text, w.seal, w.detail)
        assert w.count == 12, (text, w.detail)
        assert judge(V, D, w, L).reasons == ("SEAL_MISSING voice",)


def test_fix_3_the_same_seal_said_twice_across_a_correction_still_reads():
    w = witness("El sello es MX cuatro ocho dos uno tres, no, perdón, MX cuatro ocho dos uno tres.")
    assert w.seal == "MX48213"


def test_fix_3_accepted_cost_a_correction_before_any_number_makes_it_compete():
    # A correction before a number makes it compete whatever follows it, so "trece en total" and "trece completas"
    # count. The cost, accepted: a number that counts something else competes too. None and the reason, never 12.
    w = witness("Traigo doce tarimas, perdón, tres horas tarde.")
    assert w.count is None and w.detail == "ambiguous counts in transcript: [3, 12] (self-correction: 'perdón')"


# --- 4. negated, broken or missing seals ---------------------------------------------------------------------------

DENIED = [  # {s}: the seal as said
    "el sello no es {s}", "el sello {s} viene roto", "no traigo el sello {s}", "falta el sello {s}",
    "seal {s} is broken", "seal {s} is missing", "el sello {s} no coincide", "el sello es {s}, no coincide",
    "the seal is not {s}", "the seal isn't {s}", "there is no seal {s}", "el sello {s} está dañado", "sin sello {s}",
    "the seal {s} doesn't match", "seal {s} was tampered", "el sello {s} lo traigo abierto", "el sello {s} no corresponde",
    "missing seal {s}", "broken seal {s}", "el sello está roto, {s}",
]
SAID = ["MX-48213", "MX guión cuatro ocho dos uno tres", "mx guion cuatro ocho dos uno tres", "cuatro ocho dos uno tres"]


def test_fix_4_cases_are_in_the_corpus():
    assert len(fix(4)) == 13


@pytest.mark.parametrize("template", DENIED)
@pytest.mark.parametrize("said", SAID)
def test_fix_4_a_denied_seal_is_never_read(template, said):
    for text in (f"Traigo doce tarimas, {template.format(s=said)}.", f"{template.format(s=said)}, traigo doce tarimas."):
        w = witness(text)
        assert w.seal is None, (text, w.seal)
        assert "negated seal" in w.detail or "seal reported broken, missing or not matching" in w.detail, (text, w.detail)
        assert w.count == 12, (text, w.detail)
        assert judge(V, D, w, L).reasons == ("SEAL_MISSING voice",)


@pytest.mark.parametrize("said", ["sello No. MX-48213", "sello no. MX-48213", "sello Nº MX-48213", "sello nro. MX-48213",
                                  "seal No. MX-48213", "el sello número MX-48213"])
def test_fix_4_the_abbreviation_no_still_reads(said):
    w = witness(f"Traigo doce tarimas, {said}.")
    assert (w.count, w.seal, w.detail) == (12, "MX-48213", ""), w.detail
    assert judge(V, D, w, L).decision == "ACCEPT"


@pytest.mark.parametrize("text,seal,why", [
    # a negation after a comma is about something else
    ("Traigo doce tarimas, el sello es MX-48213, no traigo material peligroso.", "MX-48213", ""),
    ("doce bultos. sello mx guion cuatro ocho dos uno tres. sin material peligroso.", "MX-48213", ""),
    # accepted cost: with no comma a negation in the 6 words after the id denies it (None and the reason, never a value)
    ("Traigo doce tarimas, el sello es MX-48213 y no hay material peligroso.", None,
     "negated seal in transcript: 'sello es mx-48213 y no'"),
])
def test_fix_4_a_negation_about_something_else(text, seal, why):
    w = witness(text)
    assert (w.count, w.seal, w.detail) == (12, seal, why)


# --- 5. hedged counts ----------------------------------------------------------------------------------------------

HEDGED = ["creo que son {n} tarimas", "creo que traigo {n} tarimas", "tal vez {n} tarimas", "talvez son {n} tarimas",
          "quizás {n} tarimas", "a lo mejor son {n} tarimas", "supongo que son {n} tarimas", "me parece que son {n} cajas",
          "probablemente {n} bultos", "{n} tarimas, creo", "{n} tarimas, creo yo", "{n} tarimas, supongo"]
HEDGED_EN = ["maybe {n} pallets", "I think {n} pallets", "I think it's {n} pallets", "probably {n} pallets",
             "perhaps {n} boxes", "I guess {n} pallets", "it might be {n} pallets", "{n} pallets, I think",
             "{n} pallets, maybe", "I believe we have {n} pallets"]


def test_fix_5_cases_are_in_the_corpus():
    assert len(fix(5)) == 7


@pytest.mark.parametrize("n", [1, 12, 24, 100])
def test_fix_5_a_hedged_count_is_none_the_same_as_unas_doce(n):
    from tests.test_voice_variantes import en_words, es_words
    import random
    assert witness(f"Traigo unas {es_words(n, True)} tarimas.").count is None  # what it already did
    for t in HEDGED:
        text = t.format(n=es_words(n, True)).capitalize() + "."
        w = witness(text)
        assert w.count is None and w.detail.startswith("hedged count in transcript: "), (text, w.detail)
        assert "VOICE_NO_COUNT" in judge(V, D, w, L).reasons
    for t in HEDGED_EN:
        text = t.format(n=en_words(n, random.Random(n))).capitalize() + "."
        w = witness(text)
        assert w.count is None and w.detail.startswith("hedged count in transcript: "), (text, w.detail)


@pytest.mark.parametrize("text", ["Traigo doce tarimas. Creo que el sello es MX-48213.",
                                  "Creo que ya llegué. Traigo doce tarimas.", "Traigo doce tarimas, el sello creo que es "
                                  "MX-48213."])
def test_fix_5_a_hedge_about_something_else_keeps_the_count(text):
    assert witness(text).count == 12


# --- 6. a clause glued onto the seal's digits ----------------------------------------------------------------------

GLUED = ["doce paquetes", "dos horas esperando", "cinco minutos", "tres camiones atrás", "diez metros", "seis bolsas",
         "doce en total", "dos de ellas", "once grandes", "uno nada más", "twelve crates", "two hours", "five minutes"]
CLEAN = ["traigo doce tarimas", "y son de madera", "de la caja", "peso cuatro mil ochocientos kilos", "sin material "
         "peligroso", "gracias", "y el peso es cuatro mil ochocientos kilos", "es de la caja", "thanks", "eh traigo doce tarimas"]


def test_fix_6_cases_are_in_the_corpus():
    assert len(fix(6)) == 6


@pytest.mark.parametrize("clause", GLUED)
def test_fix_6_a_glued_clause_never_gives_a_longer_id(clause):
    import random
    rng = random.Random(f"nube-8:glued:{clause}")
    from tests.test_voice_variantes import ES_0_29
    for _ in range(50):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(3, 8)))
        said = f"traigo doce tarimas el sello es mx guión {' '.join(ES_0_29[int(d)] for d in digits)}"
        w = witness(f"{said} {clause}")
        assert w.seal is None and w.detail.startswith("seal digits run into the next words: "), (said, w.detail)
        assert w.count == 12
        for sep in (", ", ". "):  # with a comma or a stop the clause is apart: the seal reads, exactly
            w = witness(f"{said}{sep}{clause}")
            assert w.seal == f"MX-{digits}", (said, sep, w.seal, w.detail)


@pytest.mark.parametrize("clause", CLEAN)
def test_fix_6_words_a_number_cannot_count_end_the_seal_cleanly(clause):
    w = witness(f"doce tarimas el sello es mx guión cuatro ocho dos uno tres {clause}")
    assert (w.seal, w.count) == ("MX-48213", 12), w.detail


def test_fix_6_fuzz_the_seal_is_exact_or_none_with_a_reason():
    import random
    from tests.test_voice_variantes import ES_0_29
    rng = random.Random("nube-8:fuzz-6")
    vocab = ("doce tarimas paquetes horas minutos dos tres cinco de la el y son traigo peso kilos mil en total sin con "
             "material gracias camiones grandes cajas bultos , . eh pues no perdón").split()
    for _ in range(3000):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(3, 8)))
        tail = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 6)))
        first = next((t for t in tail.split() if t not in (",", "eh", "pues")), "")  # fillers are pauses in an id
        if first in ("doce", "dos", "tres", "cinco", "mil"):  # a number said next is the seal's own digits ("...
            continue  # cuatro dos" is ...42) unless a noun follows it: that is test_fix_6_a_glued_clause_...
        text = f"el sello es mx guión {' '.join(ES_0_29[int(d)] for d in digits)} {tail}"
        w = witness(text)
        assert w.seal in (None, f"MX-{digits}"), (text, w.seal)
        assert w.seal is not None or "seal" in w.detail, text


# --- 7. words between the keyword and the id; plural keywords; a keyword said last ---------------------------------

BETWEEN_ES = ["de la caja es", "que traigo es el", "pues es", "de la caja de atrás es el", "que me dieron es",
              "lo revisé y es", ", el de la caja, es", "de seguridad que trae es"]
BETWEEN_EN = ["on the trailer is", "we have is", "on the back door is", "they gave me is"]
IDS = [("MX-48213", "MX-48213"), ("MX guión cuatro ocho dos uno tres", "MX-48213"), ("mx guion cuatro ocho dos uno tres",
       "MX-48213"), ("eme equis guión cuatro ocho dos uno tres", "MX-48213"), ("cuatro ocho dos uno tres", "48213"),
       ("doble u equis cuatro ocho dos uno tres", "WX48213")]


def test_fix_7_cases_are_in_the_corpus():
    assert len(fix(7)) == 14


@pytest.mark.parametrize("said,seal", IDS)
def test_fix_7_words_between_the_keyword_and_an_id_that_reads_one_way(said, seal):
    for between in BETWEEN_ES:
        w = witness(f"Traigo doce tarimas. El sello {between} {said}.")
        assert (w.count, w.seal) == (12, seal), (between, said, w.detail)
    for between in BETWEEN_EN:
        w = witness(f"Twelve pallets. The seal {between} M X dash four eight two one three.")
        assert (w.count, w.seal) == (12, "MX-48213"), (between, w.detail)


@pytest.mark.parametrize("text,why", [
    ("El sello de la caja cuatro es MX-48213.", "unclear which number is the seal"),
    ("El sello del tráiler dos es cuatro ocho dos uno tres.", "unclear which number is the seal"),
    ("The seal on trailer four is MX-48213.", "unclear which number is the seal"),
    ("El sello de la caja MX-48213.", "unclear which number is the seal"),
    ("The seal should be four eight two one three.", "unclear where the seal starts"),
    ("seal s MX 48213", "unclear where the seal starts"),
    ("El sello es mxñ guión cuatro ocho dos uno tres.", "unclear where the seal starts"),
    ("El sello lo traigo y son doce tarimas.", "seal said but no id read"),
    ("El sello que me dieron ayer en la caseta de cobro del norte es MX-48213.", "seal said but no id read"),
])
def test_fix_7_an_id_that_does_not_read_one_way_gives_a_reason(text, why):
    w = witness(text)
    assert w.seal is None and why in w.detail, (w.seal, w.detail)


@pytest.mark.parametrize("plural", ["sellos", "seals", "precintos", "marchamos", "Sellos"])
def test_fix_7_a_plural_keyword_is_never_silent(plural):
    for text in (f"Traigo doce tarimas, los {plural} son MX-48213 y MX-48214.", f"Traigo doce tarimas, {plural} MX-48213.",
                 f"Traigo doce tarimas y los {plural}."):
        w = witness(text)
        assert w.seal is None and f"several seals said in transcript: {plural!r}" in w.detail, (text, w.detail)
        assert w.count == 12 and judge(V, D, w, L).reasons == ("SEAL_MISSING voice",)


@pytest.mark.parametrize("keyword", ["sello", "seal", "precinto", "marchamo", "Sello", "SEAL"])
@pytest.mark.parametrize("end", ["", ".", "...", " eh", ","])
def test_fix_7_a_keyword_said_last_is_never_silent(keyword, end):
    w = witness(f"Traigo doce tarimas y el {keyword}{end}")
    assert w.seal is None and "seal said but no id read" in w.detail, w.detail
    assert w.count == 12


def test_fix_7_fuzz_any_seal_word_anywhere_is_never_silent():
    import random
    import re
    rng = random.Random("nube-8:fuzz-7")
    vocab = ("sello sellos seal seals el la de que es is the on caja tráiler trailer cuatro ocho dos uno tres doce "
             "tarimas MX MX-48213 mx guión doble u no roto , . traigo pues eh").split()
    for _ in range(4000):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 12)))
        w = witness(text)
        assert w.seal is None or any(ch.isdigit() for ch in w.seal), text
        if w.seal is None and re.search(r"\b(sellos?|seals?)\b", text):
            assert "seal" in w.detail, text


@pytest.mark.parametrize("text", ["Traigo doce tarimas y el sello No, perdón, trece.",
                                  "Traigo doce tarimas y el sello es, perdón, trece.",
                                  "Traigo doce tarimas, el sello lo traigo, perdón, trece."])
def test_fix_7_a_correction_after_the_keyword_is_never_taken_for_the_id(text):
    # The words-between scan must not step over a correction and claim the corrected count as the seal.
    w = witness(text)
    assert w.count is None and "self-correction" in w.detail, w.detail
    assert w.seal is None and "seal" in w.detail
