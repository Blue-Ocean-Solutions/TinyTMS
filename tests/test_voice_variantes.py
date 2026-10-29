"""Voice parser (tinytms.adapters.voice_from_transcript) against phrasings it has not been tuned on.

1. tests/data/voice-es-variantes.json: a SYNTHETIC hand-written corpus, Spanish and English. Each case says what the
   speaker meant and what the parser must return; the expected value is either the meant one or None.
2. Generated number words (stdlib random, fixed seed): every count 0..1000 and a sample of seals up to 999999,
   written out in Spanish and in English by a speller in this file, independent of the parser's own tables.
3. Self-corrections and alternatives ("doce… no, trece", "doce o trece"): count None and a detail that says why.

The rule: the right count / seal, or None. Never a wrong number. The three REAL transcripts stay pinned in
tests/test_real_data.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tinytms.adapters import parse_count, parse_seal, voice_from_transcript
from tinytms.verify import normalize_seal

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "voice-es-variantes.json"
CORPUS = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
CASES = CORPUS["cases"]
SEED = 20261029


def witness(text: str):
    return voice_from_transcript({"transcript": text}, model="FIXTURE", provider="FIXTURE", source_sha256="")


def test_corpus_is_labelled_synthetic_and_consistent():
    assert CORPUS["provenance"] == "SYNTHETIC hand-written"
    assert len(CASES) >= 30
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert {c["id"].split("-")[0] for c in CASES} == {"es", "en"}
    for c in CASES:
        m, e = c["meant"], c["expect"]
        # the corpus itself never asks for a wrong number: expect is the meant value or None
        assert e["count"] is None or e["count"] == m["count"], c["id"]
        assert e["seal"] is None or normalize_seal(e["seal"]) == normalize_seal(m["seal"]), c["id"]
    # it holds what the lane order asks for
    texts = " ".join(c["text"].lower() for c in CASES)
    for needle in ("docena", "quince cajas", "cero", "guión", "raya", "eme equis",
                   "mx cuarenta y ocho mil doscientos trece", "no, trece"):
        assert needle in texts, needle
    assert sum(c["expect"]["count"] is not None for c in CASES) >= 20
    assert sum(c["expect"]["seal"] is not None for c in CASES) >= 15


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_corpus_case(case):
    w = witness(case["text"])
    assert w.status == "OK"
    assert (w.count, w.seal) == (case["expect"]["count"], case["expect"]["seal"]), w.detail
    assert w.count is None or w.count == case["meant"]["count"], "wrong count"
    assert w.seal is None or normalize_seal(w.seal) == normalize_seal(case["meant"]["seal"]), "wrong seal"
    if w.count is None:
        assert w.detail, "a missing count always says why"
        assert case.get("detail", "") in w.detail, w.detail
    assert parse_count(case["text"])[0] == w.count and parse_seal(case["text"]) == w.seal


# --- generated number words ----------------------------------------------------------------------------------------

ES_0_29 = ("cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis "
           "diecisiete dieciocho diecinueve veinte veintiuno veintidós veintitrés veinticuatro veinticinco veintiséis "
           "veintisiete veintiocho veintinueve").split()
ES_TENS = dict(zip(range(3, 10), "treinta cuarenta cincuenta sesenta setenta ochenta noventa".split()))
ES_HUNDREDS = dict(zip(range(1, 10), ("ciento doscientos trescientos cuatrocientos quinientos seiscientos setecientos "
                                      "ochocientos novecientos").split()))
EN_0_19 = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
           "seventeen eighteen nineteen").split()
EN_TENS = dict(zip(range(2, 10), "twenty thirty forty fifty sixty seventy eighty ninety".split()))


def _es_below_1000(n: int, apocope: bool) -> str:
    h, r = divmod(n, 100)
    out = []
    if h:
        out.append("cien" if h == 1 and r == 0 else ES_HUNDREDS[h])
    if r:
        if r < 30:
            out.append({1: "un", 21: "veintiún"}.get(r, ES_0_29[r]) if apocope else ES_0_29[r])
        else:
            d, u = divmod(r, 10)
            out.append(ES_TENS[d] + ("" if not u else " y " + ("un" if apocope and u == 1 else ES_0_29[u])))
    return " ".join(out)


def es_words(n: int, apocope: bool = False) -> str:
    """Spanish cardinal, 0..999999. apocope: the form before a noun ('un pallet', 'veintiún pallets')."""
    if n == 0:
        return "cero"
    t, r = divmod(n, 1000)
    parts = []
    if t:
        parts.append("mil" if t == 1 else _es_below_1000(t, True) + " mil")
    if r:
        parts.append(_es_below_1000(r, apocope))
    return " ".join(parts)


def _en_below_1000(n: int, rng: random.Random) -> str:
    h, r = divmod(n, 100)
    out = []
    if h:
        out.append(EN_0_19[h] + " hundred")
    if r:
        if h and rng.random() < 0.5:
            out.append("and")
        if r < 20:
            out.append(EN_0_19[r])
        else:
            d, u = divmod(r, 10)
            out.append(EN_TENS[d] + ("" if not u else rng.choice(("-", " ")) + EN_0_19[u]))
    return " ".join(out)


def en_words(n: int, rng: random.Random) -> str:
    if n == 0:
        return "zero"
    t, r = divmod(n, 1000)
    return " ".join(p for p in ((_en_below_1000(t, rng) + " thousand") if t else "",
                                _en_below_1000(r, rng) if r else "") if p)


def test_speller_spot_checks():
    rng = random.Random(0)
    assert es_words(48213) == "cuarenta y ocho mil doscientos trece"
    assert es_words(21, apocope=True) == "veintiún" and es_words(31, apocope=True) == "treinta y un"
    assert es_words(100) == "cien" and es_words(101) == "ciento uno" and es_words(1000) == "mil"
    assert es_words(21000) == "veintiún mil" and es_words(100000) == "cien mil"
    assert en_words(24, rng) in ("twenty-four", "twenty four") and en_words(0, rng) == "zero"


def test_every_spanish_count_up_to_1000():
    for n in range(0, 1001):
        text = f"Traigo {es_words(n, apocope=True)} pallets de madera."
        assert parse_count(text) == (n, ""), text


def test_every_english_count_up_to_1000():
    rng = random.Random(f"{SEED}:en-count")
    for n in range(0, 1001):
        text = f"I have {en_words(n, rng)} pallets."
        assert parse_count(text) == (n, ""), text


def test_spanish_seals_said_as_one_number():
    rng = random.Random(f"{SEED}:es-seal")
    for n in [0, 7, 10, 21, 100, 101, 1000, 48213, 100000, 999999] + [rng.randint(0, 999999) for _ in range(1500)]:
        text = f"El sello es MX guión {es_words(n)}. Peso cuatro mil ochocientos kilos."
        assert parse_seal(text) == f"MX-{n}", text


def test_seals_said_digit_by_digit():
    rng = random.Random(f"{SEED}:digits")
    for _ in range(1500):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(3, 8)))
        es = " ".join(ES_0_29[int(d)] for d in digits)
        en = " ".join(EN_0_19[int(d)] for d in digits)
        assert parse_seal(f"El sello es MX raya {es}, traigo doce tarimas.") == f"MX-{digits}", es
        assert parse_seal(f"The seal is M X dash {en}.") == f"MX-{digits}", en
        assert parse_count(f"El sello es MX raya {es}, traigo doce tarimas.") == (12, ""), es


def test_english_seals_said_as_one_number():
    rng = random.Random(f"{SEED}:en-seal")
    for n in [0, 13, 40, 213, 48213] + [rng.randint(0, 999999) for _ in range(1500)]:
        text = f"Seal number {en_words(n, rng)}."
        assert parse_seal(text) == str(n), text


# --- fail closed on ambiguity --------------------------------------------------------------------------------------


def _pairs(tag: str, k: int = 400) -> list[tuple[int, int]]:
    rng = random.Random(f"{SEED}:{tag}")
    return [(rng.randint(0, 60), rng.randint(0, 60)) for _ in range(k)] + [(12, 13), (13, 12), (12, 12)]


def test_self_correction_is_never_resolved_to_either_count():
    for a, b in _pairs("correction"):
        for text in (f"Traigo {es_words(a, True)} tarimas… no, {es_words(b, True)}.",
                     f"Traigo {es_words(a, True)} tarimas, perdón, {es_words(b, True)}.",
                     f"Son {es_words(a, True)}, digo, {es_words(b, True)} tarimas.",
                     f"{en_words(a, random.Random(a))} pallets, sorry, {en_words(b, random.Random(b))}."):
            count, why = parse_count(text)
            if a == b:
                assert (count, why) == (a, ""), text
            else:
                assert count is None, text
                assert f"ambiguous counts in transcript: {sorted({a, b})}" in why and "correction" in why, (text, why)
                w = witness(text)
                assert w.count is None and w.detail == why


def test_alternative_counts_are_never_resolved():
    for a, b in _pairs("alternative"):
        wb = es_words(b, True)
        conj = "u" if wb.startswith("o") else "o"
        for text in (f"Son {es_words(a, True)} {conj} {wb} tarimas.",
                     f"{en_words(a, random.Random(a))} or {en_words(b, random.Random(b))} pallets."):
            count, why = parse_count(text)
            if a == b:
                assert count == a, text
            else:
                assert count is None and f"ambiguous counts in transcript: {sorted({a, b})}" in why, (text, why)


def test_doce_no_trece_from_the_lane_order():
    w = witness("Traigo doce pallets de madera… no, trece. El sello es MX guión cuatro ocho dos uno tres.")
    assert w.count is None and w.seal == "MX-48213"
    assert w.detail == "ambiguous counts in transcript: [12, 13] (self-correction: 'no')"


def test_a_correction_appended_to_any_corpus_count_never_keeps_the_old_count():
    for c in CASES:
        n = c["expect"]["count"]
        if n is None or c["meant"]["seal"] is not None:
            continue
        for b in (n + 1, n + 10, max(n - 1, 0) if n else 3):
            for tail in (f" No, perdón, {es_words(b, True)}.", f" -perdón- {es_words(b, True)}.",
                         f" Sorry, I mean {en_words(b, random.Random(b))}."):
                count, why = parse_count(c["text"] + tail)
                assert count is None and "correction" in why, (c["text"] + tail, why)


VOCAB = (
    "traigo son no perdón digo sorry o or y and unas about tráiler el sello seal es is MX SMX M X guión raya dash "
    "eme equis doble oh de madera kilos peso cero uno dos tres cuatro doce trece veinte veintiún treinta cuarenta "
    "cien ciento mil docena docenas media half a dozen twelve twenty four hundred thousand tarimas pallets cajas "
    "payets 12 4,800 4.5 007 MX-48213 OK eh , . … - :"
).split()


def test_fuzz_never_raises_and_always_explains_a_missing_count():
    rng = random.Random(f"{SEED}:fuzz")
    for _ in range(4000):
        text = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(0, 25)))
        w = witness(text)
        assert w.status == "OK"
        assert w.count is None or (type(w.count) is int and w.count >= 0), text
        assert w.seal is None or (isinstance(w.seal, str) and any(ch.isdigit() for ch in w.seal)), text
        assert w.count is not None or w.detail, text
