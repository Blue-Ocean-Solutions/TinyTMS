"""Every word of every word list in the voice parser has a test that fails when the word is dropped (ORDER Nube-9).

Each case pins the exact witness for one phrase (count, seal, detail), then rebuilds the parser from its source with
that one word deleted from that one list (tests/voice_mutants.py) and checks the phrase no longer gives that witness.
A word list, or a word, added to tinytms/adapters.py without a case here fails test_every_word_of_every_list_has_a_case.
"""

from __future__ import annotations

import pytest

from tests import voice_mutants
from tinytms import adapters
from tinytms.adapters import voice_from_transcript

SEAL_DIGITS = "cuatro ocho dos uno tres"
EN_HEDGES = {"maybe", "perhaps", "probably", "possibly", "think", "i think", "guess", "i guess", "believe", "i believe",
             "might", "may be", "pretty sure", "fairly sure", "want to say", "not sure", "could be"}
UNIT_WORD = {"pallets?": "pallets", "palets?": "palets", "pal[eé]s": "palés", "payets?": "payets",
             "pajets?": "pajets", "payettes?": "payettes", "paletas?": "paletas", "tarimas?": "tarimas",
             "skids?": "skids", "cajas?": "cajas", "boxes": "boxes", "box": "box", "bultos?": "bultos",
             "piezas?": "piezas", "pieces?": "pieces"}
NUMBER_TABLES = ("_ES_ONES", "_ES_TEENS", "_ES_TENS", "_ES_HUNDREDS", "_EN_ONES", "_EN_TEENS", "_EN_TENS")
SPLIT = {"veinti": ("dos", 22), "dieci": ("seis", 16)}
REPORTED = "seal reported broken, missing or not matching in transcript"


def _number(name: str, w: str) -> tuple[str, tuple]:
    value = getattr(adapters, name)[w]
    if name.startswith("_EN"):
        return f"I have {w} pallets.", (value, None, "")
    return f"Traigo {w} tarimas.", (value, None, "")


def case(name: str, w: str) -> tuple[str, tuple]:
    """(phrase, (count, seal, detail)) for word w of list name."""
    if name in NUMBER_TABLES:
        return _number(name, w)
    if name == "_UNITS":
        unit = UNIT_WORD[w]
        return (("I have one box.", (1, None, "")) if unit == "box" else (f"I have twelve {unit}.", (12, None, "")))
    if name == "_WEIGHT_UNITS":
        return (f"doce tarimas sello mx guión {SEAL_DIGITS} cuatro mil {w}",
                (12, None, f"seal digits run into a weight: '{SEAL_DIGITS} cuatro mil {w}'"))
    if name == "_SEAL_DASH":
        return f"Traigo doce tarimas. El sello es MX {w} {SEAL_DIGITS}.", (12, "MX-48213", "")
    if name == "_NUMBER_WORDS":
        return {"cero": ("Traigo doce tarimas, sello MX guión cero-cuatro-ocho.", (12, "MX-048", "")),
                "zero": ("Twelve pallets, seal MX dash zero-four-eight.", (12, "MX-048", "")),
                "cien": ("Doce tarimas. Trece cien.", (None, None, "counts disagree in transcript: [12, 13, 100]")),
                "mil": ("Son cuatro-mil tarimas.", (4000, None, "")),
                "hundred": ("I have two-hundred pallets.", (200, None, "")),
                "thousand": ("I have two-thousand pallets.", (2000, None, ""))}[w]
    if name == "_DOZEN":
        return {"docena": ("Traigo una docena de cajas.", (12, None, "")),
                "dozen": ("I have a dozen boxes.", (12, None, ""))}[w]
    if name == "_DOZENS":
        return {"docenas": ("Traigo dos docenas de cajas.", (24, None, "")),
                "dozens": ("I have two dozens boxes.", (24, None, ""))}[w]
    if name == "_ID_NOUNS":
        return f"Traigo doce tarimas, {w} cuatro.", (12, None, "")
    if name == "_CORRECTIONS":
        return (f"Traigo doce tarimas, {w}, trece.",
                (None, None, f"counts disagree in transcript: [12, 13] (self-correction: '{w}')"))
    if name == "_APPROX_BEFORE":
        return f"Traigo {w} doce tarimas.", (None, None, "approximate count in transcript: 'doce tarimas'")
    if name == "_APPROX_AFTER":
        return f"Traigo doce tarimas {w}.", (None, None, "approximate count in transcript: 'doce tarimas'")
    if name == "_HEDGES":
        said = "twelve pallets" if w in EN_HEDGES else "doce tarimas"
        return f"{w} {said}.", (None, None, f"hedged count in transcript: '{w} {said}'")
    if name == "_HEDGES_AFTER":
        return f"Doce tarimas, {w}.", (None, None, f"hedged count in transcript: 'doce tarimas , {w}'")
    if name == "_NEGATIONS":
        return f"{w} traigo doce tarimas.", (None, None, f"negated count in transcript: '{w} traigo doce tarimas'")
    if name == "_NEGATED_CONTRACTIONS":
        return (f"We {w}'t have twelve pallets.",
                (None, None, f"negated count in transcript: '{w} t have twelve pallets'"))
    if name == "_COUNT_VERBS":
        return f"Doce tarimas. Trece {w}.", (None, None, "counts disagree in transcript: [12, 13]")
    if name == "_ALTERNATIVES" and w in ("ni", "nor"):  # for a count "ni" is a negation first
        return (f"Traigo doce tarimas, sello MX-48213 {w} MX-48214.",
                (12, None, "seals disagree in transcript: ['MX48213', 'MX48214']"))
    if name == "_ALTERNATIVES":
        return (f"Son doce {w} trece tarimas.",
                (None, None, f"counts disagree in transcript: [12, 13] (alternative: '{w}')"))
    if name in ("_AFTER_CORRECTION", "_SPEECH_FILLERS", "_FILLER_PHRASES"):
        said = w.replace("let s", "let's")
        return (f"Traigo doce tarimas, no {said} trece.",
                (None, None, "counts disagree in transcript: [12, 13] (self-correction: 'no')"))
    if name == "_NOT_FILLER":
        return f"Traigo doce {w} tarimas.", (None, None, "no count found in transcript")
    if name == "_COUNT_AFTER":
        return f"Doce tarimas. Trece {w}.", (None, None, "counts disagree in transcript: [12, 13]")
    if name == "_MORE":
        return f"Traigo doce tarimas y una {w}.", (None, None, f"unclear count in transcript: 'una {w}'")
    if name == "_ARTICLES":
        return ((f"Twelve pallets. Sorry, one question.", (12, None, "")) if w == "one"
                else (f"Doce tarimas. Bueno, {w} saludo.", (12, None, "")))
    if name == "_NOT_COUNT_BEFORE":
        return {"the": ("Twelve pallets, not the one I said.", (12, None, "")),
                "las": ("Doce tarimas, llegué a las tres.", (12, None, "")),
                "la": ("Doce tarimas, llegué a la una.", (12, None, "")),
                "at": ("Twelve pallets, got here at three.", (12, None, "")),
                "cada": ("Doce tarimas, cada una.", (12, None, "")),
                "each": ("Twelve pallets, each one.", (12, None, ""))}[w]
    if name == "_WEIGHT_WORDS":
        return f"Doce tarimas, {w} cuatro mil.", (12, None, "")
    if name == "_ID_LEAD":
        return f"Doce tarimas, andén {w} cuatro.", (12, None, "")
    if name == "_SEAL_KEYWORDS":
        return f"Traigo doce tarimas, {w} MX-48213.", (12, "MX-48213", "")
    if name == "_SEAL_PLURALS":
        return f"Traigo doce tarimas, {w} MX-48213 y MX-48214.", (12, None, f"several seals said in transcript: '{w}'")
    if name == "_SEAL_LEAD_IN":
        return f"Traigo doce tarimas, sello {w} MX-48213.", (12, "MX-48213", "")
    if name == "_SEAL_ABBREVIATIONS":
        return f"Traigo doce tarimas, sello {w}. MX-48213.", (12, "MX-48213", "")
    if name == "_SEAL_BAD":
        return f"Traigo doce tarimas, sello MX-48213 {w}.", (12, None, f"{REPORTED}: 'sello mx-48213 {w}'")
    if name == "_SEAL_MATCH":
        return f"Traigo doce tarimas, sello MX-48213, no {w}.", (12, None, f"{REPORTED}: 'sello mx-48213 , no {w}'")
    if name == "_NEGATION_GAP" and w == "be":  # "be" is also the letter B
        return "Twelve pallets, seal MX-48213, it will not be broken.", (12, "MX-48213", "")
    if name == "_NEGATION_GAP":
        return f"Traigo doce tarimas, sello MX-48213, no {w} roto.", (12, "MX-48213", "")
    if name == "_HAZMAT":
        return f"Traigo doce tarimas, el sello es MX-48213 y no hay material {w}.", (12, "MX-48213", "")
    if name == "_SEAL_BEFORE_SKIP":
        return f"Traigo doce tarimas, no {w} sello MX-48213.", (12, None, f"negated seal in transcript: 'no {w} sello'")
    if name == "_SPELLED":
        return (f"Traigo doce tarimas, sello eme {w} guión {SEAL_DIGITS}.",
                (12, f"M{adapters._SPELLED[w]}-48213", ""))
    if name == "_NOT_SEAL":
        return f"Traigo doce tarimas, sello MX-48213 {w}.", (12, "MX-48213", "")
    if name == "_WORD_LETTERS":
        return {"em": ("Twelve pallets, seal em ex dash four eight two one three.", (12, "MX-48213", "")),
                "el": ("Twelve pallets, seal el zee dash four four one two.", (12, "LZ-4412", "")),
                "a": (f"Traigo doce tarimas, sello a be guión {SEAL_DIGITS}.", (12, "AB-48213", "")),
                "te": (f"Traigo doce tarimas, sello te equis guión {SEAL_DIGITS}.", (12, "TX-48213", "")),
                "ese": (f"Traigo doce tarimas, sello ese eme equis guión {SEAL_DIGITS}.", (12, "SMX-48213", "")),
                "de": (f"Traigo doce tarimas, sello MX guión {SEAL_DIGITS} guión de.", (12, "MX-48213-D", ""))}[w]
    if name == "_NOT_SEAL_LETTERS" and w in ("no", "ya", "si", "ok"):  # after the keyword "NO", "YA" ... never read
        return (f"Traigo doce tarimas, sello MX guión {SEAL_DIGITS} guión {w}.",
                (12, None, f"seal ends in a dash: 'mx guión {SEAL_DIGITS} guión {w}'"))
    if name == "_NOT_SEAL_LETTERS":  # before a dash a word is never read as the seal's letters
        why = (f"negated seal in transcript: 'sello mx {w}'" if w == "ni"
               else f"seal said but no id read: 'sello mx {w}'")
        return f"Traigo doce tarimas, sello MX {w} guión {SEAL_DIGITS}.", (12, None, why)
    if name == "_LETTER_PAIRS":
        return (f"Traigo doce tarimas, sello {w} equis {SEAL_DIGITS}.",
                (12, f"{adapters._LETTER_PAIRS[tuple(w.split())]}X48213", ""))
    if name == "_ENDS_SEAL" and w == "i":  # a lone "i" ending the note may be a letter: never read
        return f"doce tarimas sello mx guión {SEAL_DIGITS} i have them", (12, "MX-48213", "")
    if name == "_ENDS_SEAL":
        return f"doce tarimas sello mx guión {SEAL_DIGITS} {w}", (12, "MX-48213", "")
    if name == "_LOW_PUNCT":
        return {"stop": ("Creo que sí… doce tarimas.", (12, None, "")),
                "comma": ("Doce tarimas: creo.", (None, None, "hedged count in transcript: 'doce tarimas , creo'")),
                "dash": ("Creo que sí — doce tarimas.", (12, None, ""))}[w]
    if name == "_ES_SPLIT":
        ones, value = SPLIT[w]
        return f"Traigo {w} {ones} tarimas.", (value, None, "")
    if name == "_ADDED":
        return (f"Traigo doce tarimas y {w}.",
                (None, None, f"more than the count said in transcript: 'doce tarimas y {w}'"))
    if name == "_SOURCES":
        return (f"Traigo doce tarimas. {w} sello MX-48213.",
                (12, None, f"seal said from the paperwork or someone else: '{w} sello'"))
    if name == "_SEAL_PARTIAL":
        return f"Traigo doce tarimas, sello MX-48213, {w}.", (12, None, f"part of a seal said in transcript: "
                                                                        f"'sello mx-48213 , {w}'")
    if name == "_LETTER_WORDS":
        return (f"Traigo doce tarimas, sello MX-48213, {w} a.",
                (12, None, f"seal letter said after the digits: 'sello mx-48213 , {w}'"))
    if name == "_SEALING":
        return f"Twelve pallets, not {w}.", (12, None, f"{REPORTED}: 'pallets , not {w}'")
    if name == "_EN_SPELLED":
        return (f"Twelve pallets, seal em {w} dash four eight two one three.",
                (12, f"M{adapters._EN_SPELLED[w]}-48213", ""))
    if name == "_TRAILER":
        return f"Twelve pallets, the {w} is broken.", (12, None, f"{REPORTED}: 'the {w} is broken'")
    if name == "_CLITICS":
        return f"Twelve pallets. The seal'{w} 48213.", (12, "48213", "")
    raise KeyError(name)


LISTS = [n for n in voice_mutants.list_names() if n not in voice_mutants.NOT_LISTS]
CASES = [(n, w) for n in LISTS for w in dict.fromkeys(voice_mutants.own_words(n))]


def witness(text: str, parse=voice_from_transcript) -> tuple:
    w = parse({"transcript": text}, model="prueba", provider="none", source_sha256="")
    return w.count, w.seal, w.detail


def test_every_word_of_every_list_has_a_case():
    assert len(LISTS) >= 50 and len(CASES) >= 800
    missing = []
    for name, w in CASES:
        try:
            case(name, w)
        except KeyError:
            missing.append((name, w))
    assert not missing, missing


@pytest.mark.parametrize("name,word", CASES, ids=[f"{n}:{w}" for n, w in CASES])
def test_dropping_the_word_changes_what_the_phrase_gives(name, word):
    text, expected = case(name, word)
    assert witness(text) == expected, text
    assert witness(text, voice_mutants.without(name, word)) != expected, f"{word!r} does nothing in {name}: {text}"
