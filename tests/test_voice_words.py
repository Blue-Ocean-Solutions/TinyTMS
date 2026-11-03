"""Every word of every word list in the voice parser has a test that fails when the word is dropped (ORDER Nube-9,
item 8; ORDER Nube-10, item 9).

WORDS below is the table of every word of every list, written out here, not read from tinytms/adapters.py: deleting a
word from a list in the parser fails test_the_table_is_the_parsers_lists (the word is in the table and no longer in the
list), and so does adding one (it has no case yet). Each case pins the exact witness for one phrase (count, seal,
detail), then rebuilds the parser from its source with that one word deleted from that one list
(tests/voice_mutants.py) and checks the phrase no longer gives that witness.
"""

from __future__ import annotations

import pytest

from tests import voice_mutants
from tinytms import adapters
from tinytms.adapters import voice_from_transcript

SEAL_DIGITS = "cuatro ocho dos uno tres"
NUMBER_TABLES = ("_ES_ONES", "_ES_TEENS", "_ES_TENS", "_ES_HUNDREDS", "_EN_ONES", "_EN_TEENS", "_EN_TENS")
SPLIT = {"veinti": ("dos", 22), "dieci": ("seis", 16)}
REPORTED = "seal reported broken, missing or not matching in transcript"
FROM = "count said from the paperwork or someone else"
SEAL_FROM = "seal said from the paperwork or someone else"
UNIT_WORD = {"pallets?": "pallets", "palets?": "palets", "pal[eé]s": "palés", "payets?": "payets",
             "pajets?": "pajets", "payettes?": "payettes", "paletas?": "paletas", "tarimas?": "tarimas",
             "skids?": "skids", "cajas?": "cajas", "boxes": "boxes", "box": "box", "bultos?": "bultos",
             "piezas?": "piezas", "pieces?": "pieces", "drums?": "drums", "cartons?": "cartons", "cases?": "cases",
             "rollos?": "rollos", "tambos?": "tambos", "bobinas?": "bobinas", "crates?": "crates", "coils?": "coils",
             "garraf[oó]n": "garrafón", "garrafones": "garrafones", "cubetas?": "cubetas", "bundles?": "bundles",
             "sacos?": "sacos", "bags?": "bags", "costales?": "costales", "rolls?": "rolls", "reels?": "reels",
             "totes?": "totes", "barrels?": "barrels", "barriles?": "barriles", "bins?": "bins", "racks?": "racks",
             "pacas?": "pacas", "bales?": "bales", "huacales?": "huacales", "bidones?": "bidones",
             "sacks?": "sacks", "carretes?": "carretes", "paquetes?": "paquetes", "packages?": "packages",
             "tires?": "tires"}
ENGLISH = set("""i you we it is are was the a an of to in on at for with and or but not no yes my your this that
there here have has had got get do does did can could would should will may might must be been being
""".split())


def said(w: str) -> str:
    """A list phrase as written: 'i m pretty sure' -> "i'm pretty sure", 'don t know' -> "don't know"."""
    out = []
    for part in w.split():
        if out and part in ("s", "t", "m", "d", "re", "ll", "ve"):
            out[-1] += "'" + part
        else:
            out.append(part)
    return " ".join(out)


def en(w: str) -> bool:
    """Is the phrase English (its cases are written in English)?"""
    return any(ch.isascii() for ch in w) and not set(w) & set("áéíóúñ") and (
        set(w.split()) & ENGLISH or w in EN_WORDS)


EN_WORDS = set("""maybe perhaps probably possibly think guess believe might supposedly supposed guessing reckon suppose
ish ballpark estimate estimated approximately roughly approx""".split())


def case(name: str, w: str) -> tuple[str, tuple]:  # noqa: C901 - one branch per list
    """(phrase, (count, seal, detail)) for word w of list name."""
    if name in NUMBER_TABLES:
        value = getattr(adapters, name)[w]
        return ((f"I have {w} pallets.", (value, None, "")) if name.startswith("_EN")
                else (f"Traigo {w} tarimas.", (value, None, "")))
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
    if name == "_ID_NOUNS" and w in adapters._NUMBER_NOUNS:  # alone, "número" names the field being corrected
        return f"Traigo doce tarimas, factura {w} cuatro.", (12, None, "")
    if name == "_ID_NOUNS":
        return f"Traigo doce tarimas, {w} cuatro.", (12, None, "")
    if name == "_NUMBER_NOUNS":
        return (f"Doce tarimas. Perdón, el {w} es trece. Sello MX-48213.",
                (None, "MX-48213", "counts disagree in transcript: [12, 13]"))
    if name == "_NUMBERED":
        return f"Twelve pallets, {w} number 4471. Seal MX-48213.", (12, "MX-48213", "")
    if name == "_ID_ARTICLES":  # past an article, the driver acts on the place: not a source
        return f"Traigo doce tarimas, vengo de {w} planta. Sello MX-48213.", (12, "MX-48213", "")
    if name == "_POSSESSIVES":
        return f"Twelve pallets. {w.capitalize()} number is 4471. Seal MX-48213.", (12, "MX-48213", "")
    if name == "_CORRECTIONS":
        return (f"Traigo doce tarimas, {w}, trece.",
                (None, None, f"counts disagree in transcript: [12, 13] (self-correction: '{w}')"))
    if name == "_APPROX_BEFORE":
        return f"Traigo {w} doce tarimas.", (None, None, "approximate count in transcript: 'doce tarimas'")
    if name == "_APPROX_AFTER":
        return f"Traigo doce tarimas {w}.", (None, None, "approximate count in transcript: 'doce tarimas'")
    if name == "_HEDGES":  # said at the end of the sentence, far from the count (ORDER Nube-10 item 5)
        if en(w):
            return (f"I have twelve pallets of tile, {said(w)}.",
                    (None, None, f"hedged count in transcript: 'twelve pallets of tile , {w}'"))
        return (f"Traigo doce tarimas de madera, {said(w)}.",
                (None, None, f"hedged count in transcript: 'doce tarimas de madera , {w}'"))
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
        return (f"Traigo doce tarimas, no {said(w)} trece.",
                (None, None, "counts disagree in transcript: [12, 13] (self-correction: 'no')"))
    if name == "_NOT_FILLER":
        return f"Traigo doce {w} tarimas.", (None, None, "no count found in transcript")
    if name == "_COUNT_AFTER":
        return f"Doce tarimas. Trece {w}.", (None, None, "counts disagree in transcript: [12, 13]")
    if name == "_MORE":
        return f"Traigo doce tarimas y una {w}.", (None, None, f"unclear count in transcript: 'una {w}'")
    if name == "_ARTICLES":
        return (("Twelve pallets. Sorry, one question.", (12, None, "")) if w == "one"
                else (f"Doce tarimas. Bueno, {w} saludo.", (12, None, "")))
    if name == "_ARTICLE_NOUNS":
        if en(w) or w in ("question", "moment", "second", "minute", "thing"):
            return f"Twelve pallets. Sorry, one {w}. Seal MX-48213.", (12, "MX-48213", "")
        return f"Doce tarimas. Perdón, un {w}. Sello MX-48213.", (12, "MX-48213", "")
    if name == "_PRONOUN_BEFORE":
        return f"Twelve pallets. Seal MX-48213. Have a {w} one.", (12, "MX-48213", "")
    if name == "_COUNT_CHANGE":
        return (f"Twelve pallets, one {w}. Seal MX-48213.",
                (None, "MX-48213", f"unclear count in transcript: 'one {w}'"))
    if name == "_ADVERBS":
        return (f"Twelve pallets. Thirteen {w}. Seal MX-48213.",
                (None, "MX-48213", "counts disagree in transcript: [12, 13]"))
    if name == "_NOT_COUNT_BEFORE":
        return {"the": ("Twelve pallets, the one.", (12, None, "")),
                "las": ("Doce tarimas, llegué a las tres.", (12, None, "")),
                "la": ("Doce tarimas, llegué a la una.", (12, None, "")),
                "at": ("Twelve pallets, got here at three.", (12, None, "")),
                "cada": ("Doce tarimas, cada una.", (12, None, "")),
                "each": ("Twelve pallets, each 3.", (12, None, "")),
                "minus": ("Twelve pallets. Set at minus ten.", (12, None, "")),
                "since": ("Twelve pallets. Here since three.", (12, None, "")),
                "desde": ("Doce tarimas. Aquí desde tres.", (12, None, "")),
                "negative": ("Twelve pallets. Set at negative ten.", (12, None, ""))}[w]
    if name == "_TIME_BEFORE":
        return f"Twelve pallets. Got here {w} six. Seal MX-48213.", (12, "MX-48213", "")
    if name == "_ONCE":
        return f"Twelve pallets. No wind {w}. Seal MX-48213.", (12, "MX-48213", "")
    if name == "_WEIGHT_WORDS":
        return f"Doce tarimas, {w} cuatro mil.", (12, None, "")
    if name == "_WEIGHT_GAP":
        return f"Doce tarimas. Pesa {w} 4800. Sello MX-48213.", (12, "MX-48213", "")
    if name == "_ID_LEAD" and w in adapters._NOT_COUNT_BEFORE:  # for the count these are a time already
        return f"Doce tarimas. Sello MX-48213. La caja es {w} 1180.", (12, "MX-48213", "")
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
    if name == "_KEYWORD_TAILS":
        return "Traigo doce tarimas, sello de seguridad MX-48213.", (12, "MX-48213", "")
    if name == "_SEAL_BAD":
        return f"Traigo doce tarimas, sello MX-48213 {w}.", (12, None, f"{REPORTED}: 'sello mx-48213 {w}'")
    if name == "_SEAL_BAD_PHRASES":  # in a later sentence (ORDER Nube-10 item 2)
        return (f"Doce tarimas. Sello MX-48213. {said(w).capitalize()}.",
                (12, None, f"{REPORTED}: 'sello mx-48213 . {w}'"))
    if name == "_BAD_NOT_SEAL":
        return f"Twelve pallets. Seal MX-48213. {w.capitalize()}.", (12, "MX-48213", "")
    if name == "_CLAUSE_BAD":
        return f"Twelve pallets, seal MX-48213 {w}.", (12, None, f"{REPORTED}: 'seal mx-48213 {w}'")
    if name == "_AFFIRMING_NEGATIONS":
        return f"Doce tarimas. Sello MX-48213, {w} roto.", (12, "MX-48213", "")
    if name == "_CONDITIONS":
        return (f"Twelve pallets. Seal MX-48213, {w} not broken.",
                (12, None, f"{REPORTED}: 'seal mx-48213 , {w} not broken'"))
    if name == "_CONDITION_GAP":
        text = {"i": "if I'm", "m": "if I'm", "am": "if I am", "it": "if it's", "s": "if it's", "is": "if it is"}[w]
        return (f"Twelve pallets. Seal MX-48213, {text} not broken.",
                (12, None, f"{REPORTED}: 'seal mx-48213 , {text.lower().replace(chr(39), ' ')} not broken'"))
    if name == "_WITHOUT":
        return f"Traigo doce tarimas, {w} sello MX-48213.", (12, None, f"{REPORTED}: '{w} sello'")
    if name == "_MISSING_CHAIN" and w in ("uno", "una", "one"):  # a count after "no" competes too
        return (f"Doce tarimas. Sello MX-48213. No {w}.",
                (None, None, f"counts disagree in transcript: [1, 12] (self-correction: 'no'); "
                             f"{REPORTED}: 'sello mx-48213 . no {w}'"))
    if name == "_MISSING_CHAIN":
        return f"Doce tarimas. Sello MX-48213. No {w}.", (12, None, f"{REPORTED}: 'sello mx-48213 . no {w}'")
    if name == "_CLAUSE_JOINS":
        return (f"Doce tarimas. Sello MX-48213, no está {w} ya.",
                (12, None, f"{REPORTED}: 'sello mx-48213 , no está'"))
    if name == "_RETRACTIONS" and w == "that s not it":  # "not ... it" also says the seal is missing
        return ("Doce tarimas. Sello MX-48213. That's not it.",
                (None, None, f"count taken back in transcript: 'that s not it .'; "
                             f"{REPORTED}: 'sello mx-48213 . that s not it'"))
    if name == "_RETRACTIONS" and w.endswith("mal"):  # "mal" also reports the seal
        return (f"Doce tarimas. Sello MX-48213. {w.capitalize()}.",
                (None, None, f"count taken back in transcript: '{w} .'; {REPORTED}: 'sello mx-48213 . {w}'"))
    if name == "_RETRACTIONS" and w.startswith("wrong "):  # "wrong" also reports the seal
        return (f"Doce tarimas. Sello MX-48213. {w.capitalize()}.",
                (None, None, f"count taken back in transcript: '{w} .'; {REPORTED}: 'sello mx-48213 . wrong'"))
    if name in ("_RETRACTIONS", "_RETRACTIONS_AT_END"):  # after both values, in a later sentence: both taken back
        return (f"Doce tarimas. Sello MX-48213. {said(w).capitalize()}.",
                (None, None, f"count taken back in transcript: '{w} .'; seal taken back in transcript: '{w} .'"))
    if name == "_DONT_USE_WHAT":
        return (f"Doce tarimas. Sello MX-48213. No {w} apuntes.",
                (None, None, f"count taken back in transcript: 'no {w} apuntes .'; seal taken back in transcript: "
                             f"'no {w} apuntes .'"))
    if name == "_DONT_USE":
        text, low = ((f"Don't {w} that.", f"don t {w} that .") if w.isascii() and not w.endswith(("es", "en"))
                     else (f"No lo {w}.", f"no lo {w} ."))
        return (f"Doce tarimas. Sello MX-48213. {text}",
                (None, None, f"count taken back in transcript: '{low}'; seal taken back in transcript: '{low}'"))
    if name == "_WRONG":
        return f"Doce tarimas. Las doce están {w}.", (None, None, f"negated count in transcript: 'doce están {w}'")
    if name == "_COPULAS":
        return (f"Doce tarimas. Sello MX-48213. No {w} MX.",
                (12, None, f"negated seal in transcript: 'sello mx-48213 . no {w} mx'"))
    if name == "_RETRACT_PRONOUNS":
        return (f"Doce tarimas. Sello MX-48213. {w.capitalize()} de la otra caja.",
                (None, None, f"count taken back in transcript: '{w} de la otra caja .'; "
                             f"seal taken back in transcript: '{w} de la otra caja .'"))
    if name == "_RETRACT_PLACES" and w in adapters._SEAL_BAD:
        return (f"Doce tarimas. Sello MX-48213. Ese es el {w}.",
                (None, None, f"count taken back in transcript: 'ese es el {w} .'; "
                             f"{REPORTED}: 'sello mx-48213 . ese es el {w}'"))
    if name == "_RETRACT_PLACES":
        return (f"Doce tarimas. Sello MX-48213. Ese es el {w}.",
                (None, None, f"count taken back in transcript: 'ese es el {w} .'; "
                             f"seal taken back in transcript: 'ese es el {w} .'"))
    if name == "_SEAL_MATCH":
        return f"Traigo doce tarimas, sello MX-48213, no {w}.", (12, None, f"{REPORTED}: 'sello mx-48213 , no {w}'")
    if name == "_MATCH_GAP":
        return f"Twelve pallets, it matches {w} the paperwork, seal MX-48213.", (12, "MX-48213", "")
    if name == "_NEGATION_GAP" and w == "be":  # "be" is also the letter B
        return "Twelve pallets, seal MX-48213, it will not be broken.", (12, "MX-48213", "")
    if name == "_NEGATION_GAP" and w in ("un", "una"):  # after "no", "un"/"una" may be a count
        return f"Doce tarimas. Sello MX-48213 no viene {w} rota.", (12, "MX-48213", "")
    if name == "_NEGATION_GAP":
        return f"Traigo doce tarimas, sello MX-48213, no {w} roto.", (12, "MX-48213", "")
    if name == "_HAZMAT":
        return f"Traigo doce tarimas, el sello es MX-48213 y no hay material {w}.", (12, "MX-48213", "")
    if name == "_CARGO_GAP":
        return f"Doce tarimas, sello MX-48213 no {w} peligroso.", (12, "MX-48213", "")
    if name == "_SEAL_BEFORE_SKIP":
        return f"Traigo doce tarimas, no {w} sello MX-48213.", (12, None, f"negated seal in transcript: 'no {w} sello'")
    if name == "_SPELLED":
        return (f"Traigo doce tarimas, sello eme {w} guión {SEAL_DIGITS}.",
                (12, f"M{adapters._SPELLED[w]}-48213", ""))
    if name == "_EN_SPELLED":
        return (f"Twelve pallets, seal zee {w} dash four eight two one three.",
                (12, f"Z{adapters._EN_SPELLED[w]}-48213", ""))
    if name == "_RADIO":
        other = "lima" if w != "lima" else "mike"
        return (f"Twelve pallets. Seal {w}, {other}, four four one two.",
                (12, f"{adapters._RADIO[w]}{adapters._RADIO[other]}4412", ""))
    if name == "_AS_IN":
        return f"Twelve pallets. Seal T {w} Tom, four four one two.", (12, "T4412", "")
    if name == "_NOT_SEAL":
        return f"Traigo doce tarimas, sello MX-48213 {w}.", (12, "MX-48213", "")
    if name == "_SENTENCE_WORDS":
        return f"Doce tarimas. Sello MX-48213. {w} la caja 4471.", (12, "MX-48213", "")
    if name == "_SIZE_UNITS":
        return f"Twelve pallets. Seal MX-48213. It's a 53-{w} dry van.", (12, "MX-48213", "")
    if name == "_WORD_LETTERS":  # inside an id; as its first letter they give None (_LETTER_NAME_WORDS)
        return {"a": (f"Traigo doce tarimas, sello eme a guión {SEAL_DIGITS}.", (12, "MA-48213", "")),
                "te": (f"Traigo doce tarimas, sello eme te equis guión {SEAL_DIGITS}.", (12, "MTX-48213", "")),
                "ese": (f"Traigo doce tarimas, sello eme ese equis guión {SEAL_DIGITS}.", (12, "MSX-48213", "")),
                "em": ("Twelve pallets, seal ex em ex dash four eight two one three.", (12, "XMX-48213", "")),
                "el": ("Twelve pallets, seal ex el zee dash four four one two.", (12, "XLZ-4412", "")),
                "de": (f"Traigo doce tarimas, sello MX guión {SEAL_DIGITS} guión de.", (12, "MX-48213-D", ""))}[w]
    if name == "_LETTER_NAME_WORDS":  # ORDER Nube-10 item 6
        return {"el": ("Twelve pallets. The seal is el zee dash four four one two.",
                       (12, None, "unclear seal letters: 'seal is el zee'")),
                "es": (f"Doce tarimas. El sello es eme equis {SEAL_DIGITS}.",
                       (12, None, "unclear seal letters: 'sello es eme'")),
                "ese": (f"Doce tarimas. Sello ese eme equis {SEAL_DIGITS}.",
                        (12, None, "unclear seal letters: 'sello ese eme'")),
                "te": (f"Doce tarimas. Sello te equis guión {SEAL_DIGITS}.",
                       (12, None, "unclear seal letters: 'sello te equis'")),
                "a": (f"Doce tarimas. Sello a be guión {SEAL_DIGITS}.",
                      (12, None, "unclear seal letters: 'sello a be'")),
                "see": ("Twelve pallets. The seal is, see, MX-48213.",
                        (12, None, "unclear seal letters: 'seal is , see , mx-48213'")),
                "em": ("Twelve pallets. Seal em ex dash four eight two one three.",
                       (12, None, "unclear seal letters: 'seal em ex'")),
                "why": ("Twelve pallets. Seal why ex dash four eight two one three.",
                        (12, None, "unclear seal letters: 'seal why ex'"))}[w]
    if name == "_NOT_SEAL_LETTERS" and w in ("ya", "si", "ok"):  # read as letters they would be "YA", never kept
        return (f"Traigo doce tarimas, sello {w} guión {SEAL_DIGITS}.",
                (12, None, f"words between the seal and its id: 'sello {w} guión cuatro'"))
    if name == "_NOT_SEAL_LETTERS" and w in ("mas", "per"):  # after a count "mas" adds to it, "per" is a source
        why = SEAL_FROM if w == "per" else "seal said but no id read"
        return f"Sello MX {w} guión {SEAL_DIGITS}. Traigo doce tarimas.", (12, None, f"{why}: 'sello mx {w}'")
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
                "dash": ("No — doce tarimas.", (12, None, ""))}[w]
    if name == "_ES_SPLIT":
        ones, value = SPLIT[w]
        return f"Traigo {w} {ones} tarimas.", (value, None, "")
    if name == "_ADDED":
        return (f"Traigo doce tarimas y {w}.",
                (None, None, f"more than the count said in transcript: 'doce tarimas y {w}'"))
    if name == "_TELLING":
        return f"{w.capitalize()} doce tarimas. Sello MX-48213.", (None, "MX-48213", f"{FROM}: '{w} doce tarimas'")
    if name == "_GIVING":
        return f"Me {w} doce tarimas. Sello MX-48213.", (None, "MX-48213", f"{FROM}: '{w} doce tarimas'")
    if name == "_GIVEN_TO":
        return (f"{w.capitalize()} pasó doce tarimas. Sello MX-48213.",
                (None, "MX-48213", f"{FROM}: 'pasó doce tarimas'"))
    if name == "_GIVEN_WHAT":
        return f"Me {w} pasó doce tarimas. Sello MX-48213.", (None, "MX-48213", f"{FROM}: 'pasó doce tarimas'")
    if name == "_SCOPING":
        text = {"per": "Per the bill", "according": "According to the bill", "según": "Según la carta",
                "segun": "Segun la carta"}[w]
        count = "twelve pallets" if w in ("per", "according") else "doce tarimas"
        return f"{text}, {count}.", (None, None, f"{FROM}: '{text.lower()} , {count}'")
    if name == "_DOCS":
        return f"The {w} has twelve pallets. Seal MX-48213.", (None, "MX-48213", f"{FROM}: '{w} has twelve pallets'")
    if name == "_ARTICLE_DOCS":
        return f"Twelve pallets de {w}. Seal MX-48213.", (12, "MX-48213", "")
    if name == "_DOC_DETERMINERS":
        return f"Twelve pallets, {w} papel. Seal MX-48213.", (None, "MX-48213", f"{FROM}: 'twelve pallets , {w} papel'")
    if name == "_PARTIES":
        return (f"The {w} has twelve pallets. Seal MX-48213.",
                (None, "MX-48213", f"{FROM}: '{w} has twelve pallets'"))
    if name == "_SOURCE_PHRASES":
        if w == "other driver":
            return "The other driver has twelve pallets.", (None, None, f"{FROM}: 'other driver has twelve pallets'")
        return f"El {w} trae doce tarimas.", (None, None, f"{FROM}: '{w} trae doce tarimas'")
    if name == "_PARTY_ACTS":
        return (f"Traigo doce tarimas del cliente, {w}. Sello MX-48213.",
                (None, "MX-48213", f"{FROM}: 'doce tarimas del cliente'"))
    if name == "_ACTS_BY_OTHERS":
        return (f"Somebody {w} twelve pallets. Seal MX-48213.",
                (None, "MX-48213", f"{FROM}: '{w} twelve pallets'"))
    if name == "_FIRST_PERSON":
        return f"Twelve pallets, {w} counted them. Seal MX-48213.", (12, "MX-48213", "")
    if name == "_AUXILIARIES":
        return f"Twelve pallets, I {w} counted them. Seal MX-48213.", (12, "MX-48213", "")
    if name == "_DRIVER_ACTING":
        return f"Traigo doce tarimas, vengo {w} la planta. Sello MX-48213.", (12, "MX-48213", "")
    if name == "_SEAL_PARTIAL":
        return (f"Traigo doce tarimas, sello MX-48213, {w}.",
                (12, None, f"part of a seal said in transcript: 'sello mx-48213 , {w}'"))
    if name == "_SOMETHING":
        return (f"Traigo doce tarimas, sello MX-48213 {w}.",
                (12, None, f"part of a seal said in transcript: 'sello mx-48213 {w}'"))
    if name == "_LETTER_WORDS":
        return (f"Traigo doce tarimas, sello MX-48213, {w} a.",
                (12, None, f"seal letter said after the digits: 'sello mx-48213 , {w}'"))
    if name == "_SEALING":
        return f"Twelve pallets, not {w}.", (12, None, f"{REPORTED}: 'pallets , not {w}'")
    if name == "_TRAILER":
        return f"Twelve pallets, the {w} is broken.", (12, None, f"{REPORTED}: 'the {w} is broken'")
    if name == "_TRAILER_ARTICLES":
        return (f"{w.capitalize()} caja viene abierta. Doce tarimas. Sello MX-48213.",
                (12, None, f"{REPORTED}: '{w} caja viene abierta'"))
    if name == "_OPEN":
        return "The door is open. Twelve pallets. Seal MX-48213.", (12, None, f"{REPORTED}: 'the door is open'")
    if name == "_COUNT_DOUBT":
        text = said(w)
        if w.startswith("no "):
            return f"Traigo doce tarimas, {text}.", (None, None, "count not checked in transcript: 'doce tarimas'")
        return f"Twelve pallets, I have {text}.", (None, None, "count not checked in transcript: 'twelve pallets'")
    if name == "_UNREAD":
        return (f"Traigo doce tarimas. Sello MX-48213. El {w}.",
                (12, None, f"seal not read in full in transcript: 'sello mx-48213 . el {w}'"))
    if name == "_UNREAD_EXCEPTIONS":
        return f"Doce tarimas. Sello MX-48213. {w.capitalize()}.", (12, "MX-48213", "")
    if name == "_CLITICS":
        return f"Twelve pallets. The seal'{w} 48213.", (12, "48213", "")
    if name == "_ARTICLE_WORDS":
        if w in ("a", "an"):
            return (f"Twelve pallets. The seal is 4471. Sorry, {w} hundred.",
                    (None, "4471", "counts disagree in transcript: [12, 100] (self-correction: 'sorry')"))
        if w == "one":
            return ("Twelve pallets. The seal is 4471. Sorry, one.",
                    (None, "4471", "counts disagree in transcript: [1, 12] (self-correction: 'sorry')"))
        return (f"Doce tarimas. El sello es 4471. Perdón, {w}.",
                (None, "4471", "counts disagree in transcript: [1, 12] (self-correction: 'perdón')"))
    if name == "_INDEFINITE":
        if w in ("un", "una"):
            return (f"Doce tarimas y {w} caja.", (None, None, f"more than the count said in transcript: "
                                                              f"'doce tarimas y {w}'"))
        return f"Twelve pallets and {w} crate.", (None, None, f"more than the count said in transcript: "
                                                             f"'twelve pallets and {w}'")
    if name == "_EN_INDEFINITE":
        return f"I have {w} hundred pallets.", (100, None, "")
    if name == "_ES_INDEFINITE":
        return (f"Doce tarimas, sello {w} cuatro ocho dos uno tres.",
                (12, None, f"words between the seal and its id: 'sello {w} cuatro'"))
    if name == "_THOUSAND":
        return ((f"Traigo 4 {w} tarimas.", (4000, None, "")) if w == "mil"
                else (f"I have 4 {w} pallets.", (4000, None, "")))
    if name == "_REPEATS":
        return (f"Traigo doce tarimas, sello MX guión {w} ocho dos uno tres.",
                (12, "MX-" + ("888" if w == "triple" else "88") + "213", ""))
    if name == "_AND_A_HALF":
        return (("Traigo dos docenas y media de cajas.", (30, None, "")) if w == "y media"
                else ("I have two dozen and a half boxes.", (30, None, "")))
    if name == "_DE_ARTICLES":
        return f"doce tarimas sello mx guión {SEAL_DIGITS} de {w} caja", (12, "MX-48213", "")
    if name == "_AND":
        return f"Doce tarimas. Llegué a las ocho {w} diez. Sello MX-48213.", (12, "MX-48213", "")
    raise KeyError(name)


LISTS = [n for n in voice_mutants.list_names() if n not in voice_mutants.NOT_LISTS]

# Every word of every list, written out (ORDER Nube-10 item 9): a word deleted from the parser fails the first test.
WORDS: dict[str, list[str]] = {
    "_UNITS": ["pallets?", "palets?", "pal[eé]s", "payets?", "pajets?", "payettes?", "paletas?", "tarimas?", "skids?",
        "cajas?", "boxes", "box", "bultos?", "piezas?", "pieces?", "drums?", "cartons?", "cases?", "rollos?",
        "tambos?", "bobinas?", "crates?", "coils?", "garraf[oó]n", "garrafones", "cubetas?", "bundles?", "sacos?",
        "bags?", "costales?", "rolls?", "reels?", "totes?", "barrels?", "barriles?", "bins?", "racks?", "pacas?",
        "bales?", "huacales?", "bidones?", "sacks?", "carretes?", "paquetes?", "packages?", "tires?"],
    "_WEIGHT_UNITS": ["kilos", "kilo", "kg", "kgs", "kilogramos", "kilogramo", "kilograms", "kilogram", "toneladas",
        "tonelada", "ton", "tons", "tonnes", "libras", "libra", "lb", "lbs", "pounds", "pound"],
    "_SEAL_DASH": ["guión", "guion", "raya", "dash", "hyphen"],
    "_ES_ONES": ["un", "uno", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
    "_ES_TEENS": ["once", "doce", "trece", "catorce", "quince", "dieciséis", "dieciseis", "diecisiete", "dieciocho",
        "diecinueve", "veintiuno", "veintiún", "veintiun", "veintiuna", "veintidós", "veintidos", "veintitrés",
        "veintitres", "veinticuatro", "veinticinco", "veintiséis", "veintiseis", "veintisiete", "veintiocho",
        "veintinueve"],
    "_ES_SPLIT": ["veinti", "dieci"],
    "_ES_TENS": ["diez", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"],
    "_ES_HUNDREDS": ["ciento", "doscientos", "doscientas", "trescientos", "trescientas", "cuatrocientos",
        "cuatrocientas", "quinientos", "quinientas", "seiscientos", "seiscientas", "setecientos", "setecientas",
        "ochocientos", "ochocientas", "novecientos", "novecientas"],
    "_EN_ONES": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "_EN_TEENS": ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"],
    "_EN_TENS": ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"],
    "_NUMBER_WORDS": ["cero", "zero", "cien", "mil", "hundred", "thousand"],
    "_DOZEN": ["docena", "dozen"],
    "_DOZENS": ["docenas", "dozens"],
    "_ID_NOUNS": ["tráiler", "trailer", "camión", "camion", "unidad", "andén", "anden", "puerta", "rampa", "bahía",
        "bahia", "muelle", "caja", "ruta", "viaje", "factura", "pedido", "orden", "folio", "número", "numero",
        "placa", "placas", "turno", "línea", "linea", "carril", "truck", "dock", "door", "bay", "unit", "route",
        "trip", "invoice", "order", "number", "plate", "lane", "gate", "po", "operador", "operator", "licencia",
        "license", "teléfono", "telefono", "phone", "celular", "cell", "cel", "bol", "pro", "ref", "cortina", "load",
        "compra", "slot", "spot", "plates", "tracto", "tractor", "chassis", "chasis", "container", "contenedor",
        "highway", "hwy", "interstate", "carretera", "autopista", "freeway", "exit", "salida"],
    "_NUMBER_NOUNS": ["number", "número", "numero"],
    "_NUMBERED": ["tráiler", "trailer", "camión", "camion", "unidad", "andén", "anden", "puerta", "rampa", "bahía",
        "bahia", "muelle", "caja", "ruta", "viaje", "factura", "pedido", "orden", "folio", "placa", "placas", "turno",
        "línea", "linea", "carril", "truck", "dock", "door", "bay", "unit", "route", "trip", "invoice", "order",
        "plate", "lane", "gate", "po", "operador", "operator", "licencia", "license", "teléfono", "telefono", "phone",
        "celular", "cell", "cel", "bol", "pro", "ref", "reference", "referencia", "tracking", "confirmation",
        "confirmación", "confirmacion", "booking", "load", "carga", "pickup", "delivery", "entrega", "appointment",
        "cita", "cuenta", "account", "pedimento", "guía", "guia", "control", "employee", "empleado", "badge",
        "gafete", "móvil", "movil", "mobile", "tractor", "chassis", "chasis", "container", "contenedor"],
    "_ID_ARTICLES": ["el", "la", "los", "las", "the", "my", "mi", "su", "your", "tu", "his", "her", "its", "our",
        "nuestro", "nuestra", "an", "un", "una"],
    "_CORRECTIONS": ["no", "perdón", "perdon", "digo", "corrijo", "mejor dicho", "más bien", "mas bien", "o sea",
        "bueno", "me equivoqué", "me equivoque", "quise decir", "sorry", "wait", "i mean", "actually", "or rather",
        "rather", "make that", "correction", "scratch that", "make it", "mejor", "nope", "oops", "whoops", "my bad",
        "strike that", "let me recount", "error", "no es cierto", "en realidad", "aguanta", "conté mal", "conte mal",
        "déjame contar", "dejame contar", "le corrijo", "me corrijo"],
    "_APPROX_BEFORE": ["como", "unas", "unos", "casi", "aproximadamente", "aprox", "alrededor de", "cerca de",
        "más de", "mas de", "menos de", "más o menos", "mas o menos", "al menos", "por lo menos", "hasta",
        "por ahí de", "about", "around", "close to", "por ahi de", "roughly", "approximately", "approx", "nearly",
        "give or take", "almost", "some", "like", "over", "under", "more than", "less than", "fewer than", "at least",
        "up to"],
    "_APPROX_AFTER": ["más o menos", "mas o menos", "o así", "o asi", "aproximadamente", "o algo así", "y pico",
        "or so", "roughly", "approximately", "give or take", "more or less"],
    "_HEDGES": ["creo", "supongo", "pienso", "parece", "quizá", "quizás", "quiza", "quizas", "tal vez", "talvez",
        "a lo mejor", "probablemente", "posiblemente", "calculo", "puede que", "según yo", "segun yo",
        "supuestamente", "han de ser", "deben ser", "deben de ser", "debe ser", "no estoy seguro", "no estoy segura",
        "si no me equivoco", "si mal no recuerdo", "si no mal recuerdo", "no me acuerdo", "no lo veo", "no se si",
        "no sé", "que yo sepa", "hasta donde sé", "hasta donde se", "casi seguro", "casi segura", "igual y",
        "se supone", "ponle", "diría", "diria", "me late", "digamos", "maybe", "perhaps", "probably", "possibly",
        "think", "guess", "believe", "might", "may be", "pretty sure", "fairly sure", "want to say", "not sure",
        "could be", "supposedly", "supposed", "should be", "guessing", "if i remember", "if i m not mistaken",
        "if i read", "if i counted", "or something", "don t quote me", "can t really read", "can t read", "i d say",
        "i would say", "reckon", "suppose", "as far as i know", "almost sure", "almost certain", "if i m not wrong",
        "if i am not wrong", "unless i m wrong", "don t know", "ish", "if i m reading", "not a hundred percent",
        "not one hundred percent", "not 100", "no estoy muy seguro", "no estoy muy segura", "no estoy tan seguro",
        "no estoy tan segura", "si las conté", "si los conté", "si las conte", "si los conte", "si no conté mal",
        "not positive", "not certain", "no se bien", "más o menos", "mas o menos", "o así", "o asi", "o algo así",
        "o algo asi", "y pico", "or so", "give or take", "more or less", "aproximadamente", "approximately",
        "roughly", "approx", "aprox", "something like", "algo así como", "ballpark", "estimate", "estimated",
        "estimo", "a ojo", "al tanteo", "a lo mucho", "cuando mucho", "at most", "at the most", "si es que",
        "si lo leí", "si lo lei", "aproximado", "aproximada"],
    "_NEGATIONS": ["no", "ni", "nunca", "tampoco", "not", "never", "nor"],
    "_NEGATED_CONTRACTIONS": ["don", "doesn", "didn", "isn", "aren", "wasn", "weren", "haven", "hasn", "won", "can",
        "couldn", "wouldn", "shouldn", "ain"],
    "_COUNT_VERBS": ["traigo", "trae", "traemos", "llevo", "lleva", "llevamos", "tengo", "tenemos", "son", "eran",
        "serían", "serian", "vienen", "van", "cargo", "cargué", "cargue", "hay", "entrego", "bring", "have", "got",
        "carrying", "carry", "are"],
    "_ALTERNATIVES": ["o", "u", "ó", "or", "y", "e", "and", "a", "to", "ni", "nor"],
    "_AFTER_CORRECTION": ["son", "eran", "es", "serían", "serian", "fueron", "traigo", "tengo", "llevo", "are", "is",
        "it", "s", "that", "i", "have", "got", "we", "were", "was"],
    "_SPEECH_FILLERS": ["eh", "ehh", "em", "emm", "mm", "mmm", "um", "uh", "uhm", "este", "hmm", "hm", "mhm", "ah",
        "ahh", "er", "erm", "pos", "pues", "espérame", "esperame", "espera", "oye", "mmh", "mmhm", "umm", "uhh",
        "ehm"],
    "_FILLER_PHRASES": ["a ver", "déjame ver", "dejame ver", "let me see", "let s see"],
    "_NOT_FILLER": ["con", "sin", "por", "para", "al", "hasta", "desde", "entre", "sobre", "cada", "with", "without",
        "in", "on", "for", "to", "from", "at", "by", "per", "each", "el", "la", "los", "las", "lo", "the", "and",
        "pero", "but", "si", "if", "que"],
    "_COUNT_AFTER": ["total", "completas", "completos", "altogether", "all together", "exactas", "exactos", "justas",
        "justos", "now", "ahora"],
    "_MORE": ["más", "mas", "more", "extra", "menos", "less", "fewer"],
    "_ARTICLES": ["un", "una", "one"],
    "_ARTICLE_NOUNS": ["saludo", "saludos", "abrazo", "favor", "momento", "pregunta", "question", "moment", "second",
        "segundo", "minute", "minuto", "cosa", "thing"],
    "_PRONOUN_BEFORE": ["good", "great", "nice", "safe", "long", "last", "next", "other", "same", "right", "first",
        "this", "that", "which", "every", "another", "quick", "easy", "short", "small", "big", "little", "busy",
        "slow", "fast", "tough", "rough", "quiet", "smooth", "blessed"],
    "_COUNT_CHANGE": ["short", "faltante", "faltantes", "missing", "falta", "faltan", "suelta", "sueltas", "suelto",
        "sueltos", "loose", "adicional", "adicionales", "additional", "sobrante", "sobrantes", "par", "pair",
        "couple"],
    "_ADVERBS": ["tops", "flat", "even", "straight", "nomás", "nomas", "ahorita", "again", "also", "too", "though"],
    "_ADDED": ["más", "mas", "plus", "además", "aparte", "another", "otra", "otras", "otro", "otros", "extra",
        "media", "half", "cacho", "change", "partial", "parcial", "incompleta", "incompleto", "incompletas",
        "incompletos", "suelta", "sueltas", "suelto", "sueltos", "loose", "falta", "faltan", "faltó", "falto",
        "faltaba", "missing", "short"],
    "_TELLING": ["dice", "dicen", "dijo", "dijeron", "decía", "decia", "decían", "decian", "diga", "digan", "dicho",
        "says", "say", "said", "told", "tell", "tells", "telling", "saying", "gave", "give", "gives", "giving",
        "sent", "send", "sends", "texted", "emailed", "informó", "informaron", "avisó", "avisaron", "reportó",
        "reportaron", "comentó", "comentaron", "contó", "contaron", "contaba", "reported", "mentioned", "entregó",
        "entregaron", "dictó", "dictaron", "leyó", "leyeron"],
    "_GIVING": ["pasó", "pasaron", "pasa", "pasan", "dio", "dió", "dieron", "da", "dan", "mandó", "mandaron", "manda",
        "mandan"],
    "_GIVEN_TO": ["me", "nos", "le", "les", "te"],
    "_GIVEN_WHAT": ["lo", "la", "los", "las"],
    "_SCOPING": ["según", "segun", "according", "per"],
    "_DOCS": ["docs", "doc", "documents", "documento", "documentos", "paperwork", "papeles", "papel", "papers",
        "paper", "sheet", "hoja", "carta", "guía", "guia", "remisión", "remision", "sistema", "system", "app",
        "aplicación", "aplicacion", "bol", "bill", "manifest", "manifiesto", "factura", "ticket", "pantalla",
        "screen", "tablet", "scanner", "portal"],
    "_ARTICLE_DOCS": ["papel", "papeles", "paper", "papers", "bill", "hoja", "carta", "sistema", "system", "ticket",
        "guía", "guia", "pantalla", "screen"],
    "_DOC_DETERMINERS": ["el", "la", "los", "las", "the", "my", "mi", "su", "this", "that", "ese", "esa", "este",
        "esta", "your", "tu"],
    "_PARTIES": ["dispatch", "despacho", "despachador", "dispatcher", "shipper", "remitente", "cliente", "customer",
        "client", "montacarguista", "planta", "plant", "compañero", "compañera", "partner", "guardia", "guard",
        "caseta", "oficina", "office", "supervisor", "jefe", "boss", "gerente", "manager", "checker", "receiving",
        "receiver", "broker", "warehouse", "almacén", "almacen", "bodega", "forklift", "montacargas", "lumper",
        "lumpers", "loader", "loaders"],
    "_SOURCE_PHRASES": ["otro chofer", "other driver", "otro operador"],
    "_PARTY_ACTS": ["count", "counts", "sealed", "selló", "sellaron", "checked", "revisó", "reviso", "revisaron",
        "cargaron", "cargó", "loaded"],
    "_ACTS_BY_OTHERS": ["counted"],
    "_FIRST_PERSON": ["i", "we"],
    "_AUXILIARIES": ["have", "haven", "had", "hadn", "t", "not", "did", "didn", "ve", "d", "just", "already", "also",
        "never"],
    "_DRIVER_ACTING": ["de", "del", "desde", "from", "for", "para", "a", "al", "to", "at", "en", "in", "con", "with",
        "hacia", "toward", "of", "pasé", "pase", "pasamos", "crucé", "cruce", "cruzamos", "passed", "crossed", "left",
        "dejé", "deje"],
    "_COUNT_DOUBT": ["no las conté", "no los conté", "no las conte", "no los conte", "didn t count", "did not count",
        "didn t check", "did not check", "haven t counted", "no vi la carga", "not looked", "haven t looked"],
    "_NOT_COUNT_BEFORE": ["las", "la", "at", "cada", "each", "the", "minus", "negative", "since", "desde"],
    "_TIME_BEFORE": ["quarter to", "quarter till", "past", "cuarto para"],
    "_ONCE": ["for once", "just once", "only once", "once again", "once in a while"],
    "_WEIGHT_WORDS": ["peso", "pesa", "pesan", "pesamos", "weight", "weighs", "weigh", "weighing"],
    "_WEIGHT_GAP": ["es", "is", "de", "of", "del", "bruto", "bruta", "gross", "neto", "neta", "net", "total", "son",
        "are", "was", "era", "como", "about", "around", "unos", "unas", "aproximadamente", "approximately", "casi",
        "almost", "roughly", "right", "just", "justo", "nearly", ","],
    "_ID_LEAD": ["es", "is", "el", "la", "los", "las", "the", "de", "del", "of", "nº", "en", "in", "on", "a", "su",
        "mi", "my", "son", "are", "was", "era"],
    "_POSSESSIVES": ["mi", "my", "su", "your", "tu", "his", "her"],
    "_SEAL_KEYWORDS": ["sello", "seal", "precinto", "marchamo"],
    "_SEAL_PLURALS": ["sellos", "seals", "precintos", "marchamos"],
    "_SEAL_LEAD_IN": ["es", "is", "el", "the", "número", "numero", "núm", "number", "nº", "reads"],
    "_SEAL_ABBREVIATIONS": ["no", "num", "núm", "nro", "n"],
    "_SEAL_BAD": ["roto", "rota", "rotos", "rotas", "dañado", "dañada", "dañados", "dañadas", "daño", "daños",
        "damage", "violado", "violada", "violentado", "violentada", "forzado", "forzada", "abierto", "abierta",
        "cortado", "cortada", "quebrado", "quebrada", "alterado", "alterada", "despegado", "despegada", "falta",
        "faltan", "faltó", "falto", "faltaba", "faltante", "perdido", "perdida", "perdí", "perdi", "perdió", "perdio",
        "diferente", "distinto", "distinta", "broken", "broke", "damaged", "tampered", "cut", "opened", "missing",
        "lost", "removed", "gone", "mismatch", "mismatched", "wrong", "different", "lacks", "lacking", "loose",
        "suelto", "suelta", "flojo", "floja", "violaron", "violó", "violo", "cortaron", "rompieron", "rompió",
        "rompio", "abrieron", "bad", "mal", "cracked", "busted", "snapped", "ripped", "torn", "bent", "popped",
        "twisted", "pulled", "sheared", "quitaron", "quitado", "quitada", "arrancaron", "arrancado", "arrancada",
        "reventado", "reventada", "chueco", "chueca", "doblado", "doblada", "colgando", "desprendido", "desprendida",
        "cambiaron", "cambiado", "cambiada", "zafado", "zafada", "hole", "holes", "agujero", "agujeros", "hoyo",
        "hoyos", "perforado", "perforada", "tronado", "tronada", "tronó", "estrellado", "estrellada", "golpeado",
        "golpeada", "golpearon", "mordido", "mordida", "partieron", "desmontaron", "soldaron", "soldadura", "smashed",
        "chewed", "glued", "pried", "drilled", "welded"],
    "_SEAL_BAD_PHRASES": ["se cayó", "se cayo", "se lo llevaron", "se la llevaron", "no sirve", "se zafó", "se zafo",
        "came off", "fell off", "taken off", "took it off", "is off", "s off", "was off", "se metió", "se metio",
        "se metieron", "se sale", "partido por", "partida por", "le sacaron", "tirado en",
        "no estaba en", "quisieron abrir", "se salió", "se salio", "comes off"],
    "_BAD_NOT_SEAL": ["pulled in", "pulled into", "pulled up", "pulled over", "wrong turn", "wrong way", "wrong exit",
        "bad traffic", "traffic was bad", "traffic is bad", "bad weather", "weather was bad", "too bad", "cut me off"],
    "_CLAUSE_BAD": ["off"],
    "_AFFIRMING_NEGATIONS": ["nothing", "nada", "sin", "without", "ningún", "ningun", "ninguno", "ninguna"],
    "_CONDITIONS": ["if", "si", "unless"],
    "_CONDITION_GAP": ["i", "m", "am", "it", "s", "is"],
    "_WITHOUT": ["sin", "without"],
    "_MISSING_CHAIN": ["está", "esta", "estaba", "están", "estan", "hay", "había", "habia", "tengo", "tiene",
        "tienen", "trae", "traigo", "traemos", "viene", "vienen", "veo", "encuentro", "encontré", "encontre", "find",
        "found", "see", "have", "has", "had", "got", "is", "was", "are", "there", "here", "it", "lo", "la", "le",
        "one", "uno", "una"],
    "_CLAUSE_JOINS": ["y", "and", "pero", "but", "so"],
    "_COPULAS": ["es", "era"],
    "_RETRACTIONS": ["mentira", "olvídalo", "olvidalo", "olvida", "olvídate", "olvidate", "olvídense", "olvidense",
        "borra", "táchalo", "tachalo", "tacha", "retiro", "retirar", "retiré", "retire lo", "no me hagas caso",
        "no le hagas caso", "no hagas caso", "no le haga caso", "nada de eso", "ese no", "esa no", "no es ese",
        "no es esa", "me equivoqué", "me equivoque", "me confundí", "me confundi", "conté mal", "conte mal",
        "está mal", "esta mal", "forget it", "forget that", "forget both", "forget all", "forget everything",
        "forget those", "forget about", "never mind", "nevermind", "ignore it", "ignore that", "ignore the",
        "ignore those", "disregard", "take back", "take it back", "take that back", "taking back", "scratch that",
        "scratch the", "strike that", "strike all", "cancel that", "none of that", "miscounted", "i messed up",
        "that s not it", "not that one", "wrong load", "wrong paperwork", "wrong trailer", "wrong number",
        "throw that", "throw all", "throw it", "throw those", "scratch both", "scratch all", "scratch those",
        "scratch it", "mixed up", "mixed it up", "turned around", "al revés", "al reves", "backwards", "no te sirve",
        "no les sirve", "not the count", "not the number", "no es el número", "no es el numero", "way off",
        "wrong count", "carga equivocada", "número equivocado", "numero equivocado"],
    "_RETRACTIONS_AT_END": ["not right", "isn t right", "s not right", "no va", "no es así", "no es asi",
        "no es correcto", "not correct", "no es", "isn t correct"],
    "_RETRACT_PRONOUNS": ["ese", "esa", "eso", "that", "count", "conteo", "cuenta"],
    "_RETRACT_PLACES": ["otra", "otro", "otras", "otros", "other", "another", "last", "previous", "anterior",
        "pasada", "pasado", "ayer", "yesterday", "old", "vieja", "viejo", "equivocado", "equivocada", "wrong",
        "before", "earlier", "antes", "entregué", "delivered"],
    "_DONT_USE_WHAT": ["lo", "la", "los", "las", "eso", "that", "it", "this", "those"],
    "_DONT_USE": ["use", "log", "write", "record", "go", "take", "enter", "put", "apuntes", "anotes", "tomes", "uses",
        "registres", "pongas", "escribas", "captures", "apunten", "anoten", "tomen", "usen", "registren"],
    "_WRONG": ["wrong", "mal", "off", "incorrect", "incorrecto", "incorrecta", "equivocado", "equivocada"],
    "_SEAL_PARTIAL": ["last", "últimos", "últimas", "ultimos", "ultimas"],
    "_SOMETHING": ["algo", "something"],
    "_UNREAD": ["resto", "rest", "borroso", "borrosa", "smudged", "scratched", "ilegible", "illegible", "unreadable",
        "faded", "despintado", "despintada"],
    "_LETTER_WORDS": ["letra", "letras", "letter", "letters"],
    "_TRAILER": ["caja", "tráiler", "trailer", "remolque", "puerta", "puertas", "door", "doors"],
    "_OPEN": ["open"],
    "_SEALING": ["sealed", "unsealed", "sellar", "sellaron", "sellado", "sellada", "sellados", "selladas"],
    "_MATCH_GAP": ["con", "with", "as", "que", "to"],
    "_SEAL_MATCH": ["coincide", "coinciden", "corresponde", "corresponden", "cuadra", "cuadran", "concuerda", "match",
        "matches", "matched", "matching", "agrees", "same", "igual", "mismo", "misma", "intact", "intacto", "intacta",
        "trust", "confío", "confio", "confiar", "fiar"],
    "_NEGATION_GAP": ["está", "esta", "estaba", "están", "estan", "viene", "vienen", "venía", "venia", "ha", "han",
        "has", "have", "had", "been", "be", "is", "was", "are", "fue", "se", "le", "lo", "la", "it", "es", "el",
        "the", "a", "un", "una", "look", "looks", "seem", "seems", "ve", "veo", "even", "too", "tan", "so", "muy"],
    "_HAZMAT": ["peligroso", "peligrosa", "peligrosos", "peligrosas", "hazmat", "hazardous"],
    "_CARGO_GAP": ["traigo", "trae", "hay", "tengo", "tiene", "llevo", "lleva", "material", "materiales", "carga",
        "cargo", "producto", "productos", "any", "is", "there", "have", "has", "we", "i", "it", "de", "del", "el",
        "la", "los", "las", "the", "a", "an", "es", "son"],
    "_SEAL_BEFORE_SKIP": ["el", "la", "los", "las", "lo", "un", "una", "the", "a", "an", "any", "ningún", "ningun",
        "ninguno", "ninguna", "su", "sus", "mi", "this", "that", "traigo", "trae", "traemos", "tengo", "tiene",
        "tienen", "tenemos", "hay", "había", "habia", "viene", "vienen", "venía", "venia", "llevo", "lleva",
        "llevamos", "has", "have", "had", "got", "is", "está", "esta", "es", "was", "there", "veo", "vi", "se", "le"],
    "_EN_SPELLED": ["bee", "cee", "dee", "gee", "jay", "kay", "pee", "cue", "tee", "vee", "zee", "ex", "ef", "ar",
        "why", "see"],
    "_SPELLED": ["be", "ce", "efe", "ge", "hache", "jota", "ka", "ele", "eme", "ene", "eñe", "pe", "cu", "erre",
        "uve", "equis", "ye", "zeta"],
    "_NOT_SEAL": ["OK", "NO", "SI", "YA"],
    "_SENTENCE_WORDS": ["Y", "O", "E", "U"],
    "_SIZE_UNITS": ["foot", "feet", "ft", "footer", "pies", "pie", "inch", "inches", "pulgadas", "metros", "meters",
        "metres"],
    "_UNREAD_EXCEPTIONS": ["el resto está bien", "el resto esta bien", "the rest is fine", "get some rest",
        "rest area"],
    "_WORD_LETTERS": ["a", "de", "te", "ese", "em", "el"],
    "_LETTER_NAME_WORDS": ["el", "es", "ese", "te", "a", "see", "em", "why"],
    "_NOT_SEAL_LETTERS": ["es", "el", "la", "los", "las", "de", "del", "y", "o", "en", "con", "sin", "que", "se",
        "su", "mi", "tu", "al", "lo", "a", "e", "u", "ni", "le", "les", "me", "te", "nos", "mis", "tus", "sus", "por",
        "pa", "pal", "ya", "si", "son", "era", "fue", "ha", "he", "han", "hay", "va", "van", "voy", "da", "dan",
        "doy", "ver", "vez", "muy", "tan", "mas", "ese", "esa", "eso", "ok", "oh", "an", "and", "are", "as", "at",
        "by", "do", "for", "go", "has", "i", "if", "in", "is", "it", "its", "of", "on", "or", "per", "so", "the",
        "to", "up", "us", "was", "we", "yes"],
    "_RADIO": ["alpha", "alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
        "juliett", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
        "uniform", "victor", "whiskey", "whisky", "xray", "yankee", "zulu"],
    "_AS_IN": ["as in", "for", "like", "de", "como"],
    "_LETTER_PAIRS": ["be grande", "be larga", "be alta", "ve chica", "ve corta", "ve baja", "doble ve", "doble uve",
        "uve doble", "doble u", "i griega", "double u", "double you"],
    "_ENDS_SEAL": ["el", "la", "lo", "los", "las", "the", "pero", "but", "que", "traigo", "llevo", "tengo", "cargo",
        "cargué", "cargue", "entrego", "hay", "viene", "trae", "tiene", "lleva", "has", "does", "did", "seems",
        "looks", "i", "we", "it", "ok", "okay", "gracias", "thanks", "thank", "please", "listo", "repito", "peso",
        "weight", "sin", "without", "del", "pesa", "pesan", "weighs", "weigh", "son", "ahí", "ahi", "buenas",
        "buenos", "buen", "muchas", "nada", "saludos", "saludo", "delivering", "bringing", "carrying", "hauling",
        "unloading", "loading"],
    "_LOW_PUNCT": ["stop", "comma", "dash"],
    "_CLITICS": ["s", "t", "re", "ll", "ve", "d", "m"],
    "_ARTICLE_WORDS": ["un", "una", "a", "an", "one"],
    "_INDEFINITE": ["un", "una", "a", "an"],
    "_EN_INDEFINITE": ["a", "an"],
    "_ES_INDEFINITE": ["un", "una"],
    "_THOUSAND": ["mil", "thousand"],
    "_REPEATS": ["doble", "double", "triple"],
    "_AND_A_HALF": ["y media", "and a half"],
    "_TRAILER_ARTICLES": ["la", "el", "the"],
    "_DE_ARTICLES": ["la", "el"],
    "_AND": ["y", "and"],
    "_KEYWORD_TAILS": ["de seguridad"],
}


def witness(text: str, parse=voice_from_transcript) -> tuple:
    w = parse({"transcript": text}, model="prueba", provider="none", source_sha256="")
    return w.count, w.seal, w.detail


CASES = [(n, w) for n, words in WORDS.items() for w in words]


def test_the_table_is_the_parsers_lists():
    """WORDS is every word of every list, no more, no less: delete a word from a list in tinytms/adapters.py, or add
    one, and this fails."""
    assert sorted(WORDS) == sorted(LISTS)
    for name in LISTS:
        assert sorted(dict.fromkeys(voice_mutants.own_words(name))) == sorted(WORDS[name]), name
    assert len(LISTS) >= 100 and len(CASES) >= 1800


def test_every_word_of_every_list_has_a_case():
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
