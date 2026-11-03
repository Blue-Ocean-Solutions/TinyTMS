"""Seeded generator of SYNTHETIC driver voice notes, and how the voice parser is measured on them (ORDER Nube-9).

Each note is written from parts (a count, a seal, a weight, hazmat, fillers, other ids) with what the driver meant
known in advance: the count and the seal, or None when no one value may be taken from the note (a correction, two
values, a hedge, a negation, a broken seal, an id said after other words). Every note comes in two forms: punctuated,
and lowercase with no punctuation, as the local ASR writes. The parser's tables are never used here; numbers are
spelled by tests/test_voice_variantes.py.

Each field (count, seal) of each note falls in one class:
  right        the meant value (or nothing, when nothing was said)
  none_reason  None with a non-empty reason (the judge HOLDs; never wrong)
  wrong        a value that is not the meant one: a misread, or a value where None was meant
  picked       one of two different values said for the field
  none_empty   None with an empty reason although the field was said
A note's class is its worst field's. "good" notes have one clear count and one clear seal said with lead-ins only.

  python -m tests.voice_notes [ROOT ...]     measure the checkouts at ROOT (default: this one)
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tests.test_voice_variantes import EN_0_19, ES_0_29, en_words, es_words

SEED = 20261103
CLASSES = ("right", "none_reason", "wrong", "picked", "none_empty")
WORST = {c: i for i, c in enumerate(("right", "none_reason", "none_empty", "picked", "wrong"))}


@dataclass(frozen=True)
class Note:
    kind: str
    text: str
    count: int | None  # meant; None: no one count may be taken
    seal: str | None  # meant ("MX-48213"); None: no seal may be taken (or none was said)
    seal_said: bool  # a seal keyword or an id was said
    counts_said: tuple[int, ...]
    seals_said: tuple[str, ...]
    good: bool
    low: bool  # the lowercase, unpunctuated form
    spoken: str = ""  # the meant seal as said, when its digits are said as words
    count_number: str = ""  # the meant count's number as said


def asr(text: str) -> str:
    """A sentence as the local ASR writes it: lowercase, no punctuation."""
    return " ".join(re.sub(r"[.,;:!?…]", " ", text.lower()).split())


def norm(seal: str | None) -> str | None:
    return re.sub(r"[\W_]", "", seal).upper() if seal else None


# --- parts -----------------------------------------------------------------------------------------------------------

ES_UNITS = [("tarimas", "tarima", "una"), ("pallets", "pallet", "un"), ("cajas", "caja", "una"),
            ("bultos", "bulto", "un"), ("piezas", "pieza", "una"), ("palets", "palet", "un")]
EN_UNITS = [("pallets", "pallet"), ("boxes", "box"), ("skids", "skid"), ("pieces", "piece")]
ES_LETTER = {"M": "eme", "X": "equis", "T": "te", "A": "a", "B": "be", "S": "ese", "C": "ce", "K": "ka", "F": "efe",
             "D": "de"}


def count_phrase(rng: random.Random, n: int, lang: str) -> str:
    if lang == "es":
        plural, singular, one = rng.choice(ES_UNITS)
        if n == 1:
            return f"{one} {singular}"
        said = str(n) if rng.random() < 0.15 else es_words(n, apocope=True)
        return f"{said} {plural}"
    plural, singular = rng.choice(EN_UNITS)
    if n == 1:
        return f"one {singular}"
    said = str(n) if rng.random() < 0.15 else en_words(n, rng)
    return f"{said} {plural}"


def number_of(phrase: str) -> str:
    """The number words of a count phrase ('cuarenta y ocho tarimas' -> 'cuarenta y ocho')."""
    return phrase.rsplit(" ", 1)[0]


def bare_number(rng: random.Random, n: int, lang: str) -> str:
    return es_words(n) if lang == "es" else en_words(n, rng)


def new_count(rng: random.Random) -> int:
    return rng.choice([rng.randint(1, 30), rng.randint(1, 30), rng.randint(31, 120), rng.randint(121, 999)])


def new_seal(rng: random.Random) -> tuple[str, str, str]:
    """(prefix, digits, suffix) of a seal id."""
    prefix = rng.choice(["MX", "MX", "MX", "TX", "AB", "SMX", "CA", ""])
    digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(3, 8)))
    if rng.random() < 0.7 and digits[0] == "0":
        digits = rng.choice("123456789") + digits[1:]
    suffix = rng.choice("BCDFKX") if rng.random() < 0.1 else ""
    return prefix, digits, suffix


def canonical(seal: tuple[str, str, str]) -> str:
    p, d, s = seal
    return "-".join(x for x in (p, d, s) if x)


def say_seal(rng: random.Random, seal: tuple[str, str, str], lang: str) -> tuple[str, bool]:
    """(the seal as said, whether its digits are said as number words)."""
    p, d, s = seal
    style = rng.choice(["written", "written", "spoken", "spoken", "spoken"])
    if style == "written":
        if not p:
            return d + (f"-{s}" if s else ""), False
        return rng.choice([f"{p}-{d}", f"{p}{d}", f"{p} {d}"]) + (f"-{s}" if s else ""), False
    if lang == "es":
        letters = rng.choice([p, " ".join(p), " ".join(ES_LETTER[c] for c in p)]) if p else ""
        dash = rng.choice(["guión ", "guion ", "raya ", ""]) if p else ""
        if d[0] != "0" and len(d) <= 6 and rng.random() < 0.25:
            digits = es_words(int(d))
        else:
            digits = " ".join(ES_0_29[int(c)] for c in d)
        tail = f" guión {ES_LETTER[s]}" if s else ""
    else:
        letters = rng.choice([p, " ".join(p)]) if p else ""
        dash = rng.choice(["dash ", ""]) if p else ""
        if d[0] != "0" and len(d) <= 6 and rng.random() < 0.15:
            digits = en_words(int(d), rng)
        else:
            digits = " ".join("oh" if c == "0" and k and rng.random() < 0.3 else EN_0_19[int(c)]
                              for k, c in enumerate(d))
        tail = f" dash {rng.choice([s, s.lower()])}" if s else ""
    return (f"{letters} {dash}{digits}{tail}" if letters else f"{dash}{digits}{tail}").strip(), True


GOOD_COUNT = {"es": ["Traigo {c}", "Son {c}", "Llevo {c}", "{c}", "Vienen {c}", "Tengo {c}", "Cargué {c}",
                     "Traigo {c} de madera", "Entrego {c}"],
              "en": ["I have {c}", "{c}", "Got {c}", "We're carrying {c}", "Bringing {c}", "I've got {c}",
                     "Delivering {c}"]}
GOOD_SEAL = {"es": ["El sello es {s}", "Sello {s}", "Sello número {s}", "El número de sello es {s}", "Sello: {s}",
                    "Con sello {s}", "Sello No. {s}", "El sello de seguridad es {s}", "Traigo el sello {s}",
                    "El precinto es {s}", "Marchamo {s}", "El sello es el {s}"],
             "en": ["The seal is {s}", "Seal {s}", "Seal number {s}", "Seal is {s}", "Seal No. {s}", "With seal {s}",
                    "The seal number is {s}"]}
WEIGHT = {"es": ["Pesa {w} kilos", "Peso {w} kilos", "{w} kilos", "El peso es {w} kilos", "Peso bruto {w} kg"],
          "en": ["Weighs {w} kilos", "{w} kilos", "Weight {w} kg", "Gross weight is {w} kilos"]}
HAZMAT = {"es": ["Sin material peligroso", "No traigo material peligroso", "No hay material peligroso"],
          "en": ["No hazmat", "Nothing hazardous", "No hazardous material"]}
FILLERS = {"es": ["Buenas tardes", "Buenos días", "Gracias", "Muchas gracias", "Bueno, un saludo",
                  "Todo bien, no falta nada", "El sello coincide", "Saludos", "Listo", "Ahí le va", "Llegué a las tres",
                  "Eh, pues, ya llegué"],
           "en": ["Thanks", "Hello, this is the driver", "It matches the paperwork", "The seal is not broken",
                  "Okay, that's it", "Thank you", "All good"]}
OTHER_IDS = {"es": ["Andén cuatro", "La factura es 77812", "Placas ABC-123", "Unidad 15", "Ruta nueve"],
             "en": ["Trailer number 53", "Dock seven", "Truck 4471", "Door 12"]}

COUNT_TWO = {"es": ["Traigo {a}, no, {b}", "Traigo {a}... no, {b}", "Son {an}, perdón, {b}", "Traigo {a}, digo, {bn}",
                    "Traigo {a}, no, espérame, {bn} en total", "Traigo {a}, me equivoqué, son {bn}",
                    "{a}, mejor dicho {bn}", "Son {an} o {b}", "Traigo {a}, eh, {bn} en total",
                    "Traigo {a}. Bueno, {b}"],
             "en": ["{a}, sorry, {bn}", "{a}, no wait, {bn}", "{a}, I mean {bn}", "{a}, nope, uh, {bn} total",
                    "{a}, make that {bn}", "{an}, sorry, {b}", "{an} or {b}", "{a}, uh, {bn} altogether",
                    "{a}. Actually, {b}"]}
COUNT_NONE = {"es": ["Traigo unas {c}", "Como {c}", "{c} más o menos", "Casi {c}", "Creo que son {c}", "{c}, creo",
                     "Tal vez {c}", "No traigo {c}", "No son {c}"],
              "en": ["About {c}", "Around {c}", "{c} or so", "I think {c}", "{c}, I think", "Maybe {c}",
                     "Probably {c}", "It's not {c}"]}
SEAL_TWO = {"es": ["El sello es {s}, no, perdón, {t}", "El sello es {s}. Perdón, {t}", "Sello {s}, digo, {t}",
                   "El sello es {s}, perdón, es el {t}", "Los sellos son {s} y {t}",
                   "El sello es {s}. El sello es {t}"],
            "en": ["The seal is {s}. Sorry, {t}", "Seal {s}, I mean {t}", "Seal {s}, no wait, {t}",
                   "The seal is {s}, sorry, it's {t}", "Seals {s} and {t}"]}
SEAL_NONE = {"es": ["El sello creo que es {s}", "Creo que el sello es {s}", "Sello, creo, {s}", "El sello es {s}, creo",
                    "El sello no es {s}", "No traigo el sello {s}", "El sello {s} viene roto",
                    "El sello {s} no coincide",
                    "Falta el sello {s}", "El sello, según la carta, es {s}", "El sello de la carta es {s}",
                    "El sello de ayer era {s}", "El sello que me dieron es {s}"],
             "en": ["I think the seal is {s}", "The seal is {s}, I think", "Maybe the seal is {s}",
                    "The seal is not {s}",
                    "The seal isn't {s}", "Seal {s} is broken", "The seal {s} doesn't match", "Seal {s} is missing",
                    "The seal on the paperwork is {s}", "The seal on the bill says {s}"]}
SEAL_ELSEWHERE = {"es": ["Con sello, la factura es {n}", "El sello lo vi en la {small}", "Y el sello",
                         "El sello lo traigo bien"],
                  "en": ["Seal is fine, trailer number is {n}", "The seal on trailer number {small} is intact",
                         "And the seal", "Seal looks fine"]}
SEAL_BENIGN = {"es": ["El sello era {s}", "El sello dice {s}", "El sello de la caja es {s}"],
               "en": ["The seal was {s}", "The seal reads {s}", "The seal on the trailer is {s}"]}


def _sentences(rng: random.Random, parts: list[str], keep_first: int = 0) -> str:
    """Join clauses into a note: mostly sentences, sometimes a comma."""
    head, rest = parts[:keep_first], parts[keep_first:]
    rng.shuffle(rest)
    parts = head + rest
    out = parts[0][0].upper() + parts[0][1:]
    for p in parts[1:]:
        if rng.random() < 0.25:
            out += ", " + p[0].lower() + p[1:] if not p[:2].isupper() else ", " + p
        else:
            out += ". " + p[0].upper() + p[1:]
    return out + "."


def _extras(rng: random.Random, lang: str) -> list[str]:
    out = []
    if rng.random() < 0.6:
        out.append(rng.choice(WEIGHT[lang]).format(w=_weight(rng, lang)))
    if rng.random() < 0.5:
        out.append(rng.choice(HAZMAT[lang]))
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        out.append(rng.choice(FILLERS[lang]))
    if rng.random() < 0.3:
        out.append(rng.choice(OTHER_IDS[lang]))
    return out


def _weight(rng: random.Random, lang: str) -> str:
    w = rng.choice([4800, 1800, 950, 12000, 22000, 3500, 7250])
    r = rng.random()
    if r < 0.3:
        return str(w)
    if r < 0.4:
        return f"{w:,}"
    return es_words(w) if lang == "es" else en_words(w, rng)


NUMBER_START = set(ES_0_29) | set(EN_0_19) | {"un", "una", "veintiún", "mil", "cien", "ciento", "thousand", "hundred",
                                              "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta",
                                              "noventa", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                                              "eighty", "ninety", "doscientos", "trescientos", "cuatrocientos",
                                              "quinientos", "seiscientos", "setecientos", "ochocientos", "novecientos"}


def _low_seal_reads(text: str, said: str) -> bool:
    """In the lowercase, unpunctuated form: does a seal whose digits are said as words still read one way? Not when a
    number follows them with no stop between ('... dos uno tres cuatro mil ochocientos kilos'): two readings. A seal
    that ends in a letter ('... dos uno tres guión be') still reads."""
    low, said = asr(text), asr(said)
    if said.split()[-1] not in NUMBER_START | {"oh"}:
        return True
    i = low.find(said)
    nxt = low[i + len(said):].split()[:1] if i >= 0 else []
    return not nxt or not (nxt[0] in NUMBER_START or nxt[0].replace(",", "").isdecimal())


def notes(seed: int = SEED, n_good: int = 1500, n_other: int = 1700) -> list[Note]:
    """The punctuated notes; forms() adds the lowercase ones."""
    rng = random.Random(seed)
    out: list[Note] = []

    def add(kind, text, count, seal, seal_said, counts, seals, good=False):
        out.append(Note(kind, text, count, seal, seal_said, tuple(counts), tuple(seals), good, False,
                        said if spoken and seal else "", number_of(cphrase)))

    for _ in range(n_good):
        lang = rng.choice(["es", "es", "en"])
        n, seal = new_count(rng), new_seal(rng)
        said, spoken = say_seal(rng, seal, lang)
        cphrase = count_phrase(rng, n, lang)
        parts = [rng.choice(GOOD_COUNT[lang]).format(c=cphrase),
                 rng.choice(GOOD_SEAL[lang]).format(s=said)] + _extras(rng, lang)
        add("good", _sentences(rng, parts), n, canonical(seal), True, [n], [canonical(seal)], good=True)

    kinds = ["count_two", "count_none", "seal_two", "seal_none", "seal_elsewhere", "seal_benign", "no_seal",
             "correction_after_seal"]
    for k in range(n_other):
        kind = kinds[k % len(kinds)]
        lang = rng.choice(["es", "es", "en"])
        n, seal = new_count(rng), new_seal(rng)
        said, spoken = say_seal(rng, seal, lang)
        cphrase = count_phrase(rng, n, lang)
        good_count = rng.choice(GOOD_COUNT[lang]).format(c=cphrase)
        good_seal = rng.choice(GOOD_SEAL[lang]).format(s=said)
        extras = _extras(rng, lang)
        if kind == "count_two":
            b = n + rng.choice([1, 2, 10, -1]) if n > 2 else n + 1
            t = rng.choice(COUNT_TWO[lang])
            text = _sentences(rng, [t.format(a=count_phrase(rng, n, lang), b=count_phrase(rng, b, lang),
                                             an=bare_number(rng, n, lang), bn=bare_number(rng, b, lang)),
                                    good_seal] + extras, keep_first=1)
            add(kind, text, None, canonical(seal), True, [n, b], [canonical(seal)])
        elif kind == "count_none":
            t = rng.choice(COUNT_NONE[lang])
            text = _sentences(rng, [t.format(c=count_phrase(rng, n, lang)), good_seal] + extras)
            add(kind, text, None, canonical(seal), True, [n], [canonical(seal)])
        elif kind == "seal_two":
            other = (seal[0], seal[1][:-2] + seal[1][-1] + seal[1][-2], seal[2])
            if other == seal:
                other = (seal[0], seal[1] + "7", seal[2])
            said2, _ = say_seal(rng, other, lang)
            t = rng.choice(SEAL_TWO[lang])
            text = _sentences(rng, [good_count, t.format(s=said, t=said2)] + extras, keep_first=0)
            add(kind, text, n, None, True, [n], [canonical(seal), canonical(other)])
        elif kind == "seal_none":
            t = rng.choice(SEAL_NONE[lang])
            text = _sentences(rng, [good_count, t.format(s=said)] + extras)
            add(kind, text, n, None, True, [n], [canonical(seal)])
        elif kind == "seal_elsewhere":
            t = rng.choice(SEAL_ELSEWHERE[lang])
            text = _sentences(rng, [good_count, t.format(n=rng.choice(["48213", "53", "7781"]),
                                                         small=rng.choice(["uno", "four", "dos", "seven"]))] + extras)
            add(kind, text, n, None, True, [n], [])
        elif kind == "seal_benign":
            t = rng.choice(SEAL_BENIGN[lang])
            text = _sentences(rng, [good_count, t.format(s=said)] + extras)
            add(kind, text, n, canonical(seal), True, [n], [canonical(seal)])
        elif kind == "no_seal":
            text = _sentences(rng, [good_count] + [e for e in extras if "sello" not in e.lower()
                                                    and "seal" not in e.lower()])
            add(kind, text, n, None, False, [n], [])
        else:  # a count correction said right after the seal: "Doce tarimas, sello MX-48213, no, trece."
            b = n + 1
            corr = rng.choice(["no", "perdón", "digo"] if lang == "es" else ["sorry", "no", "I mean"])
            text = _sentences(rng, [good_count, f"{good_seal}, {corr}, {bare_number(rng, b, lang)}"] + extras,
                              keep_first=2)
            add(kind, text, None, canonical(seal), True, [n, b], [canonical(seal)])
    return out


def forms(base: list[Note]) -> list[Note]:
    """Every note punctuated, then lowercase and unpunctuated (where a seal that ran into a number reads None)."""
    out = list(base)
    for nt in base:
        low = asr(nt.text)
        seal = None if nt.spoken and not _low_seal_reads(nt.text, nt.spoken) else nt.seal
        # Spoken seal digits right before a count said in more than one word ("... t x six hundred and eighty eight
        # boxes"): where the seal ends, and so the count, does not read one way either.
        count = nt.count
        if nt.spoken and asr(nt.spoken).split()[-1] in NUMBER_START | {"oh"} and len(
                re.split(r"[\s-]+", nt.count_number)) > 1 and (
                f"{asr(nt.spoken)} {asr(nt.count_number)}" in low):
            count = None
        out.append(Note(nt.kind, low, count, seal, nt.seal_said, nt.counts_said, nt.seals_said, nt.good, True,
                        nt.spoken, nt.count_number))
    return out


# --- classify --------------------------------------------------------------------------------------------------------


def classify_field(meant, said: tuple, got, why: str, said_any: bool, seal: bool) -> str:
    same = (lambda a, b: norm(a) == norm(b)) if seal else (lambda a, b: a == b)
    if got is None:
        if why:
            return "none_reason"
        return "none_empty" if said_any else "right"
    if meant is not None and same(got, meant):
        return "right"
    if len({norm(x) if seal else x for x in said}) > 1 and any(same(got, v) for v in said):
        return "picked"
    return "wrong"


def classify(nt: Note, parsed: tuple) -> tuple[str, str]:
    count, why, seal, seal_why = parsed
    c = classify_field(nt.count, nt.counts_said, count, why, True, False)
    s = classify_field(nt.seal, nt.seals_said, seal, seal_why, nt.seal_said, True)
    return c, s


def accepts(nt: Note, parsed: tuple) -> bool:
    from tinytms.verify import Limits, judge
    from tinytms.witness import DocWitness, VisionWitness, VoiceWitness

    count, _, seal, _ = parsed
    v = VisionWitness(status="OK", model="FIXTURE", provider="FIXTURE", count=nt.count)
    d = DocWitness(status="OK", model="FIXTURE", provider="FIXTURE", pieces=nt.count, weight_kg=4800.0, seal=nt.seal,
                   hazmat=False)
    w = VoiceWitness(status="OK", model="FIXTURE", provider="FIXTURE", count=count, seal=seal)
    return judge(v, d, w, Limits(max_weight_kg=24000)).decision == "ACCEPT"


# --- measure ---------------------------------------------------------------------------------------------------------

_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from tinytms.adapters import _parse_voice
print(json.dumps([list(_parse_voice(t)) for t in json.load(sys.stdin)], ensure_ascii=False))
"""


def parse_all(root: Path, texts: list[str]) -> list[tuple]:
    """_parse_voice of the checkout at root on every text, in a fresh interpreter."""
    p = subprocess.run([sys.executable, "-I", "-c", _CHILD, str(root)], input=json.dumps(texts), text=True,
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(f"{root}: {p.stderr}")
    return [tuple(x) for x in json.loads(p.stdout)]


def measure(all_notes: list[Note], parsed: list[tuple]) -> dict:
    out: dict = {}
    for low in (False, True):
        sel = [(nt, p) for nt, p in zip(all_notes, parsed) if nt.low == low]
        form = "lowercase" if low else "punctuated"
        notes_c, count_c, seal_c, good_c = Counter(), Counter(), Counter(), Counter()
        for nt, p in sel:
            c, s = classify(nt, p)
            count_c[c] += 1
            seal_c[s] += 1
            worst = max((c, s), key=WORST.__getitem__)
            notes_c[worst] += 1
            if nt.good:
                good_c[worst] += 1
        good = [(nt, p) for nt, p in sel if nt.good]
        out[form] = {"notes": len(sel), "classes": dict(notes_c), "count": dict(count_c), "seal": dict(seal_c),
                     "good_notes": len(good), "good_classes": dict(good_c),
                     "good_accept": sum(accepts(nt, p) for nt, p in good)}
    return out


def main(argv: list[str]) -> None:
    roots = [Path(a) for a in argv] or [Path(__file__).resolve().parent.parent]
    all_notes = forms(notes())
    print(f"seed {SEED}: {len(all_notes) // 2} notes, each punctuated and lowercase-unpunctuated "
          f"({sum(nt.good for nt in all_notes) // 2} good)")
    for root in roots:
        m = measure(all_notes, parse_all(root, [nt.text for nt in all_notes]))
        print(f"\n{root}")
        for form, r in m.items():
            print(f"  {form}: {r['notes']} notes")
            for what in ("classes", "count", "seal", "good_classes"):
                print(f"    {what:13} " + "  ".join(f"{c}={r[what].get(c, 0)}" for c in CLASSES))
            rate = 100 * r["good_accept"] / r["good_notes"]
            print(f"    good ACCEPT   {r['good_accept']}/{r['good_notes']} = {rate:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
