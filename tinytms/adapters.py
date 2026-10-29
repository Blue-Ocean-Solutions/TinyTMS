"""Map raw model outputs (or hand-written FIXTURE files) to witnesses.

Unknown or malformed shapes become status=ERROR with a detail; nothing is guessed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .witness import DocWitness, VisionWitness, VoiceWitness, Witness, sha256_file

FIXTURE_PROVENANCE_PREFIX = "FIXTURE"


# --- generic response helpers -------------------------------------------------------------


def output_text(data: dict[str, Any]) -> str | None:
    """Text from a Responses payload: output_text, or output[type=message].content[type=output_text].text.

    Measured (Spark 1.3): output = [{type: reasoning}, {type: message, content: [{type: output_text, text}]}].
    """
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type", "message") != "message":
            continue
        for c in item.get("content") or []:
            if isinstance(c, dict) and c.get("type", "output_text") == "output_text" and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "".join(parts) if parts else None


def parse_json_text(text: str) -> Any:
    """json.loads tolerant of a ```json fence around the object."""
    t = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, flags=re.S)
    return json.loads(m.group(1) if m else t)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# --- SAM 3.1 SSE ---------------------------------------------------------------------------


def parse_sse(sse_text: str) -> list[dict[str, Any]]:
    """Events of a text/event-stream body: every 'data: {...}' line that is a JSON object."""
    events = []
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except ValueError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


def sam_text(events: list[dict[str, Any]]) -> str | None:
    """Full SAM text: response.output_text.done 'text', else the text inside response.completed."""
    for ev in events:
        if ev.get("type") == "response.output_text.done" and isinstance(ev.get("text"), str):
            return ev["text"]
    for ev in events:
        if ev.get("type") == "response.completed" and isinstance(ev.get("response"), dict):
            return output_text(ev["response"]) or ""
    return None


_MASK_DATA = re.compile(r"(<\|mask;[^|]*?)data=.*?(?=\|>)", flags=re.S)
_BOX = re.compile(r"<\|box;([^|]*)\|>")


def strip_masks(text: str) -> str:
    """Drop the heavy 'data=' payload of every <|mask;...|> record (never goes to receipts or tests)."""
    return _MASK_DATA.sub(lambda m: f"{m.group(1)}data=<omitted {len(m.group(0)) - len(m.group(1)) - 5} chars>", text)


def strip_sse(sse_text: str) -> str:
    """SSE body safe to store: mask data dropped from every string, text deltas replaced by their length."""

    def walk(v: Any, key: str = "") -> Any:
        if isinstance(v, dict):
            return {k: walk(x, k) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            return f"<delta omitted {len(v)} chars>" if key == "delta" else strip_masks(v)
        return v

    out = []
    for line in sse_text.splitlines():
        if line.startswith("data:"):
            try:
                line = "data: " + json.dumps(walk(json.loads(line[5:].strip())), ensure_ascii=False)
            except ValueError:
                line = strip_masks(line)
        out.append(line)
    return "\n".join(out)


def sam_boxes(text: str) -> list[dict[str, float]]:
    """Every <|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|> record as a dict of floats."""
    out = []
    for m in _BOX.finditer(text):
        kv: dict[str, float] = {}
        for part in m.group(1).split(";"):
            k, _, v = part.partition("=")
            try:
                kv[k.strip()] = float(v)
            except ValueError:
                continue
        out.append(kv)
    return out


def vision_from_sam(
    sse_text: str, *, model: str, provider: str, source_sha256: str, concept: str = "pallet"
) -> VisionWitness:
    """SAM 3.1 SSE body -> witness. count = number of <|box; records; zero records is count 0, status OK."""
    meta = dict(model=model, provider=provider, source_sha256=source_sha256, concept=concept)
    if not isinstance(sse_text, str):
        return VisionWitness(status="ERROR", detail=f"SAM body must be SSE text, got {type(sse_text).__name__}",
                             **meta)
    events = parse_sse(sse_text)
    for ev in events:
        if ev.get("type") in ("error", "response.failed", "response.incomplete"):
            err = ev.get("error") or (ev.get("response") or {}).get("error") or ev.get("type")
            return VisionWitness(status="ERROR", detail=f"SAM stream {ev.get('type')}: {str(err)[:300]}", **meta)
    text = sam_text(events)
    if text is None:
        kinds = sorted({str(e.get("type")) for e in events})
        return VisionWitness(status="ERROR", detail=f"SAM stream has no output_text.done/completed; events={kinds}",
                             **meta)
    recs = sam_boxes(text)
    boxes, need = [], ("x1", "y1", "x2", "y2")
    for r in recs:
        if all(k in r for k in need):
            boxes.append(tuple(r[k] for k in need) + ((r["c"],) if "c" in r else ()))
    w = int(recs[0]["w"]) if recs and "w" in recs[0] else None
    h = int(recs[0]["h"]) if recs and "h" in recs[0] else None
    return VisionWitness(status="OK", count=len(recs), boxes=tuple(boxes), image_w=w, image_h=h, **meta)


def vision_from_fields(
    data: dict[str, Any], *, model: str, provider: str, source_sha256: str, concept: str = "pallet"
) -> VisionWitness:
    """Hand-written witness fields ({"count": n, "boxes": [[x1,y1,x2,y2(,c)],...]}), used by FIXTURE files."""
    meta = dict(model=model, provider=provider, source_sha256=source_sha256, concept=concept)
    if not _is_int(data.get("count")):
        return VisionWitness(status="ERROR", detail=f"no integer 'count'; keys={sorted(data)[:12]}", **meta)
    boxes = tuple(
        tuple(float(x) for x in b) for b in data.get("boxes") or []
        if isinstance(b, (list, tuple)) and len(b) in (4, 5) and all(isinstance(x, (int, float)) for x in b)
    )
    return VisionWitness(status="OK", count=data["count"], boxes=boxes, **meta)


# --- document ------------------------------------------------------------------------------


def doc_from_fields(
    f: dict[str, Any], *, model: str, provider: str, source_sha256: str
) -> DocWitness:
    meta = dict(model=model, provider=provider, source_sha256=source_sha256)
    errs = []
    pieces, weight, seal, hazmat = f.get("pieces"), f.get("weight_kg"), f.get("seal"), f.get("hazmat")
    if pieces is not None and not _is_int(pieces):
        errs.append(f"pieces not int: {pieces!r}")
    if weight is not None and (isinstance(weight, bool) or not isinstance(weight, (int, float))):
        errs.append(f"weight_kg not number: {weight!r}")
    if seal is not None and not isinstance(seal, str):
        errs.append(f"seal not string: {seal!r}")
    if hazmat is not None and not isinstance(hazmat, bool):
        errs.append(f"hazmat not bool/null: {hazmat!r}")
    if errs:
        return DocWitness(status="ERROR", detail="; ".join(errs), **meta)
    return DocWitness(
        status="OK",
        pieces=pieces,
        weight_kg=float(weight) if weight is not None else None,
        seal=seal,
        shipper=f.get("shipper") if isinstance(f.get("shipper"), str) else None,
        consignee=f.get("consignee") if isinstance(f.get("consignee"), str) else None,
        hazmat=hazmat,
        **meta,
    )


def doc_from_spark(
    data: dict[str, Any], *, model: str, provider: str, source_sha256: str
) -> DocWitness:
    meta = dict(model=model, provider=provider, source_sha256=source_sha256)
    if data.get("status") == "incomplete":
        why = (data.get("incomplete_details") or {}).get("reason", "?")
        return DocWitness(status="ERROR", detail=f"response incomplete ({why}); usage={data.get('usage')}", **meta)
    text = output_text(data)
    if text is None:
        return DocWitness(status="ERROR", detail=f"no output text; keys={sorted(data)[:12]}", **meta)
    try:
        fields = parse_json_text(text)
    except ValueError as e:
        return DocWitness(status="ERROR", detail=f"output is not JSON: {e}; text={text[:200]!r}", **meta)
    if not isinstance(fields, dict):
        return DocWitness(status="ERROR", detail="output JSON is not an object", **meta)
    return doc_from_fields(fields, **meta)


# --- voice ---------------------------------------------------------------------------------
#
# The transcript is cut into tokens (words, digit groups, punctuation). Seal mentions are read first ("sello",
# "seal", ...) and their tokens are set aside; then every number phrase is classified: a count (number + unit noun),
# a weight (+ kilos), an id ("tráiler cuatro"), or a bare number. A count comes out only if exactly one value is
# said for it. Self-corrections ("doce… no, trece"), alternatives ("doce o trece"), approximations ("unas doce"),
# negations ("no traigo doce") and unclear number words ("veinte cuatro") give None and a detail that says why.

# "payets", "pajets", "payettes": how Muse Voice Transcribe wrote "pallets" said in Spanish, three runs over the
# same audio (real transcripts, tests/data/real-voice-es-*).
_UNITS = (
    r"pallets?|palets?|pal[eé]s|payets?|pajets?|payettes?|paletas?|tarimas?|skids?|cajas?|boxes|box|bultos?|"
    r"piezas?|pieces?"
)
_UNIT = re.compile(_UNITS)
_WEIGHT_UNITS = frozenset(
    "kilos kilo kg kgs kilogramos kilogramo toneladas tonelada ton tons tonnes libras libra lb lbs pounds".split()
)
# A spoken dash inside a seal id ("MX guión cuatro ocho...", same real transcript); "guion" is the RAE 2010 spelling.
_SEAL_DASH = frozenset({"guión", "guion", "raya", "dash", "hyphen"})

_ES_ONES = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
            "ocho": 8, "nueve": 9}
_ES_TEENS = {
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veintiuno": 21, "veintiún": 21, "veintiun": 21,
    "veintiuna": 21, "veintidós": 22, "veintidos": 22, "veintitrés": 23, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiséis": 26, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}
# tens that take "y" + a unit: "treinta y dos", and the long forms "veinte y cuatro", "diez y seis"
_ES_TENS = {"diez": 10, "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
            "ochenta": 80, "noventa": 90}
_ES_HUNDREDS = {"ciento": 100, **{w + s: v for w, v in (
    ("doscient", 200), ("trescient", 300), ("cuatrocient", 400), ("quinient", 500), ("seiscient", 600),
    ("setecient", 700), ("ochocient", 800), ("novecient", 900)) for s in ("os", "as")}}
_EN_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_EN_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
             "seventeen": 17, "eighteen": 18, "nineteen": 19}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
            "ninety": 90}
_NUMBER_WORDS = frozenset({*_ES_ONES, *_ES_TEENS, *_ES_TENS, *_ES_HUNDREDS, *_EN_ONES, *_EN_TEENS, *_EN_TENS,
                           "cero", "zero", "cien", "mil", "hundred", "thousand"})
_DOZEN = frozenset({"docena", "dozen"})
_DOZENS = frozenset({"docenas", "dozens"})

# Words around a number that change what it means.
_ID_NOUNS = frozenset(
    "tráiler trailer camión camion unidad andén anden puerta rampa bahía bahia muelle caja ruta viaje factura pedido "
    "orden folio número numero placa placas turno línea linea carril truck dock door bay unit route trip invoice order "
    "number plate lane gate".split()
)
_CORRECTIONS = tuple(tuple(p.split()) for p in (
    "no", "perdón", "perdon", "digo", "corrijo", "mejor dicho", "más bien", "mas bien", "o sea", "bueno",
    "me equivoqué", "me equivoque", "quise decir", "sorry", "wait", "i mean", "actually", "or rather", "rather",
    "make that", "correction", "scratch that",
))
_APPROX_BEFORE = tuple(tuple(p.split()) for p in (
    "como", "unas", "unos", "casi", "aproximadamente", "aprox", "alrededor de", "cerca de", "más de", "mas de",
    "menos de", "más o menos", "mas o menos", "al menos", "por lo menos", "hasta", "about", "around", "roughly",
    "approximately", "approx", "nearly", "almost", "some", "like", "over", "under", "more than", "less than",
    "fewer than", "at least", "up to",
))
_APPROX_AFTER = tuple(tuple(p.split()) for p in (
    "más o menos", "mas o menos", "o así", "o asi", "aproximadamente", "o algo así", "y pico", "or so", "roughly",
    "approximately", "give or take", "more or less",
))
_NEGATIONS = frozenset({"no", "ni", "nunca", "tampoco", "not", "never", "nor"})
_NEGATED_CONTRACTIONS = frozenset({"don", "doesn", "didn", "isn", "aren", "wasn", "weren", "haven", "hasn", "won",
                                   "can", "couldn"})
_COUNT_VERBS = frozenset(
    "traigo trae traemos llevo lleva llevamos tengo tenemos son eran serían serian vienen van cargo cargué cargue hay "
    "entrego bring have got carrying carry are".split()
)
_ALTERNATIVES = frozenset({"o", "u", "ó", "or", "y", "e", "and", "a", "to", "ni", "nor"})
_AFTER_CORRECTION = frozenset(  # "no, son trece", "sorry, it's thirteen", "make that thirteen"
    "son eran es serían serian fueron traigo tengo llevo are is it s that i have got we were was".split()
)
_SPEECH_FILLERS = frozenset({"eh", "ehh", "em", "emm", "mm", "mmm", "um", "uh", "uhm", "este"})
_NOT_FILLER = frozenset(
    "con sin en por para al hasta desde entre sobre cada with without in on for to from at by per each "
    "el la los las lo the y e o u ni and or pero but si if que".split()
) | _NEGATIONS | _COUNT_VERBS | _ID_NOUNS | _WEIGHT_UNITS | _NUMBER_WORDS | {w for p in _CORRECTIONS for w in p}

# Seal mentions. Spelled letter names are only the ones that are not also everyday words ("de", "te", "ese", "ve").
_SEAL_KEYWORDS = frozenset({"sello", "seal", "precinto", "marchamo"})
_SEAL_LEAD_IN = frozenset(
    "number no num nro n nº número numero es is was era dice says reads de del seguridad security el la the ese".split()
)
_SPELLED = {"be": "B", "ce": "C", "efe": "F", "ge": "G", "hache": "H", "jota": "J", "ka": "K", "ele": "L",
            "eme": "M", "ene": "N", "eñe": "Ñ", "pe": "P", "cu": "Q", "erre": "R", "uve": "V", "equis": "X",
            "ye": "Y", "zeta": "Z"}
_NOT_SEAL = frozenset({"OK", "Ok", "NO", "SÍ", "SI", "YA"})
_DIGIT_OF = {w: str(v) for w, v in (*_ES_ONES.items(), *_EN_ONES.items(), ("cero", 0), ("zero", 0))
             if w not in ("un", "una")}

_TOKEN = re.compile(
    r"(?P<num>\d{1,3}(?:[.,]\d{3})+(?![\w-])|\d+[.,]\d+(?![\w-]))"
    r"|(?P<word>[^\W_]+(?:[-‐][^\W_]+)*)"
    r"|(?P<stop>[.;!?…]+)"
    r"|(?P<comma>[,:])"
    r"|(?P<dash>[-‐–—])"
)
_THOUSANDS = re.compile(r"\d{1,3}(?:[.,]\d{3})+")
_GROUPED = re.compile(r"\d+(?:[.,]\d+)+")
_LOW_PUNCT = {"stop": ".", "comma": ",", "dash": "-"}


def _tokens(text: str) -> list[tuple[str, str]]:
    """(kind, text) per token. 'twenty-four' splits into words; 'MX-48213' stays one token."""
    out: list[tuple[str, str]] = []
    for m in _TOKEN.finditer(text):
        kind, t = m.lastgroup or "", m.group()
        parts = [p for p in re.split(r"[-‐]", t) if p]
        if kind == "word" and len(parts) > 1 and all(p.lower() in _NUMBER_WORDS for p in parts):
            out += [("word", p) for p in parts]
        else:
            out.append((kind, t))
    return out


def _w(ws: list[str], i: int) -> str:
    return ws[i] if 0 <= i < len(ws) else ""


def _is_unit(w: str) -> bool:
    return bool(_UNIT.fullmatch(w))


def _is_filler(w: str) -> bool:
    """One word allowed between a number and its unit: 'doce grandes tarimas', 'una docena de cajas'."""
    return w.isalpha() and len(w) > 1 and w not in _NOT_FILLER and not _is_unit(w) and w not in _DOZEN | _DOZENS


def _phrase_at(ws: list[str], i: int, phrases: tuple[tuple[str, ...], ...]) -> int:
    """Length of the longest phrase starting at ws[i], 0 if none."""
    return max((len(p) for p in phrases if tuple(ws[i:i + len(p)]) == p), default=0)


def _phrase_ending_at(ws: list[str], e: int, phrases: tuple[tuple[str, ...], ...]) -> int:
    return max((len(p) for p in phrases if len(p) <= e and tuple(ws[e - len(p):e]) == p), default=0)


# number words -> int. Each returns (value, end) for the longest well-formed phrase at ws[i], or None.


def _es_tens_part(ws: list[str], j: int) -> tuple[int, int] | None:
    w = _w(ws, j)
    if w in _ES_TENS:
        if _w(ws, j + 1) == "y" and _w(ws, j + 2) in _ES_ONES:
            return _ES_TENS[w] + _ES_ONES[ws[j + 2]], j + 3
        return _ES_TENS[w], j + 1
    for table in (_ES_TEENS, _ES_ONES):
        if w in table:
            return table[w], j + 1
    return None


def _es_below_1000(ws: list[str], i: int) -> tuple[int, int] | None:
    w = _w(ws, i)
    if w == "cien":
        return 100, i + 1
    if w in _ES_HUNDREDS:
        t = _es_tens_part(ws, i + 1)
        return (_ES_HUNDREDS[w] + t[0], t[1]) if t else (_ES_HUNDREDS[w], i + 1)
    return _es_tens_part(ws, i)


def _es_cardinal(ws: list[str], i: int) -> tuple[int, int] | None:
    if _w(ws, i) == "cero":
        return 0, i + 1
    a = _es_below_1000(ws, i)
    j = a[1] if a else i
    if _w(ws, j) != "mil":
        return a
    b = _es_below_1000(ws, j + 1)
    v = (a[0] if a else 1) * 1000
    return (v + b[0], b[1]) if b else (v, j + 1)


def _en_tens_part(ws: list[str], j: int) -> tuple[int, int] | None:
    w = _w(ws, j)
    if w in _EN_TENS:
        if _w(ws, j + 1) in _EN_ONES:
            return _EN_TENS[w] + _EN_ONES[ws[j + 1]], j + 2
        return _EN_TENS[w], j + 1
    for table in (_EN_TEENS, _EN_ONES):
        if w in table:
            return table[w], j + 1
    return None


def _en_below_1000(ws: list[str], i: int) -> tuple[int, int] | None:
    w = _w(ws, i)
    m = _EN_ONES.get(w) or (1 if w in ("a", "an") else None)
    if m is not None and _w(ws, i + 1) == "hundred":
        j = i + 2
        t = _en_tens_part(ws, j + 1 if _w(ws, j) == "and" else j)
        return (m * 100 + t[0], t[1]) if t else (m * 100, j)
    return _en_tens_part(ws, i)


def _en_cardinal(ws: list[str], i: int) -> tuple[int, int] | None:
    if _w(ws, i) == "zero":
        return 0, i + 1
    if _w(ws, i) in ("a", "an") and _w(ws, i + 1) == "thousand":
        a: tuple[int, int] | None = (1, i + 1)
    else:
        a = _en_below_1000(ws, i)
    if a is None or _w(ws, a[1]) != "thousand":
        return a
    k = a[1] + 1
    b = _en_below_1000(ws, k + 1 if _w(ws, k) == "and" else k)
    return (a[0] * 1000 + b[0], b[1]) if b else (a[0] * 1000, k)


def _number_at(ws: list[str], i: int, dozens: bool = True) -> tuple[int | None, int] | None:
    """Number phrase at ws[i]: (value, end). value None: a number was said but no single whole one ('docenas')."""
    w = _w(ws, i)
    base: tuple[int | None, int] | None
    if w.isdecimal():
        base = (int(w), i + 1)
        if _w(ws, i + 1) in ("mil", "thousand"):
            base = (int(w) * 1000, i + 2)
    elif _GROUPED.fullmatch(w):  # a 'num' token: "4,800" is 4800, "4.5" is not a count
        base = (int(re.sub(r"\D", "", w)) if _THOUSANDS.fullmatch(w) else None, i + 1)
    else:
        es, en = _es_cardinal(ws, i), _en_cardinal(ws, i)
        base = max((x for x in (es, en) if x), key=lambda x: x[1], default=None)
    if not dozens:
        return base
    if base is None:
        if w in _DOZEN:
            base = (1, i)
        elif w in ("a", "an") and _w(ws, i + 1) == "dozen":
            base = (1, i + 1)
        elif w == "media" and _w(ws, i + 1) == "docena":
            return 6, i + 2
        elif w == "half" and _w(ws, i + 1) in ("a", "an") and _w(ws, i + 2) == "dozen":
            return 6, i + 3
        elif w in _DOZENS:
            return None, i + 1  # "docenas de cajas": how many?
        else:
            return None
    v, j = base
    if _w(ws, j) not in _DOZEN | _DOZENS or v is None:
        return base
    v, j = v * 12, j + 1
    if ws[j:j + 2] == ["y", "media"]:
        v, j = v + 6, j + 2
    elif ws[j:j + 3] == ["and", "a", "half"]:
        v, j = v + 6, j + 3
    return v, j


def _unclear_pair(last: str, first: str) -> bool:
    """Two number groups said back to back that do not read one way: 'cuarenta ocho' (48? 408?), 'cien tres'."""
    return (last in _ES_TENS and first in _ES_ONES) or (last == "cien" and (first in _ES_ONES or first in _ES_TEENS
                                                                             or first in _ES_TENS))


# --- seal --------------------------------------------------------------------------------------


def _seal_like(toks: list[tuple[str, str]], lows: list[str], i: int) -> bool:
    """Would the token at i continue a seal id (and not start a count or a weight)?"""
    kind, text = toks[i]
    if kind not in ("word", "num") or text in _NOT_SEAL or lows[i] in ("un", "una"):
        return False
    n = _number_at(lows, i, dozens=False)
    if n:
        after = _w(lows, n[1])
        return not (_is_unit(after) or after in _WEIGHT_UNITS)
    return lows[i] in _SPELLED or bool(
        re.fullmatch(r"[A-Za-z0-9-]+", text) and (any(c.isdigit() for c in text) or (text.isupper() and len(text) <= 3))
    )


def _seal_value(toks: list[tuple[str, str]], lows: list[str], i: int) -> tuple[str, int, str]:
    """Read one seal id from token i: (id, index of the token that ended it, why it is unreadable or '')."""
    out: list[str] = []
    while i < len(toks):
        kind, text = toks[i]
        low = lows[i]
        if kind == "stop":
            break
        if kind == "comma" or low in _SPEECH_FILLERS:
            i += 1
            continue
        if kind == "dash" or low in _SEAL_DASH:
            if not out:
                break
            if out[-1] != "-":
                out.append("-")
            i += 1
            continue
        nxt = _w(lows, i + 1)
        if low in ("doble", "double", "triple") and nxt in _DIGIT_OF:
            out.append(_DIGIT_OF[nxt] * (3 if low == "triple" else 2))
            i += 2
            continue
        if (low, nxt) in (("doble", "ve"), ("doble", "uve"), ("uve", "doble")):
            out.append("W")
            i += 2
            continue
        if (low, nxt) == ("i", "griega"):
            out.append("Y")
            i += 2
            continue
        if low == "oh" and out and out[-1][-1:].isdigit():  # "four oh two"
            out.append("0")
            i += 1
            continue
        if low not in ("un", "una") and _number_at(lows, i, dozens=False):
            groups: list[tuple[str, int, int]] = []  # (digits, start, end)
            j = i
            while (n := _number_at(lows, j, dozens=False)) and n[0] is not None:
                said = lows[j] if n[1] == j + 1 and lows[j].isdecimal() else str(n[0])  # "007" stays 007
                groups.append((said, j, n[1]))
                j = n[1]
            if not groups:  # "4.5": a number, but not digits of an id
                break
            after = _w(lows, j)
            if _is_unit(after) or after in _WEIGHT_UNITS or (_is_filler(after) and _is_unit(_w(lows, j + 1))):
                if len(groups) == 1:
                    break  # "..., doce tarimas": the number after the seal is the count
                return "", i, f"seal digits run into a count: {' '.join(lows[i:j + 1])!r}"
            for (_, _, e), (_, s, _) in zip(groups, groups[1:]):
                if _unclear_pair(lows[e - 1], lows[s]):
                    return "", j, f"unclear seal digits: {lows[e - 1] + ' ' + lows[s]!r}"
            out += [g for g, _, _ in groups]
            i = j
            continue
        if low in _SPELLED:
            out.append(_SPELLED[low])
            i += 1
            continue
        if text not in _NOT_SEAL and re.fullmatch(r"[A-Za-z0-9\-]+", text) and (
                any(ch.isdigit() for ch in text) or (text.isupper() and len(text) <= 3)):
            # "MX MX guión ..." (real transcript): a letter group repeated back to back is one ASR stutter.
            # Letters only: "cuatro cuatro" stays 44. Residual risk, accepted and written down: a driver who
            # really says "MX MX 48213" against a paper reading "MX-48213" would match (a "MXMX" paper HOLDs).
            if not (text.isalpha() and out and out[-1] == text):
                out.append(text)
            i += 1
            continue
        break
    return "".join(out), i, ""


def _seals(toks: list[tuple[str, str]], lows: list[str]) -> tuple[str | None, str, set[int]]:
    """Every seal mention: (seal or None, why None when one was said, token indices the mentions cover)."""
    values: list[str] = []
    whys: list[str] = []
    covered: set[int] = set()
    k = 0
    while k < len(toks):
        if lows[k] not in _SEAL_KEYWORDS:
            k += 1
            continue
        i = k + 1
        while i < len(toks) and (toks[i][0] == "comma" or lows[i] in _SEAL_LEAD_IN
                                 or (toks[i][1] == "." and lows[i - 1] in ("no", "num", "nro", "n"))):
            i += 1
        while True:
            value, i, why = _seal_value(toks, lows, i)
            if why:
                whys.append(why)
            if any(ch.isdigit() for ch in value):
                values.append(value)
            if i >= len(toks) or toks[i][0] != "word" or not value:
                break
            # The id stopped at a word. Another id after it ("..., perdón, MX-48231", "MX-48213 y MX-48214") is a
            # second value; any other word followed by more seal-like tokens means we cannot tell where the id ends.
            skip = _phrase_at(lows, i, _CORRECTIONS) or (1 if lows[i] in _ALTERNATIVES else 0)
            nxt = i + max(skip, 1)
            while nxt < len(toks) and toks[nxt][0] == "comma":
                nxt += 1
            if nxt >= len(toks) or not _seal_like(toks, lows, nxt) or lows[i] in _ID_NOUNS:
                break
            if not skip:
                whys.append(f"seal interrupted by {toks[i][1]!r}")
                break
            i = nxt
        covered.update(range(k, i))
        k = max(i, k + 1)
    if whys:
        return None, "; ".join(whys), covered
    distinct = sorted({v.replace("-", "").upper() for v in values})
    if len(distinct) > 1:
        return None, f"seals disagree in transcript: {distinct}", covered
    return (values[0] if values else None), "", covered


# --- count -------------------------------------------------------------------------------------


def _count(ws: list[str]) -> tuple[int | None, str]:
    phrases: list[tuple[int, int, int | None]] = []  # (start, end, value)
    i = 0
    while i < len(ws):
        n = _number_at(ws, i)
        if n:
            phrases.append((i, n[1], n[0]))
            i = n[1]
        else:
            i += 1
    ends = {e for _, e, _ in phrases}
    starts = {s for s, _, _ in phrases}

    def standalone(e: int) -> bool:  # nothing but punctuation, a conjunction or a correction after it
        return _w(ws, e) in ("", ".", ",", "-", "|") or _w(ws, e) in _ALTERNATIVES or bool(
            _phrase_at(ws, e, _CORRECTIONS))

    counts: list[tuple[int, int, int | None]] = []  # (start, unit end, value)
    other: dict[int, str] = {}  # value -> why it competes with the count
    unit_ends: set[int] = set()
    bare: list[tuple[int, int, int]] = []
    for s, e, v in phrases:
        if _w(ws, s - 1) in _ID_NOUNS or _w(ws, e) in _WEIGHT_UNITS:
            continue
        if _is_unit(_w(ws, e)):
            u = e + 1
        elif _is_filler(_w(ws, e)) and _is_unit(_w(ws, e + 1)):
            u = e + 2
        else:
            if v is not None:
                bare.append((s, e, v))
            continue
        counts.append((s, u, None if s in ends else v))  # a group said right before it: 'veinte cuatro tarimas'
        unit_ends.add(u)
    if not counts:
        return None, "no count found in transcript"

    for s, e, v in bare:
        if not standalone(e):
            continue
        p = s
        while _w(ws, p - 1) in _AFTER_CORRECTION and p > s - 2:
            p -= 1
        while _w(ws, p - 1) in (",", ".", "-"):
            p -= 1
        q = e
        while _w(ws, q) in (",", ".", "-"):
            q += 1
        before, after = _phrase_ending_at(ws, p, _CORRECTIONS), _phrase_at(ws, q, _CORRECTIONS)
        if before or after:
            marker = " ".join(ws[p - before:p] if before else ws[q:q + after])
            other.setdefault(v, f"self-correction: {marker!r}")
        elif (_w(ws, e) in _ALTERNATIVES and e + 1 in starts) or (
                _w(ws, s - 1) in _ALTERNATIVES and (s - 1 in ends or s - 1 in unit_ends)):
            other.setdefault(v, f"alternative: {_w(ws, e) if e + 1 in starts else _w(ws, s - 1)!r}")
        elif _w(ws, s - 1) in _COUNT_VERBS:
            other.setdefault(v, "")

    for s, u, v in counts:
        said = " ".join(ws[s:u])
        q = u + 1 if _w(ws, u) == "," else u
        if _phrase_ending_at(ws, s, _APPROX_BEFORE) or _phrase_at(ws, q, _APPROX_AFTER):
            return None, f"approximate count in transcript: {said!r}"
        for k in range(s - 1, max(s - 5, -1), -1):
            if ws[k] in (".", ",", "-", "|"):
                break
            if ws[k] in _NEGATIONS or (ws[k] == "t" and _w(ws, k - 1) in _NEGATED_CONTRACTIONS):
                return None, f"negated count in transcript: {' '.join(ws[k - (ws[k] == 't'):u])!r}"
        if v is None:
            return None, f"unclear count in transcript: {said!r}"

    values = {v for _, _, v in counts if v is not None} | set(other)
    if len(values) == 1:
        return values.pop(), ""
    notes = sorted({n for n in other.values() if n})
    why = f"ambiguous counts in transcript: {sorted(values)}"
    return None, why + (f" ({'; '.join(notes)})" if notes else "")


def _parse_voice(transcript: str) -> tuple[int | None, str, str | None, str]:
    toks = _tokens(transcript)
    lows = [_LOW_PUNCT.get(k, t.lower()) for k, t in toks]
    seal, seal_why, covered = _seals(toks, lows)
    count, why = _count(["|" if k in covered else w for k, w in enumerate(lows)])
    return count, why, seal, seal_why


def parse_count(transcript: str) -> tuple[int | None, str]:
    """The one count said with a unit noun ('doce tarimas', 'a dozen pallets'); else None + why."""
    count, why, _, _ = _parse_voice(transcript)
    return count, why


def parse_seal(transcript: str) -> str | None:
    """The seal id said after 'sello'/'seal'/'precinto'/'marchamo' (number words -> digits); None if unclear."""
    return _parse_voice(transcript)[2]


def voice_from_transcript(
    data: dict[str, Any], *, model: str, provider: str, source_sha256: str
) -> VoiceWitness:
    meta = dict(model=model, provider=provider, source_sha256=source_sha256)
    """Measured ASR shape: {sessionId, transcript, audioDurationMs, turns[]}. Empty transcript -> OK, count None."""
    if not isinstance(data, dict) or not isinstance(data.get("transcript"), str):
        keys = sorted(data)[:12] if isinstance(data, dict) else type(data).__name__
        return VoiceWitness(status="ERROR", detail=f"no 'transcript' string; keys={keys}", **meta)
    text = data["transcript"]
    count, why, seal, seal_why = _parse_voice(text)
    if not text.strip():
        why = "empty transcript"
    lang = data.get("language") if isinstance(data.get("language"), str) else None
    dur = data.get("audioDurationMs") if _is_int(data.get("audioDurationMs")) else None
    return VoiceWitness(
        status="OK", transcript=text, language=lang, count=count, seal=seal,
        detail="; ".join(x for x in (why, seal_why) if x), audio_duration_ms=dur, **meta
    )


# --- fixtures ------------------------------------------------------------------------------


def from_fixture(path: str | Path, source_sha256: str = "") -> Witness:
    """Load a hand-written FIXTURE witness file. Refuses files not labelled FIXTURE."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    kind = data.get("witness")
    prov = data.get("provenance", "")
    fx_sha = sha256_file(p)
    meta = dict(model=f"FIXTURE:{p.name}", provider="FIXTURE", source_sha256=source_sha256)
    cls = {"vision": VisionWitness, "doc": DocWitness, "voice": VoiceWitness}.get(kind)
    if cls is None:
        raise ValueError(f"{p}: 'witness' must be vision|doc|voice, got {kind!r}")
    if FIXTURE_PROVENANCE_PREFIX not in p.name or not str(prov).startswith(FIXTURE_PROVENANCE_PREFIX):
        return cls(status="ERROR", detail=f"{p.name}: not labelled FIXTURE (name and provenance)", **meta)
    if "status" in data and data["status"] != "OK":  # fixtures may model a dead witness
        return cls(status=data["status"], detail=data.get("detail", ""), **meta)
    note = f"fixture {p.name} sha256={fx_sha}"
    if kind == "vision":
        w = vision_from_fields(data, concept=data.get("concept", "pallet"), **meta)
    elif kind == "doc":
        w = doc_from_fields(data, **meta)
    else:
        w = voice_from_transcript(data, **meta)
    return type(w)(**{**_fields(w), "detail": "; ".join(x for x in (note, w.detail) if x)})


def _fields(w: Witness) -> dict[str, Any]:
    from dataclasses import fields

    return {f.name: getattr(w, f.name) for f in fields(w)}


# --- call records (what goes into a receipt about one API call) -----------------------------

RAW_LIMIT = 2000


def call_record(r: Any, raw_limit: int = RAW_LIMIT) -> dict[str, Any]:
    """Receipt view of one meta_api.ApiResult: request without base64, raw answer truncated, masks dropped."""
    rec: dict[str, Any] = {
        "model": r.model, "provider": r.provider, "status": r.status, "http": r.http_status,
        "utc": r.utc, "s": r.seconds, "request": r.request, "media": r.media,
    }
    if r.detail:
        rec["detail"] = r.detail[:raw_limit]
    if r.raw_text is not None and r.data is None and r.status == "OK":  # SAM SSE
        events = parse_sse(r.raw_text)
        text = sam_text(events)
        rec["sse_events"] = [str(e.get("type")) for e in events]
        rec["raw_chars"] = len(r.raw_text)
        rec["raw"] = strip_masks(text)[:raw_limit] if text is not None else None
        for ev in events:
            resp = ev.get("response") if ev.get("type") == "response.completed" else None
            if isinstance(resp, dict) and resp.get("usage") is not None:
                rec["usage"] = resp["usage"]
    elif r.data is not None:
        rec["raw"] = json.dumps(r.data, ensure_ascii=False)[:raw_limit]
        if r.data.get("usage") is not None:
            rec["usage"] = r.data["usage"]
    return rec
