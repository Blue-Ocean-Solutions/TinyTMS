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
# said for it anywhere in the note. Self-corrections ("doce… no, trece", "nope, uh, thirteen total"), alternatives
# ("doce o trece"), approximations ("unas doce"), hedges ("creo que son doce"), negations ("no traigo doce"), more said
# after it ("y una suelta", "and a half"), a count taken from the paperwork and unclear number words ("veinte cuatro")
# give None and a detail that says why. So does a seal that does not read one way: corrected or said two ways
# anywhere ("... Sorry, MX-48231."), hedged, negated, broken or missing ("el sello no es ...", "... viene roto"), taken
# from the paperwork ("según la carta ..."), said after any word but a lead-in ("el sello de la carta es ..."), run
# into what follows it ("... uno tres cuatro mil ochocientos ..."), or said in the plural (ORDER Nube-9: the witness
# returns the right value, or None with a non-empty reason; it never picks one of two readings).

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
# "veinti dos", "dieci seis": how an ASR may split one word
_ES_SPLIT = {"veinti": 20, "dieci": 10}
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
                           *_ES_SPLIT, "cero", "zero", "cien", "mil", "hundred", "thousand"})
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
    "make that", "correction", "scratch that", "make it", "mejor", "nope", "oops", "whoops", "my bad", "strike that",
    "let me recount", "error", "no es cierto", "en realidad", "aguanta", "conté mal", "conte mal", "déjame contar",
    "dejame contar",
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
# The driver is not sure: "creo que son doce", "tal vez doce", "maybe twelve", "doce tarimas, creo".
_HEDGES = tuple(tuple(p.split()) for p in (
    "creo", "supongo", "pienso", "parece", "quizá", "quizás", "quiza", "quizas", "tal vez", "talvez", "a lo mejor",
    "probablemente", "posiblemente", "calculo", "puede que", "según yo", "segun yo", "maybe", "perhaps", "probably",
    "possibly", "think", "i think", "guess", "i guess", "believe", "i believe", "might", "may be", "pretty sure",
    "fairly sure", "want to say", "not sure", "could be",
))
_HEDGES_AFTER = tuple(tuple(p.split()) for p in (
    "creo", "creo yo", "supongo", "me parece", "quizá", "quizás", "tal vez", "a lo mejor", "según yo", "segun yo",
    "si no me equivoco", "si mal no recuerdo", "no estoy seguro", "no estoy segura", "no me acuerdo", "no lo veo",
    "i think", "i guess", "i believe", "maybe", "probably", "perhaps", "i m not sure",
    "not sure", "i m pretty sure", "pretty sure", "if i m not mistaken", "if i remember", "if i read", "if i counted",
    "or something", "don t quote me", "can t really read", "can t read", "i didn t count", "could be",
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
_SPEECH_FILLERS = frozenset({"eh", "ehh", "em", "emm", "mm", "mmm", "um", "uh", "uhm", "este", "hmm", "hm", "mhm", "ah",
                             "ahh", "er", "erm", "pos", "pues", "espérame", "esperame", "espera", "oye"})
_FILLER_PHRASES = tuple(tuple(p.split()) for p in ("a ver", "déjame ver", "dejame ver", "let me see", "let s see"))
_NOT_FILLER = frozenset(
    "con sin por para al hasta desde entre sobre cada with without in on for to from at by per each "
    "el la los las lo the and pero but si if que".split()
) | _NEGATIONS | _COUNT_VERBS | _ID_NOUNS | _WEIGHT_UNITS | _NUMBER_WORDS | {w for p in _CORRECTIONS for w in p}
# A number said with no unit is a count value when one of these follows it: "nope, uh, thirteen total".
_COUNT_AFTER = tuple(tuple(p.split()) for p in (  # ("en total", "de ellas" already count: "en", "de" name nothing)
    "total", "completas", "completos", "altogether", "all together", "exactas", "exactos", "justas", "justos",
))
_MORE = frozenset({"más", "mas", "more", "extra", "menos", "less", "fewer"})  # "y una más": 13, or 1? no count
_ARTICLES = frozenset({"un", "una", "one"})  # "bueno, un saludo": an article, a count only before a count noun
# Said after a count, any of these adds to it or makes it a part: "doce tarimas más una suelta", "and a half".
_ADDED = frozenset("más mas plus además aparte another otra otras otro otros extra media half cacho change partial "
                   "parcial incompleta incompleto incompletas incompletos suelta sueltas suelto sueltos loose falta "
                   "faltan faltó falto faltaba missing short".split())  # "me faltó cargar una": 11 or 12?
# Before the keyword or the count, the value is the paperwork's or someone else's, not what the driver saw: "según la
# carta porte el sello es ...", "the BOL says fourteen pallets", "dispatch told me the seal is ...".
_SOURCES = frozenset("según segun according bol paperwork papeles papel carta factura bill manifest manifiesto "
                     "documento documentos dispatch despacho despachador shipper remitente".split())
# "a las tres", "at three", "cada una", "the one on the BOL": the number after these is not a count
_NOT_COUNT_BEFORE = frozenset({"las", "la", "at", "cada", "each", "the"})
_WEIGHT_WORDS = frozenset("peso pesa pesan pesamos weight weighs weigh weighing".split())
# What may stand between an id noun and its number: "trailer number is 53", "la factura es 48213", "en el andén 4".
_ID_LEAD = frozenset("es is el de del of no nº en in on a su mi my ,".split())

# Seal mentions. Spelled letter names are only the ones that are not also everyday words ("de", "te", "ese", "ve").
_SEAL_KEYWORDS = frozenset({"sello", "seal", "precinto", "marchamo"})
_SEAL_PLURALS = frozenset({"sellos", "seals", "precintos", "marchamos"})  # more than one seal: the witness has one
# The only words that may stand between the keyword and the id (ORDER Nube-9): any other word there gives None and
# the reason, the id is never read past it ("el sello de la carta es ...", "the seal on the paperwork is ...").
# "No." is the abbreviation ("sello No. MX-48213"; "el sello no es MX-48213" negates it); "de seguridad" is part of
# the keyword ("sello de seguridad").
_SEAL_LEAD_IN = frozenset("es is el the número numero núm number nº".split())
_SEAL_ABBREVIATIONS = frozenset({"no", "num", "núm", "nro", "n"})  # before a "."
# Said of the seal, any of these means it cannot be taken as read: "el sello MX-48213 viene roto", "falta el sello",
# "seal MX-48213 is missing". After a negation they say the seal is fine: "no está roto", "is not broken", "no falta".
_SEAL_BAD = frozenset(
    "roto rota rotos rotas dañado dañada dañados dañadas violado violada violentado violentada forzado forzada abierto "
    "abierta cortado cortada quebrado quebrada alterado alterada despegado despegada falta faltan faltó falto faltaba "
    "faltante perdido perdida perdí perdi perdió perdio diferente distinto distinta broken damaged tampered cut opened "
    "missing lost removed gone mismatch mismatched wrong different lacks lacking loose suelto suelta flojo floja "
    "violaron violó violo cortaron rompieron rompió rompio abrieron off bad mal".split()
)
# A part of the seal said: "seal 7731, last four", "los últimos cuatro".
_SEAL_PARTIAL = frozenset("last últimos últimas ultimos ultimas".split())
# A letter said after the digits in words: "con letra b", "with a B at the end".
_LETTER_WORDS = frozenset("letra letras letter letters".split())
# The trailer itself said to be open or broken: "la caja viene abierta", "the trailer door is open".
_TRAILER = frozenset("caja tráiler trailer remolque puerta puertas door doors".split())
# The seal said as a verb: "it's not sealed", "sin sellar", "came unsealed".
_SEALING = frozenset("sealed unsealed sellar sellaron sellado sellada sellados selladas".split())
# Said of the seal these say it is fine ("el sello coincide", "it matches the paperwork"); negated, that it is not
# ("no coincide", "doesn't match").
_SEAL_MATCH = frozenset(
    "coincide coinciden corresponde corresponden cuadra cuadran concuerda match matches matched matching agrees same "
    "igual mismo misma".split()
)
# Words that may stand between a negation and the word it negates: "no está roto", "has not been opened".
_NEGATION_GAP = frozenset("está esta estaba están estan viene vienen venía venia ha han has have had been be is was "
                          "are fue se le lo la it es el the a un una".split())
# A negation about the cargo, not the seal: "MX-48213 no hay material peligroso".
_HAZMAT = frozenset("peligroso peligrosa peligrosos peligrosas hazmat hazardous".split())
# Words that may stand between a negation and the keyword before it: "no traigo el sello", "there is no seal".
_SEAL_BEFORE_SKIP = frozenset(
    "el la los las lo un una the a an any ningún ningun ninguno ninguna su sus mi this that traigo trae traemos tengo "
    "tiene tienen tenemos hay había habia viene vienen venía venia llevo lleva llevamos has have had got is está esta "
    "es was there veo vi se le".split()
)
# English letter names that are not words: "em ex dash four ...", "el zee ..." (with _WORD_LETTERS).
_EN_SPELLED = {"bee": "B", "cee": "C", "dee": "D", "gee": "G", "jay": "J", "kay": "K", "pee": "P", "cue": "Q",
               "tee": "T", "vee": "V", "zee": "Z", "ex": "X", "ef": "F", "ar": "R", "why": "Y"}
_SPELLED = {**_EN_SPELLED, "be": "B", "ce": "C", "efe": "F", "ge": "G", "hache": "H", "jota": "J", "ka": "K",
            "ele": "L", "eme": "M", "ene": "N", "eñe": "Ñ", "pe": "P", "cu": "Q", "erre": "R", "uve": "V", "equis": "X",
            "ye": "Y", "zeta": "Z"}
_NOT_SEAL = frozenset({"OK", "NO", "SI", "YA"})
# Letter names that are also everyday words ("a", "de", "te", "ese"): a letter only next to another spelled letter
# ("a be", "te equis", "ese eme equis", "ce a guión ..."), or after a dash at the end of a sentence ("guión de.").
_WORD_LETTERS = {"a": "A", "de": "D", "te": "T", "ese": "S", "em": "M", "el": "L"}  # "el" only before an English name
# The local ASR writes a seal's letters in lowercase ("el sello es mx guión cuatro..."). Function words never are seal
# letters, whatever follows them ("el sello es de plástico", "cuatro ocho y dos uno tres").
_NOT_SEAL_LETTERS = frozenset(
    "es el la los las de del y o no en con sin que se su mi tu al lo "
    "a e u ni le les me te nos mis tus sus por pa pal ya si son era fue ha he han hay va van voy da dan doy ver vez "
    "muy tan mas ese esa eso ok oh "
    "an and are as at by do for go has i if in is it its of on or per so the to up us was we yes".split()
) | _WEIGHT_UNITS | _SPEECH_FILLERS
_LETTER_PAIRS = {("doble", "ve"): "W", ("doble", "uve"): "W", ("uve", "doble"): "W", ("doble", "u"): "W",
                 ("i", "griega"): "Y", ("double", "u"): "W", ("double", "you"): "W"}
# Words that can follow a seal's last digit without being counted by it: "... uno tres traigo doce tarimas", "... tres
# y son de madera", "... tres de la caja", "... tres peso cuatro mil ...", "... tres pesa cuatro mil ...", "... tres
# sin material peligroso", "... tres buenas tardes". Any other word right after digits said in a run ("... uno tres
# doce paquetes", "... tres dos horas esperando") may be counted by the last number, and a copula may make it the
# subject ("... tres cuatro es el andén"), so the id does not read one way.
_ENDS_SEAL = frozenset(
    "el la lo los las the pero but que traigo llevo tengo cargo cargué cargue entrego hay viene trae tiene lleva has "
    "does did seems looks i we it ok okay gracias thanks thank please listo repito peso weight sin without del "
    "pesa pesan weighs weigh son ahí ahi buenas buenos buen muchas nada saludos saludo".split()
)
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
_CLITICS = frozenset({"s", "t", "re", "ll", "ve", "d", "m"})


def _tokens(text: str) -> list[tuple[str, str]]:
    """(kind, text) per token. 'twenty-four' splits into words; 'MX-48213' stays one token. The 's' of "seal's" (and
    the 't' of "don't") is a 'clitic': never a seal letter."""
    out: list[tuple[str, str]] = []
    for m in _TOKEN.finditer(text):
        kind, t = m.lastgroup or "", m.group()
        parts = [p for p in re.split(r"[-‐]", t) if p]
        if kind == "word" and len(parts) > 1 and all(p.lower() in _NUMBER_WORDS for p in parts):
            out += [("word", p) for p in parts]
        elif kind == "word" and t.lower() in _CLITICS and m.start() >= 2 and text[m.start() - 1] in "'’" and (
                text[m.start() - 2].isalnum()):
            out.append(("clitic", t))
        else:
            out.append((kind, t))
    return out


def _w(ws: list[str], i: int) -> str:
    return ws[i] if 0 <= i < len(ws) else ""


def _is_unit(w: str) -> bool:
    return bool(_UNIT.fullmatch(w))


def _unit(ws: list[str], j: int) -> bool:
    """A count noun at j, and not one that names an id: 'caja 53' is the trailer box, 'doce cajas' a count."""
    return _is_unit(_w(ws, j)) and not (_w(ws, j) in _ID_NOUNS and _number_at(ws, j + 1, dozens=False))


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
    if w in _ES_SPLIT:
        return (_ES_SPLIT[w] + _ES_ONES[ws[j + 1]], j + 2) if _w(ws, j + 1) in _ES_ONES else None
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
    m = _EN_ONES.get(w) or _EN_TEENS.get(w) or (1 if w in ("a", "an") else None)  # "fifteen hundred"
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


def _letters(toks: list[tuple[str, str]], lows: list[str], j: int) -> bool:
    t = toks[j][1]
    return toks[j][0] == "word" and len(t) <= 3 and t.isascii() and t.isalpha() and t.islower() and (
        lows[j] not in _NOT_SEAL_LETTERS and lows[j] not in _NUMBER_WORDS and lows[j] not in _SEAL_BAD)


def _lowercase_letters(toks: list[tuple[str, str]], lows: list[str], i: int) -> bool:
    """Is the token at i a seal's letters written lowercase: up to 3 ASCII letters, not a function word, with a dash or
    digits after it (past more such letters: 'm x guión', the stutter 'mx mx guión')? Number words count as its digits
    only when they are not a count or a weight: 'all twelve pallets' is not a seal. Only before the seal's first
    digit: after it, a lowercase word ends the seal ('MX-48213, got twelve pallets')."""
    if not _letters(toks, lows, i):
        return False
    # "seal is new 7731", "seal tag 7731": a group with a vowel may be a word, so it is letters only before a dash
    groups = []
    for q in range(i, len(toks)):
        if _letters(toks, lows, q):
            groups.append(lows[q])
        elif not (toks[q][0] == "comma" or lows[q] in _SPEECH_FILLERS or lows[q] in _SPELLED
                  or lows[q] in _WORD_LETTERS):
            break
    word = any(len(g) > 1 and set(g) & set("aeiou") for g in groups)
    for j in range(i + 1, len(toks)):
        kind, text = toks[j]
        if kind == "dash" or lows[j] in _SEAL_DASH:
            return True
        if any(ch.isdigit() for ch in text):
            return not word
        if lows[j] not in ("un", "una") and _number_at(lows, j, dozens=False):
            return not word and (_seal_like(toks, lows, j) or all(len(lows[q]) == 1 for q in range(i, j)
                                                                  if toks[q][0] == "word"))
        if lows[j] in _WORD_LETTERS and _letters(toks, lows, j - 1) and len(lows[j - 1]) == 1:
            continue  # "c a guion ...": the letter A
        if not (_letters(toks, lows, j) or lows[j] in _SPELLED or kind == "comma" or lows[j] in _SPEECH_FILLERS):
            return False
    return False


def _unclear_lowercase_prefix(toks: list[tuple[str, str]], lows: list[str], i: int) -> str:
    """The lowercase letter groups from i to the digits read one way ('mx', 'm x', the stutter 'mx mx'): ''. Else what
    was said: two different groups ('hoy mx 48213'), or letters glued onto an id that has its own ('s MX-48213')."""
    groups, j = [], i
    while j < len(toks):
        if _letters(toks, lows, j):
            groups.append(lows[j])
        elif lows[j] in _WORD_LETTERS and groups and len(groups[-1]) == 1:
            groups.append(lows[j])  # "c a guion ...": the letter A
        elif not (toks[j][0] == "comma" or lows[j] in _SPEECH_FILLERS or lows[j] in _SPELLED):
            break
        j += 1
    nxt = toks[j][1] if j < len(toks) else ""
    glued = j < len(toks) and toks[j][0] == "word" and any(ch.isalpha() for ch in nxt) and (
        lows[j] not in _SEAL_DASH and not _number_at(lows, j, dozens=False))
    if glued or (len(set(groups)) > 1 and any(len(g) > 1 for g in groups)):
        return " ".join(lows[i:j + 1])
    return ""


def _word_letter(lows: list[str], i: int) -> str:
    """The letter a letter name that is also a word stands for here ('a be', 'te equis', 'em ex', 'el zee'), or ''."""
    w, nxt = lows[i], _w(lows, i + 1)
    if w not in _WORD_LETTERS or w == "de":
        return ""
    return _WORD_LETTERS[w] if nxt in (_EN_SPELLED if w == "el" else _SPELLED) else ""


def _seal_like(toks: list[tuple[str, str]], lows: list[str], i: int) -> bool:
    """Would the token at i continue a seal id (and not start a count or a weight)?"""
    kind, text = toks[i]
    if kind not in ("word", "num") or text in _NOT_SEAL or lows[i] in ("un", "una"):
        return False
    n = _number_at(lows, i, dozens=False)
    if n:
        after = _w(lows, n[1])
        return not (_unit(lows, n[1]) or after in _WEIGHT_UNITS)
    nxt = _w(lows, i + 1)
    return lows[i] in _SPELLED or (lows[i], nxt) in _LETTER_PAIRS or bool(_word_letter(lows, i)) or (
        lows[i] in ("doble", "double", "triple") and nxt in _DIGIT_OF) or _lowercase_letters(toks, lows, i) or bool(
        re.fullmatch(r"[A-Za-z0-9-]+", text) and (any(c.isdigit() for c in text) or (text.isupper() and len(text) <= 3))
    )


def _number_run(lows: list[str], j: int) -> tuple[list[tuple[int, int, int]], int]:
    """Number groups said back to back from j: ([(value, start, end)], end)."""
    groups: list[tuple[int, int, int]] = []
    while (n := _number_at(lows, j, dozens=False)) and n[0] is not None:
        groups.append((n[0], j, n[1]))
        j = n[1]
    return groups, j


def _compound(group: tuple[int, int, int]) -> bool:
    """A group that is not one digit: 'doce', 'cuatro mil ochocientos', 'cuarenta y ocho'."""
    value, start, end = group
    return value >= 10 or end - start > 1


def _id_shaped(toks: list[tuple[str, str]], lows: list[str], j: int) -> bool:
    """Does the token at j start an id, not a count: digits, letters, spelled letters, or number words that are not
    one small number ('cuatro ocho dos tres uno', 'cuarenta y ocho mil ...', '48231')? After a correction this is a
    second seal; a small number there ('..., no, trece') may correct the count."""
    kind, text = toks[j]
    if kind not in ("word", "num") or text in _NOT_SEAL or lows[j] in ("un", "una", "a", "an"):
        return False
    if _number_at(lows, j, dozens=False):
        groups, e = _number_run(lows, j)
        after = _w(lows, e)
        if not groups or _unit(lows, e) or after in _WEIGHT_UNITS:
            return False
        return len(groups) > 1 or groups[0][0] >= 1000
    return _seal_like(toks, lows, j) and text != "I"


def _one_number(toks: list[tuple[str, str]], lows: list[str], j: int) -> bool:
    """One number and nothing of an id around it ('one thousand', '48231'): after a correction it may correct the count
    as well as the seal."""
    if not _number_at(lows, j, dozens=False):
        return False
    groups, e = _number_run(lows, j)
    return len(groups) == 1 and not (e < len(toks) and _continues_seal(toks, lows, e))


def _continues_seal(toks: list[tuple[str, str]], lows: list[str], j: int) -> bool:
    """Would the word at j be read as more of the id right after its digits ('guión', 'oh', 'doble ocho', 'AB')?"""
    w, nxt, text = lows[j], _w(lows, j + 1), toks[j][1]
    return ((w in _SPELLED and w not in _EN_SPELLED) or w in _SEAL_DASH or w == "oh" or (
            w in ("doble", "double", "triple") and nxt in _DIGIT_OF)
            or (w, nxt) in _LETTER_PAIRS or any(ch.isdigit() for ch in text)
            or (text.isupper() and text.isalpha() and len(text) <= 3 and text not in _NOT_SEAL and text != "I"))


def _ends_seal(lows: list[str], j: int) -> bool:
    """Does the word at j end a seal's digits cleanly (it cannot be what the last number counts)?"""
    w = lows[j]
    return (w in _ENDS_SEAL or w in _NEGATIONS or w in _NEGATED_CONTRACTIONS or w in _SPEECH_FILLERS
            or w in _SEAL_KEYWORDS or w in _ALTERNATIVES or bool(_phrase_at(lows, j, _CORRECTIONS))
            or (w == "de" and _w(lows, j + 1) in ("la", "el")) or (w in _ID_NOUNS and not _unit(lows, j))
            or (w in _NEGATION_GAP and bool(_negated_word(lows, j)[0])))  # "... tres está dañado", "is broken"


def _seal_value(toks: list[tuple[str, str]], lows: list[str], i: int) -> tuple[str, int, str]:
    """Read one seal id from token i: (id, index of the token that ended it, why it is unreadable or '')."""
    out: list[str] = []
    lower = False  # a lowercase letter group already read
    written = False  # digits written as digits ('48213') already read: number words never lengthen them
    spoken: list[tuple[int, int, int]] = []  # number groups read so far
    run_end = -1  # where the last digits said as words ended ("cuatro oh" then "dos cientos ..." runs on)
    start = i

    def lower_pause(j: int) -> bool:  # a comma or a filler right before j: "sello MX, doce tarimas"
        return j > 0 and (toks[j - 1][0] == "comma" or lows[j - 1] in _SPEECH_FILLERS)

    while i < len(toks):
        kind, text = toks[i]
        low = lows[i]
        digits = any(ch.isdigit() for ch in "".join(out))
        if kind in ("stop", "clitic"):
            break
        if kind == "comma" or (low in _SPEECH_FILLERS and not (not digits and _word_letter(lows, i))):
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
            i = run_end = i + 2
            continue
        if (low, nxt) in _LETTER_PAIRS:
            out.append(_LETTER_PAIRS[low, nxt])
            i += 2
            continue
        if low == "oh" and out and out[-1][-1:].isdigit():  # "four oh two"
            out.append("0")
            i = run_end = i + 1
            continue
        if digits and out[-1] == "-" and _letters(toks, lows, i) and low not in _SPELLED and not (
                _number_at(lows, i, dozens=False)):
            out.append(text.upper())  # "... two one three dash b": the letter after the dash is the id's
            i += 1
            continue
        if low not in ("un", "una") and _number_at(lows, i, dozens=False):
            groups, j = _number_run(lows, i)
            if not groups:  # "4.5": a number, but not digits of an id
                break
            if len(groups) > 1 and lows[i].isdecimal() and not lows[groups[1][1]].isdecimal():
                groups, j = groups[:1], groups[0][2]  # "48213 doce tarimas": number words never lengthen digits
            said = " ".join(lows[start:j + 1])
            after = _w(lows, j)
            if _unit(lows, j) or after in _WEIGHT_UNITS or (_is_filler(after) and _unit(lows, j + 1)):
                what = "a weight" if after in _WEIGHT_UNITS else "a count"
                if len(groups) == 1 and run_end != i and out and not digits and not lower_pause(i):
                    # "t x six hundred and eighty eight boxes": TX-680 and 8 boxes, or TX and 688 boxes?
                    return "", j, f"seal letters run into {what}: {' '.join(lows[start:j + 1])!r}"
                if len(groups) == 1 and run_end != i:
                    break  # "..., doce tarimas": the number after the seal is the count
                # "... uno tres cuatro mil ochocientos kilos" with no stop reads two ways (48213 and 4800 kg, or
                # 482134 and 1800 kg), so no seal; a stop before the weight ("... uno tres. Cuatro mil ...") reads one.
                # set the whole run aside: its last group may be the count or the seal's ("... tres dos tarimas":
                # 2 pallets, or 482132 and no count), so no count comes out of it either
                return "", j, f"seal digits run into {what}: {' '.join(lows[i:j + 1])!r}"
            if written:  # "MX-48213 dos horas esperando": the number goes with the words after it
                if "".join(str(v) for v, _, _ in groups) == re.sub(r"\D", "", "".join(out)) and len(groups) > 1:
                    i = j  # "48213, cuatro ocho dos uno tres": the same digits said again
                    continue
                if j < len(toks) and toks[j][0] == "word" and not _ends_seal(lows, j):
                    break
                return "", j, f"seal digits run into a number: {said!r}"
            if len(groups) == 1 and lows[i].isdecimal() and not spoken:  # "mx 48213 got ...": digits as digits
                out.append(lows[i])
                written, i = True, j
                continue
            if j < len(toks) and toks[j][0] == "word" and not _continues_seal(toks, lows, j) and not (
                    _ends_seal(lows, j)):
                if len(groups) == 1 and digits and run_end != i:
                    break  # "..., uno tres, dos horas esperando": after a pause the number goes with the words after it
                return "", j, f"seal digits run into the next words: {' '.join(lows[i:j + 1])!r}"
            if len(spoken + groups) > 1 and any(_compound(g) for g in spoken + groups):
                # "... dos uno tres cuatro mil ochocientos sin material peligroso": 48213 and 4800, or 482134800?
                return "", j, f"seal digits run into a number: {said!r}"
            for (_, _, e), (_, s, _) in zip(groups, groups[1:]):
                if _unclear_pair(lows[e - 1], lows[s]):
                    return "", j, f"unclear seal digits: {lows[e - 1] + ' ' + lows[s]!r}"
            out += [lows[s] if e == s + 1 and lows[s].isdecimal() else str(v) for v, s, e in groups]  # "007" stays
            written = written or any(lows[s].isdecimal() or _GROUPED.fullmatch(lows[s]) for _, s, _ in groups)
            spoken += [g for g in groups if not lows[g[1]].isdecimal()]
            i = j
            if not lows[j - 1].isdecimal():
                run_end = j
            continue
        if low in _SPELLED and not (digits and low in _EN_SPELLED and out[-1] != "-"):  # "... 7731 gee thanks"
            out.append(_SPELLED[low])
            i += 1
            continue
        after_letter = _w(lows, i - 1) in _SPELLED or _w(lows, i - 1) in _WORD_LETTERS or (
            i > 0 and len(lows[i - 1]) == 1 and _letters(toks, lows, i - 1))  # "ce a guión ...", "c a guion ..."
        before_digits = nxt in _SEAL_DASH or nxt == "-" or bool(_number_at(lows, i + 1, dozens=False))
        if low in _WORD_LETTERS and kind == "word" and (
                (not digits and (_word_letter(lows, i) or (after_letter and before_digits)))
                or (digits and out[-1] == "-" and (i + 1 == len(toks) or toks[i + 1][0] == "stop"))):
            out.append(_WORD_LETTERS[low])
            i += 1
            continue
        if not digits and _lowercase_letters(toks, lows, i):
            if not lower and (unclear := _unclear_lowercase_prefix(toks, lows, i)):
                return "", i, f"unclear seal letters: {unclear!r}"
            lower = True
            text = text.upper()  # "mx guión cuatro ..." reads as "MX guión cuatro ..."
        if digits and text == "I":
            break  # "Seal MX-48213 I have twelve pallets": the English pronoun, not a letter of the seal
        if text not in _NOT_SEAL and re.fullmatch(r"[A-Za-z0-9\-]+", text) and (
                any(ch.isdigit() for ch in text) or (text.isupper() and len(text) <= 3)):
            # "MX MX guión ..." (real transcript): a letter group repeated back to back is one ASR stutter.
            # Letters only: "cuatro cuatro" stays 44. Residual risk, accepted and written down: a driver who
            # really says "MX MX 48213" against a paper reading "MX-48213" would match (a "MXMX" paper HOLDs).
            if not (text.isalpha() and out and out[-1] == text):
                out.append(text)
            written = written or any(ch.isdigit() for ch in text)
            i += 1
            continue
        if digits and kind == "word" and len(text) == 1 and text.isalpha() and text.isascii() and (
                _letters(toks, lows, i) or i + 1 == len(toks) or toks[i + 1][0] == "stop"):
            # "seal mx 48213 b", "sello 48213 a": a letter of the id, or not? Never dropped, never guessed.
            return "", i + 1, f"unclear seal letter after the digits: {' '.join(lows[start:i + 1])!r}"
        break
    if out and out[-1] == "-" and any(ch.isdigit() for ch in "".join(out)):
        return "", i, f"seal ends in a dash: {' '.join(lows[start:i + 1])!r}"  # what came after it was not read
    said = "".join(str(v) for v, s, e in spoken if e == s + 1 and v < 10)
    if len(said) >= 6 and len(said) % 2 == 0 and said[:len(said) // 2] == said[len(said) // 2:]:
        return "", i, f"seal digits said twice: {' '.join(lows[start:i])!r}"  # "siete siete tres uno siete siete ..."
    return "".join(out), i, ""


def _past_correction(toks: list[tuple[str, str]], lows: list[str], i: int) -> tuple[int, bool]:
    """From the word at i that ended an id: (the token after the corrections, alternatives and fillers said there, and
    whether a correction or an alternative was said). Past one, also 'es'/'the'...: 'no, perdón, el MX-48231'."""
    j, corrected = i, False
    while j < len(toks) and j - i < 12:
        if toks[j][0] == "comma":
            j += 1
            continue
        n = _phrase_at(lows, j, _CORRECTIONS) or (
            1 if lows[j] in _ALTERNATIVES and not _phrase_at(lows, j, _FILLER_PHRASES) else 0)  # "a" but not "a ver"
        if n:
            j, corrected = j + n, True
            continue
        n = _phrase_at(lows, j, _FILLER_PHRASES) or (1 if lows[j] in _SPEECH_FILLERS or (
            corrected and (lows[j] in _AFTER_CORRECTION or lows[j] in _SEAL_LEAD_IN)) else 0)
        if not n:
            break
        j += n
    if j == i:  # nothing of the kind: the token right after the word
        j += 1
        while j < len(toks) and toks[j][0] == "comma":
            j += 1
    return j, corrected


def _negation_at(toks: list[tuple[str, str]], lows: list[str], j: int) -> int:
    """Length of a negation at j ('no' but not the abbreviation 'No.', 'not', "isn't"), 0 if none."""
    if lows[j] in _NEGATIONS and not (lows[j] == "no" and _w(lows, j + 1) == "."):
        return 1
    if lows[j] in _NEGATED_CONTRACTIONS and j + 1 < len(toks) and toks[j + 1] == ("clitic", "t"):
        return 2
    return 0


def _negated_word(lows: list[str], j: int) -> tuple[str, int]:
    """What the negation ending before j negates: ('bad', end) for a broken/missing word ('no está roto', "isn't
    broken", 'no falta': the seal is fine), ('match', end) for a matching word ('no coincide', "doesn't match": it is
    not), ('', j) for anything else ('el sello no es MX-48213')."""
    e = j
    for _ in range(4):
        w = _w(lows, e)
        if w in _SEAL_BAD:
            return "bad", e + 1
        if w in _SEAL_MATCH:
            return "match", e + 1
        if w not in _NEGATION_GAP:
            break
        e += 1
    return "", j


def _negated(phrase: list[str]) -> str:
    return f"negated seal in transcript: {' '.join(phrase)!r}"


def _reported(phrase: list[str]) -> str:
    return f"seal reported broken, missing or not matching in transcript: {' '.join(phrase)!r}"


def _hedged(phrase: list[str]) -> str:
    return f"hedged seal in transcript: {' '.join(phrase)!r}"


def _negated_before(toks: list[tuple[str, str]], lows: list[str], j: int) -> bool:
    """Is the word at j negated by what comes right before it ('no falta', "doesn't match", 'no se ha abierto')?"""
    p = j - 1
    while p >= 0 and j - p <= 4 and lows[p] in _NEGATION_GAP:
        p -= 1
    return p >= 0 and (lows[p] in _NEGATIONS or (toks[p] == ("clitic", "t") and _w(lows, p - 1) in
                                                 _NEGATED_CONTRACTIONS))


def _seal_denied_before(toks: list[tuple[str, str]], lows: list[str], k: int) -> str:
    """'no traigo el sello', 'falta el sello', 'sin sello', "doesn't have a seal": why the seal said at k is denied."""
    for j in range(k - 1, max(k - 5, -1), -1):
        if toks[j][0] not in ("word", "clitic"):
            return ""
        if lows[j] in _SEAL_BAD or lows[j] in _SEAL_MATCH:  # "falta el sello"; "no falta el sello", "coincide el sello"
            return "" if (lows[j] in _SEAL_BAD) == _negated_before(toks, lows, j) else _reported(lows[j:k + 1])
        if lows[j] in _NEGATIONS or (toks[j] == ("clitic", "t") and _w(lows, j - 1) in _NEGATED_CONTRACTIONS):
            return _negated(lows[j - (lows[j] == "t"):k + 1])
        if lows[j] in ("sin", "without"):
            return _reported(lows[j:k + 1])
        if lows[j] not in _SEAL_BEFORE_SKIP:
            return ""
    return ""


def _hedged_before(toks: list[tuple[str, str]], lows: list[str], k: int) -> str:
    """'creo que el sello es ...', 'I think the seal is ...': a hedge up to 6 words before the keyword, in the same
    sentence."""
    a = k
    while a > 0 and k - a < 6 and toks[a - 1][0] != "stop":
        a -= 1
    return next((" ".join(lows[j:k + 1]) for j in range(a, k) if _phrase_at(lows, j, _HEDGES)), "")


def _source_before(toks: list[tuple[str, str]], lows: list[str], k: int) -> str:
    """'según la carta porte el sello es ...', 'the BOL has seal ...', 'dispatch told me the seal is ...': a word for
    the paperwork or another person up to 8 words before k, same sentence; past a comma only after 'según'/'according'.
    Not one followed by a number: 'factura 77812, sello ...' names an invoice."""
    a, comma = k, False
    while a > 0 and k - a < 8 and toks[a - 1][0] != "stop":
        a -= 1
    for j in range(k - 1, a - 1, -1):
        comma = comma or toks[j][0] == "comma"
        if lows[j] in _SOURCES and _w(lows, j + 1) != "yo" and not _number_at(lows, j + 1, dozens=False) and (
                not comma or any(lows[q] in ("según", "segun", "according") for q in range(a, j + 1))):
            return " ".join(lows[j:k + 1])
    return ""


def _seal_denied_after(toks: list[tuple[str, str]], lows: list[str], k: int, i: int) -> str:
    """'el sello MX-48213 viene roto', 'seal MX-48213 is missing', '... no coincide', '..., creo': why the seal said at
    k is denied or hedged, looking from i up to the end of the clause (a stop, a number, another seal) and 6 words at
    most. A bare negation counts before a comma only ('..., no traigo material peligroso' is about something else),
    and never one about the cargo ('... no hay material peligroso'). 'no está roto', 'no falta nada' say it is fine."""
    comma, words, negated = False, 0, ""
    j = i - 1
    while j > k and (toks[j][0] == "comma" or lows[j] in _SPEECH_FILLERS):  # the id reader steps over these
        comma, j = comma or toks[j][0] == "comma", j - 1
    j, denial = i, False
    while j < len(toks) and words < 8:
        kind, text = toks[j]
        if kind == "stop" or lows[j] in _SEAL_KEYWORDS or any(ch.isdigit() for ch in text) or (
                lows[j] not in ("un", "una", "a", "an", "one") and _number_at(lows, j, dozens=False)):
            break
        if kind in ("comma", "dash"):
            comma, j = comma or kind == "comma", j + 1
            continue
        if words == 0 and (n := _phrase_at(lows, j, _HEDGES_AFTER) or _phrase_at(lows, j, _APPROX_AFTER)):
            return _hedged(lows[k:j + n])
        if n := _negation_at(toks, lows, j):
            what, e = _negated_word(lows, j + n)
            if what == "match":
                return _reported(lows[k:e])
            denial = denial or not what
            if not what and not comma and not negated and not _HAZMAT & set(lows[j + n:j + n + 4]) and words < 6:
                negated = _negated(lows[k:j + n])
            j, words = max(e, j + n), words + 1
            continue
        if (lows[j] in _SEAL_BAD and words < 6) or (lows[j] in _SOURCES and denial):  # "..., no es el de la carta"
            return _reported(lows[k:j + 1])
        if lows[j] in _SEAL_PARTIAL and words < 6:
            return f"part of a seal said in transcript: {' '.join(lows[k:j + 1])!r}"
        if words < 6 and (lows[j] in _LETTER_WORDS or (kind == "word" and len(text) == 1 and text.isalpha() and (
                text.isupper() and text != "I" or _letters(toks, lows, j)))):
            return f"seal letter said after the digits: {' '.join(lows[k:j + 1])!r}"  # "with a B at the end"
        words, j = words + 1, j + 1
    return negated


def _past_lead_in(toks: list[tuple[str, str]], lows: list[str], i: int) -> int:
    """Past the lead-ins after a keyword ('es', 'el', 'número', 'No.', 'de seguridad'), commas and fillers."""
    while i < len(toks):
        if lows[i] in _SEAL_ABBREVIATIONS and toks[i][0] == "word" and _w(lows, i + 1) == ".":
            i += 2  # "No. MX-48213", "núm. MX-48213"
        elif _word_letter(lows, i):
            break  # "seal is el zee ...", "seal is em ex ...": the letters L, M
        elif toks[i][0] in ("comma", "clitic") or lows[i] in _SEAL_LEAD_IN or lows[i] in _SPEECH_FILLERS:
            i += 1
        elif (lows[i], _w(lows, i + 1)) == ("de", "seguridad"):
            i += 2
        elif n := _phrase_at(lows, i, _FILLER_PHRASES):
            i += n
        else:
            break
    return i


def _seal_start(toks: list[tuple[str, str]], lows: list[str], k: int, i: int) -> tuple[int, str]:
    """Where the id of the seal said at k starts: i (past the keyword and its lead-ins) when it is there. Else the id
    is never read past the words said before it ('el sello de la carta es MX-48213', 'the seal on trailer number four
    is intact', 'el sello creo que es ...'): (index of the id or -1, why it cannot be taken). The words are looked at up
    to the end of the sentence (a stop or another seal), so the reason says what they were."""
    if i < len(toks) and _seal_like(toks, lows, i):
        return i, ""
    j, why, hedge, comma = i, "", False, False
    while j < len(toks):
        kind, w = toks[j][0], lows[j]
        if kind == "stop" or w in _SEAL_KEYWORDS or w in _SEAL_PLURALS:
            break
        comma = comma or kind == "comma"
        if n := _negation_at(toks, lows, j):  # after a comma, or about the cargo, a negation is about something else
            what, e = _negated_word(lows, j + n)
            if what == "match" or (not what and not comma and not _HAZMAT & set(lows[j + n:j + n + 4])):
                why = why or (_reported(lows[k:e]) if what else _negated(lows[k:j + n]))
            j = max(e, j + n)
            continue
        if w in _SEAL_BAD:
            why, j = why or _reported(lows[k:j + 1]), j + 1
            continue
        if _phrase_at(lows, j, _CORRECTIONS):
            break  # "el sello es, perdón, trece": what follows may correct the count
        if n := _phrase_at(lows, j, _HEDGES):
            hedge, j = True, j + n
            continue
        if kind in ("word", "num") and _seal_like(toks, lows, j):
            said = lows[k:j + 1]
            return j, why or (_hedged(said) if hedge else f"words between the seal and its id: {' '.join(said)!r}")
        j += 1
    return -1, why


def _other_ids(toks: list[tuple[str, str]], lows: list[str], covered: set[int]) -> list[tuple[str, int, int, bool]]:
    """Ids said outside every seal mention: (as read, start, end). 'Sorry, MX-48231.' after the seal is one; 'la factura
    es 48213', 'placas ABC-123', 'trailer number is 53' are not (a noun says what they number), nor counts, weights
    ('pesa cuatro mil ochocientos') or small numbers. The flag: its number is one group said as words or digits
    ('one thousand', '48231', 'ce a guion doscientos sesenta y ocho'), which may correct the count as well."""
    out: list[tuple[str, int, int, bool]] = []
    j = 0
    while j < len(toks):
        kind, text = toks[j]
        e, value, one = j + 1, "", False
        if j in covered or kind not in ("word", "num") or lows[j] in ("un", "una", "a", "an"):
            pass
        elif any(ch.isdigit() for ch in text) and any(ch.isalpha() for ch in text):
            value = text
        elif not _number_at(lows, j, dozens=False) and _seal_like(toks, lows, j) and text != "I":
            # letters said before digits: "Perdón, ce a guion doscientos sesenta y ocho", "Perdón, SMX 278"
            read, end, _ = _seal_value(toks, lows, j)
            q = next((q for q in range(j, max(end, j + 1)) if _number_at(lows, q, dozens=False)), -1)
            if sum(ch.isdigit() for ch in read) >= 3:  # "bay cuatro", "o sea, trece" are not ids
                value, e = read, max(end, j + 1)
                one = q >= 0 and len(_number_run(lows, q)[0]) == 1 and not lows[q].isdecimal()
            elif read and not any(ch.isdigit() for ch in read):
                # letters, then a word, then digits ("ce a guion doscientos ..."): an id that does not read
                for q in range(end, min(end + 3, len(toks))):
                    if lows[q] not in ("un", "una", "a", "an") and _number_at(lows, q, dozens=False):
                        groups, e = _number_run(lows, q)
                        said = " ".join(lows[j:e])
                        digits = "".join(str(v) for v, _, _ in groups)
                        if not _is_unit(_w(lows, e)) and len(digits) >= 3:
                            value, one = said, len(groups) == 1
                        break
        elif _number_at(lows, j, dozens=False):
            groups, e = _number_run(lows, j)
            after = _w(lows, e)
            if len(groups) > 1 and (_is_unit(after) or after in _WEIGHT_UNITS):  # "... cuatro siete 15 tarimas"
                groups, after = groups[:-1], "|"
                e = groups[-1][2]
            if groups and not (_is_unit(after) or after in _WEIGHT_UNITS or (
                    _is_filler(after) and _is_unit(_w(lows, e + 1)))) and (
                    (len(groups) >= 3 and not any(_compound(g) for g in groups))
                    or max(v for v, _, _ in groups) >= 1000):
                value = "".join(lows[s] if lows[s].isdecimal() else str(v) for v, s, _ in groups)
                one = len(groups) == 1
            e = max(e, j + 1)
        if value and not _named(lows, j) and not _WEIGHT_WORDS & set(lows[max(j - 3, 0):j]):
            out.append((value, j, e, one))
        j = e
    return out


def _corrected_number(toks: list[tuple[str, str]], lows: list[str], j: int, covered: set[int],
                      mention_ends: list[int]) -> str:
    """A number said right after a correction, outside the seal mentions, with no unit, when the seal is the last thing
    said before it: 'The seal is 203. Sorry, 230.' may correct the seal (or the count). Its digits, or ''."""
    if j in covered or lows[j] in ("un", "una", "a", "an", "one") or not _number_at(lows, j, dozens=False):
        return ""
    groups, e = _number_run(lows, j)
    after = _w(lows, e)
    if len(groups) != 1 or _is_unit(after) or after in _WEIGHT_UNITS or _named(lows, j) or (
            _w(lows, j - 1) in _NOT_COUNT_BEFORE or (_is_filler(after) and len(after) > 2)):
        return ""
    p = j - 1
    while p >= 0 and (toks[p][0] == "comma" or lows[p] in _SPEECH_FILLERS or lows[p] in _AFTER_CORRECTION):
        p -= 1
    n = _phrase_ending_at(lows, p + 1, _CORRECTIONS) if p >= 0 else 0
    last = max((m for m in mention_ends if m <= p + 1 - n), default=-1)
    if n and p not in covered and last >= 0 and not any(_is_unit(w) for w in lows[last:p + 1 - n]):
        return lows[j] if lows[j].isdecimal() else str(groups[0][0])
    return ""


def _named(ws: list[str], s: int) -> bool:
    """Is the number at s said after a noun that says what it numbers ('andén cuatro', 'trailer number is 53')?"""
    for p in range(s - 1, max(s - 4, -1), -1):
        if ws[p] in _ID_NOUNS:
            return True
        if ws[p] not in _ID_LEAD and not any(ch.isdigit() for ch in ws[p]):
            return False
    return False


def _agrees(other: str, value: str) -> bool:
    """An id said again: the same letters and digits, or the same digits with no letters ('MX-48213 ... 48213')."""
    o, v = re.sub(r"[\W_]", "", other).upper(), re.sub(r"[\W_]", "", value).upper()
    return o == v or (o.isdecimal() and o == re.sub(r"\D", "", v))


def _seals(toks: list[tuple[str, str]], lows: list[str]) -> tuple[str | None, str, set[int], set[int], str]:
    """Every seal mention: (seal or None, why None when one was said, token indices the mentions cover, indices where
    an id stopped reading one way, why no count may be taken: one was said inside a seal's digits)."""
    values: list[str] = []
    whys: list[str] = []
    unread: list[str] = []  # mentions with no id: "el sello es de plástico", "the seal is not broken"
    covered: set[int] = set()
    stuck: set[int] = set()  # where an id stopped reading one way: a count right after it does not either
    ends: list[int] = []  # where each mention ended
    count_why = ""
    k = 0
    while k < len(toks):
        if lows[k] not in _SEAL_KEYWORDS and lows[k] not in _SEAL_PLURALS:
            k += 1
            continue
        before = len(values), len(whys)
        if denied := _seal_denied_before(toks, lows, k):
            whys.append(denied)
        elif lows[k] in _SEAL_PLURALS:  # "los sellos son MX-48213 y MX-48214": their ids are read only to set aside
            whys.append(f"several seals said in transcript: {toks[k][1]!r}")
        elif hedged := _hedged_before(toks, lows, k):
            whys.append(_hedged(hedged.split()))
        elif source := _source_before(toks, lows, k):
            whys.append(f"seal said from the paperwork or someone else: {source!r}")
        i = _past_lead_in(toks, lows, k + 1)
        if i < len(toks) and (n := _negation_at(toks, lows, i)):  # "el sello no es MX-48213": read only to set aside
            what, e = _negated_word(lows, i + n)
            j = _past_lead_in(toks, lows, max(e, i + n))
            if what != "bad" and len(whys) == before[1]:  # "the seal is not broken" says it is fine
                whys.append(_reported(lows[k:e]) if what else _negated(lows[k:j]))
            i = j
        covered.update(range(k, i))
        start, why = _seal_start(toks, lows, k, i)
        if why and len(whys) == before[1]:
            whys.append(why)
        i = start if start >= 0 else i
        while start >= 0:
            value, i, why = _seal_value(toks, lows, i)
            if why:
                whys.append(why)
                stuck.add(i)
                if why.startswith(("seal digits run into a count", "seal letters run into a count")):
                    count_why = why.split(": ", 1)[-1]
            if any(ch.isdigit() for ch in value):
                values.append(value)
            if i >= len(toks) or toks[i][0] != "word" or not value:
                break
            if not any(ch.isdigit() for ch in value):  # "ce a cuatro ...": letters read, then a word; never drop them
                if (nxt := _past_correction(toks, lows, i)[0]) < len(toks) and _seal_like(toks, lows, nxt):
                    whys.append(f"unclear seal letters: {' '.join(lows[start:nxt + 1])!r}")
                    i = max(_seal_value(toks, lows, nxt)[1], nxt + 1)  # set its digits aside
                break
            # The id stopped at a word. Another id after a correction ("..., no, perdón, MX-48231", "MX-48213 y
            # MX-48214", "..., me equivoqué, es 48231") is a second value; a small number there ("..., no, trece") may
            # correct the count as well as the seal. Any other word followed by an id: where does the id end?
            nxt, corrected = _past_correction(toks, lows, i)
            if nxt >= len(toks) or lows[i] in _ID_NOUNS:
                break
            if corrected and _id_shaped(toks, lows, nxt) and not _one_number(toks, lows, nxt):
                i = nxt
                continue
            near = range(nxt, min(nxt + 3, len(toks)))  # "..., perdón, guión B", "perdón es el ab quinientos ocho"
            if corrected and any(_seal_like(toks, lows, q) or toks[q][0] == "dash" or lows[q] in _SEAL_DASH
                                 for q in near if toks[q][0] != "stop"):
                whys.append(f"correction right after the seal: {' '.join(lows[i:nxt + 1])!r}")
            elif _id_shaped(toks, lows, nxt):
                if _agrees(_seal_value(toks, lows, nxt)[0] or "?", value):  # "MX 48213, repito, MX 48213"
                    i = nxt
                    continue
                whys.append(f"seal interrupted by {toks[i][1]!r}")
            break
        if len(whys) == before[1] and (denied := _seal_denied_after(toks, lows, k, i)):
            whys.append(denied)
        if (len(values), len(whys)) == before:  # "el sello es de plástico", and a keyword said last: "... y el sello"
            unread.append(" ".join(lows[k:i + 1]))
        covered.update(range(start if start >= 0 else k, i))
        ends.append(i)
        k = max(i, k + 1)
    for j, w in enumerate(lows):  # "la caja viene abierta", "the trailer door was opened": the seal is not intact
        if w in _TRAILER and _w(lows, j - 1) in ("la", "el", "the") and j not in covered:
            bad = next((q for q in range(j + 1, min(j + 4, len(lows))) if lows[q] in _SEAL_BAD | {"open"}), -1)
            if bad >= 0 and not any(lows[q] in _NEGATIONS for q in range(j + 1, bad)):
                whys.append(_reported(lows[j - 1:bad + 1]))
    for j, w in enumerate(lows):  # "it's not sealed", "sin sellar", "came unsealed"; "sealed" alone says nothing
        if w in _SEALING and j not in covered:
            p = max(j - 3, 0)
            if w == "unsealed" or any(lows[q] in _NEGATIONS or lows[q] in ("sin", "without")
                                      or toks[q] == ("clitic", "t") for q in range(p, j)):
                whys.append(_reported(lows[p:j + 1]))
            else:
                unread.append(w)
    others = _other_ids(toks, lows, covered)
    others += [(v, j, j + 1, True) for j in range(len(toks)) if (v := _corrected_number(toks, lows, j, covered, ends))
               and not any(s <= j < e for _, s, e, _ in others)]
    for value, s, e, one in others:
        # An id said again, or one written with letters or said digit by digit, is the seal's business; one number
        # that disagrees stays for the count to see too: "..., no, one thousand" may correct either.
        if any(_agrees(value, v) for v in values) or not one:
            covered.update(range(s, e))
    if whys:
        return None, "; ".join(whys), covered, stuck, count_why
    distinct = sorted({re.sub(r"[\W_]", "", v).upper() for v in values})
    disagree = sorted({re.sub(r"[\W_]", "", o).upper() for o, _, _, _ in others
                       if not any(_agrees(o, v) for v in values)})
    if values and (len(distinct) > 1 or disagree):
        return None, f"seals disagree in transcript: {sorted(set(distinct + disagree))}", covered, stuck, count_why
    if not values and unread:  # never silent: a seal was said and no id came out of it
        return None, "seal said but no id read: " + ", ".join(repr(u) for u in unread), covered, stuck, count_why
    if not values and others:
        said = ", ".join(repr(o) for o, _, _, _ in others)
        return None, f"id said with no seal keyword: {said}", covered, stuck, count_why
    return (values[0] if values else None), "", covered, stuck, count_why


# --- count -------------------------------------------------------------------------------------


def _count(ws: list[str], stuck: frozenset[int] | set[int] = frozenset()) -> tuple[int | None, str]:
    """The one count said in ws (seal mentions masked as '|'). stuck: where a seal id stopped reading one way; a count
    said right after one ("... one thousand five hundred eighty one hundred and nine pallets") does not either."""
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

    def correction_before(s: int) -> str:  # "doce, no, eh, trece", "no a ver trece", "sorry, uh, it's thirteen"
        p = s
        for _ in range(6):
            n = _phrase_ending_at(ws, p, _CORRECTIONS)
            if n:
                return " ".join(ws[p - n:p])
            n = _phrase_ending_at(ws, p, _FILLER_PHRASES) or (1 if _w(ws, p - 1) in (",", ".", "-") or (
                _w(ws, p - 1) in _SPEECH_FILLERS | _AFTER_CORRECTION) else 0)
            if not n or p - n < 0:
                return ""
            p -= n
        return ""

    counts: list[tuple[int, int, int | None]] = []  # (start, unit end, value)
    other: dict[int, str] = {}  # value -> why it competes with the count
    unit_ends: set[int] = set()
    bare: list[tuple[int, int, int]] = []
    for s, e, v in phrases:
        if _w(ws, e) in _WEIGHT_UNITS and (marker := correction_before(s)):
            # "21 cajas ... no veintidós mil ochocientos kilos": a corrected count (22) and 1800 kg, or 22800 kg?
            head = max((x for x in (_es_below_1000(ws, s), _en_below_1000(ws, s)) if x), key=lambda x: x[1],
                       default=None)
            if head and head[1] < e and _w(ws, head[1]) in ("mil", "thousand"):
                other.setdefault(head[0], f"self-correction: {marker!r}")
        if _w(ws, s - 1) in _ID_NOUNS or _w(ws, e) in _WEIGHT_UNITS:
            continue
        if _unit(ws, e):
            u = e + 1
        elif _is_filler(_w(ws, e)) and _unit(ws, e + 1):
            u = e + 2
        else:
            if v is not None:
                bare.append((s, e, v))
            continue
        counts.append((s, u, None if s in ends else v))  # a group said right before it: 'veinte cuatro tarimas'
        unit_ends.add(u)
    if not counts:
        return None, "no count found in transcript"

    named_ends: set[int] = set()
    for s, e, v in bare:
        # Any number said with no unit, anywhere in the note, is a value for the count unless it numbers or weighs
        # something else ("andén cuatro", "factura nueve nueve ocho", "trailer number is 53", "a las tres", "pesa
        # cuatro mil ochocientos") or counts something else ("dos horas esperando"): "nope, uh, thirteen total"
        # disagrees with "twelve pallets" whether or not a correction word was recognized.
        if _named(ws, s) or s in named_ends:
            named_ends.add(e)
            continue
        if _w(ws, s - 1) in _NOT_COUNT_BEFORE or _WEIGHT_WORDS & set(ws[max(s - 3, 0):s]):
            continue
        if _w(ws, e) in _MORE and not _phrase_at(ws, e, _APPROX_AFTER) and not (
                _is_filler(_w(ws, e + 1)) and len(_w(ws, e + 1)) > 2 and _w(ws, e + 1) not in _ADDED | _SEAL_BAD):
            # "y una más": 13, or 1? ("one more thing" says nothing of the count)
            return None, f"unclear count in transcript: {' '.join(ws[s:e + 1])!r}"
        nxt = _w(ws, e)
        # "horas", "saludo"; "|" is a seal mention: "y un sello MX-48213" counts the seal, not the load
        article = e == s + 1 and ws[s] in _ARTICLES
        noun = (not _phrase_at(ws, e, _COUNT_AFTER) and _is_filler(nxt) and len(nxt) > 2) or (nxt == "|" and article)
        if article and nxt == "total":
            continue  # "un total de doce tarimas"
        # A correction before a number makes it compete whatever follows it ("no, trece en total", "perdón, trece
        # completas"). Accepted cost: "perdón, tres horas tarde" competes too (count None, the detail says why). Not
        # an article before a noun: "bueno, un saludo" ("un"/"una"/"one" counts there only before a count noun).
        if marker := correction_before(s):
            if not (noun and article):
                other.setdefault(v, f"self-correction: {marker!r}")
            continue
        if noun:
            continue  # "dos horas esperando", "two hours late": it counts something else
        q = e
        while _w(ws, q) in (",", ".", "-"):
            q += 1
        if after := _phrase_at(ws, q, _CORRECTIONS):
            other.setdefault(v, f"self-correction: {' '.join(ws[q:q + after])!r}")
        elif (_w(ws, e) in _ALTERNATIVES and e + 1 in starts) or (
                _w(ws, s - 1) in _ALTERNATIVES and (s - 1 in ends or s - 1 in unit_ends)):
            other.setdefault(v, f"alternative: {_w(ws, e) if e + 1 in starts else _w(ws, s - 1)!r}")
        else:
            other.setdefault(v, "")

    for s, u, v in counts:
        said = " ".join(ws[s:u])
        a = s
        while a > 0 and s - a < 6 and ws[a - 1] not in (".", ",", "|"):
            a -= 1
        if any(w in _SOURCES and not (w in ("según", "segun") and _w(ws, a + n + 1) == "yo")
               for n, w in enumerate(ws[a:s])):  # "según la factura son doce tarimas", "the bol says fourteen ..."
            return None, f"count said from the paperwork or someone else: {' '.join(ws[a:u])!r}"
        thousands = next((n for n in range(s, u) if ws[n] in ("mil", "thousand")), -1)
        if thousands >= 0 and (_WEIGHT_WORDS & set(ws[max(s - 3, 0):s]) or (
                v is not None and v >= 10000 and _number_at(ws, thousands + 1, dozens=False))):
            # "pesa veinte mil doce tarimas", "forty thousand twelve pallets": a weight and a count, or one number?
            return None, f"count runs into a weight: {' '.join(ws[max(s - 3, 0):u])!r}"
        if any(p < s <= p + 4 and all(w in _NUMBER_WORDS or w in ("y", "and") for w in ws[p:s]) for p in stuck):
            return None, f"count runs into the seal's digits: {' '.join(ws[min(stuck, key=lambda p: abs(s - p)):u])!r}"
        q = u + 1 if _w(ws, u) == "," else u
        if _phrase_ending_at(ws, s, _APPROX_BEFORE) or _phrase_at(ws, q, _APPROX_AFTER):
            return None, f"approximate count in transcript: {said!r}"
        k = s
        while k > 0 and s - k < 5 and _w(ws, k - 1) not in (".", "|", "-"):  # a hedge up to 5 words before it
            if n := _phrase_ending_at(ws, k, _HEDGES):
                return None, f"hedged count in transcript: {' '.join(ws[k - n:u])!r}"
            k -= 1
        if n := _phrase_at(ws, q, _HEDGES_AFTER):  # "doce tarimas, creo", and "... boxes i think hello ..." too
            return None, f"hedged count in transcript: {' '.join(ws[s:q + n])!r}"
        for k in range(s - 1, max(s - 5, -1), -1):
            if ws[k] in (".", ",", "-", "|"):
                break
            if ws[k] in _NEGATIONS or (ws[k] == "t" and _w(ws, k - 1) in _NEGATED_CONTRACTIONS):
                return None, f"negated count in transcript: {' '.join(ws[k - (ws[k] == 't'):u])!r}"
        # said after it, more than the count: "doce tarimas más una suelta", "twelve pallets and a box", "y media"
        end = next((n for n in range(u, min(u + 6, len(ws))) if ws[n] in (".", "|")), min(u + 6, len(ws)))
        for n in range(u, end):
            if _phrase_at(ws, n, _CORRECTIONS) or _phrase_at(ws, n, _APPROX_AFTER):
                break  # "doce tarimas, más bien trece", "más o menos": said for what they are
            seal_next = ws[n] in _SEAL_BAD and (_w(ws, n + 1) == "|" or (
                _w(ws, n + 1) in ("el", "la", "the", "a", "un", "una") and _w(ws, n + 2) == "|"))
            if (ws[n] in _ADDED and _w(ws, n - 1) not in _NEGATIONS and not seal_next) or (
                    ws[n] in ("a", "an", "un", "una") and (
                    _unit(ws, n + 1) or (_is_filler(_w(ws, n + 1)) and _unit(ws, n + 2)))):
                # not "no falta nada", nor "falta el sello": that is the seal's
                return None, f"more than the count said in transcript: {' '.join(ws[s:n + 1])!r}"
        if v is None:
            return None, f"unclear count in transcript: {said!r}"

    values = {v for _, _, v in counts if v is not None} | set(other)
    if len(values) == 1:
        return values.pop(), ""
    notes = sorted({n for n in other.values() if n})
    why = f"counts disagree in transcript: {sorted(values)}"
    return None, why + (f" ({'; '.join(notes)})" if notes else "")


def _parse_voice(transcript: str) -> tuple[int | None, str, str | None, str]:
    toks = _tokens(transcript)
    lows = [_LOW_PUNCT.get(k, t.lower()) for k, t in toks]
    seal, seal_why, covered, stuck, inside = _seals(toks, lows)
    count, why = _count(["|" if k in covered else w for k, w in enumerate(lows)], stuck)
    if inside:  # "sello cuatro ocho dos uno tres dos tarimas": 2 pallets, or 482132 and no count?
        count, why = None, f"count said inside the seal's digits: {inside}"
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
