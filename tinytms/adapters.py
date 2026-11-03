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
# The transcript is cut into tokens (words, digit groups, clock times, punctuation). Seal mentions are read first
# ("sello", "seal", ...) and their tokens are set aside; then every number phrase is classified: a count (number + unit
# noun), a weight (+ kilos), an id ("tráiler cuatro"), a time ("a las tres", "1:45"), or a bare number. A count comes
# out only if exactly one value is said for it anywhere in the note: any bare number competes unless it numbers,
# weighs or times something else, and "número" alone names the field being corrected ("Perdón, el número es trece").
# Self-corrections, alternatives, approximations, a hedge anywhere in its sentence ("Creo que lo que traigo son doce
# tarimas"), negations, more said after it ("y una suelta", "one short"), a count from someone else or a document
# ("me dijeron ...", "The sheet has ...") or taken back ("Mentira.", "no lo apuntes") and unclear number words give
# None and a detail that says why. So does a seal that does not read one way: corrected or said two ways anywhere,
# hedged in its sentence, negated, taken back, reported broken, missing or not matching anywhere after it in the
# note ("... Está roto.", "..., no está.") or before it in its sentence ("they broke the seal ..."), from someone else
# or a document, said after any word but lead-ins and fillers ("el sello de la carta es ..."; "el sello pues es ..."
# reads), spelled with a first letter that is also a word ("ese eme equis ...", "el sello es eme equis ..."), run into
# what follows it, or said in the plural. Each scan for a hedge, a source or a negation ends at the end of its
# sentence, never after a fixed number of words (ORDER Nube-10). House rule (ORDER Nube-9): the witness returns the
# right value, or None with a non-empty reason; it never picks one of two readings.

# "payets", "pajets", "payettes": how Muse Voice Transcribe wrote "pallets" said in Spanish, three runs over the
# same audio (real transcripts, tests/data/real-voice-es-*).
_UNITS = (
    r"pallets?|palets?|pal[eé]s|payets?|pajets?|payettes?|paletas?|tarimas?|skids?|cajas?|boxes|box|bultos?|"
    r"piezas?|pieces?|drums?|cartons?|cases?|rollos?|tambos?|bobinas?|crates?|coils?|garraf[oó]n|garrafones|cubetas?|"
    r"bundles?|sacos?|bags?|costales?|rolls?|reels?|totes?|barrels?|barriles?|bins?|racks?|pacas?|bales?|huacales?|"
    r"bidones?|sacks?|carretes?|paquetes?|packages?|tires?"
)
_UNIT = re.compile(_UNITS)
_WEIGHT_UNITS = frozenset(
    "kilos kilo kg kgs kilogramos kilogramo kilograms kilogram toneladas tonelada ton tons tonnes libras libra lb lbs "
    "pounds pound".split()
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
    "orden folio número numero placa placas turno línea linea carril truck dock door bay unit route trip invoice "
    "order number plate lane gate po operador operator licencia license teléfono telefono phone celular cell cel bol "
    "pro ref cortina load compra slot spot plates tracto tractor chassis chasis container contenedor highway hwy "
    "interstate carretera autopista freeway exit salida".split()
)
# "number"/"número" names what it numbers only after one of these ("PO number 55123", "trailer number 53"; "número de
# factura 77812" is named by "factura"). Alone it names the field being corrected, so its value competes:
# "Sorry, the number is MX-48231", "Perdón, el número es trece" (ORDER Nube-10 item 3).
_NUMBER_NOUNS = frozenset("number número numero".split())
_NUMBERED = frozenset(
    "tráiler trailer camión camion unidad andén anden puerta rampa bahía bahia muelle caja ruta viaje factura pedido "
    "orden folio placa placas turno línea linea carril truck dock door bay unit route trip invoice order plate lane "
    "gate po operador operator licencia license teléfono telefono phone celular cell cel bol pro ref reference "
    "referencia tracking confirmation confirmación confirmacion booking load carga pickup delivery entrega "
    "appointment cita cuenta account pedimento guía guia control employee empleado badge gafete móvil movil mobile "
    "tractor chassis chasis container contenedor".split()
)
_ID_ARTICLES = frozenset("el la los las the my mi su your tu his her its our nuestro nuestra an un una".split())
_CORRECTIONS = tuple(tuple(p.split()) for p in (
    "no", "perdón", "perdon", "digo", "corrijo", "mejor dicho", "más bien", "mas bien", "o sea", "bueno",
    "me equivoqué", "me equivoque", "quise decir", "sorry", "wait", "i mean", "actually", "or rather", "rather",
    "make that", "correction", "scratch that", "make it", "mejor", "nope", "oops", "whoops", "my bad", "strike that",
    "let me recount", "error", "no es cierto", "en realidad", "aguanta", "conté mal", "conte mal", "déjame contar",
    "dejame contar", "le corrijo", "me corrijo",
))
_APPROX_BEFORE = tuple(tuple(p.split()) for p in (
    "como", "unas", "unos", "casi", "aproximadamente", "aprox", "alrededor de", "cerca de", "más de", "mas de",
    "menos de", "más o menos", "mas o menos", "al menos", "por lo menos", "hasta", "por ahí de", "about", "around",
    "close to", "por ahi de",
    "roughly", "approximately", "approx", "nearly", "give or take", "almost", "some", "like", "over", "under",
    "more than", "less than", "fewer than", "at least", "up to",
))
_APPROX_AFTER = tuple(tuple(p.split()) for p in (
    "más o menos", "mas o menos", "o así", "o asi", "aproximadamente", "o algo así", "y pico", "or so", "roughly",
    "approximately", "give or take", "more or less",
))
# The driver is not sure. Said anywhere in the sentence of a value, before or after it, a hedge covers it (ORDER
# Nube-10 item 5): "creo que son doce", "doce tarimas, creo", "Creo que lo que traigo son doce tarimas", "No sé si el
# sello es ...", "Ponle que doce tarimas", "Seal MX-48213, if I'm not wrong", "twelve-ish pallets".
_HEDGES = tuple(tuple(p.split()) for p in (
    "creo", "supongo", "pienso", "parece", "quizá", "quizás", "quiza", "quizas", "tal vez", "talvez", "a lo mejor",
    "probablemente", "posiblemente", "calculo", "puede que", "según yo", "segun yo", "supuestamente", "han de ser",
    "deben ser", "deben de ser", "debe ser", "no estoy seguro", "no estoy segura", "si no me equivoco",
    "si mal no recuerdo", "si no mal recuerdo", "no me acuerdo", "no lo veo", "no se si", "no sé", "que yo sepa",
    "hasta donde sé", "hasta donde se", "casi seguro", "casi segura", "igual y", "se supone", "ponle", "diría",
    "diria", "me late", "digamos", "maybe", "perhaps", "probably", "possibly", "think", "guess", "believe", "might",
    "may be", "pretty sure", "fairly sure", "want to say", "not sure", "could be", "supposedly", "supposed",
    "should be", "guessing", "if i remember", "if i m not mistaken", "if i read", "if i counted", "or something",
    "don t quote me", "can t really read", "can t read", "i d say", "i would say", "reckon", "suppose",
    "as far as i know", "almost sure", "almost certain", "if i m not wrong", "if i am not wrong", "unless i m wrong",
    "don t know", "ish", "if i m reading", "not a hundred percent", "not one hundred percent", "not 100",
    "no estoy muy seguro", "no estoy muy segura", "no estoy tan seguro", "no estoy tan segura", "si las conté",
    "si los conté", "si las conte", "si los conte", "si no conté mal", "not positive", "not certain", "no se bien",
    "más o menos", "mas o menos", "o así", "o asi", "o algo así", "o algo asi", "y pico", "or so", "give or take",
    "more or less", "aproximadamente", "approximately", "roughly", "approx", "aprox", "something like",
    "algo así como", "ballpark", "estimate", "estimated", "estimo", "a ojo", "al tanteo", "a lo mucho",
    "cuando mucho", "at most", "at the most", "si es que", "si lo leí", "si lo lei", "aproximado", "aproximada",
))
_NEGATIONS = frozenset({"no", "ni", "nunca", "tampoco", "not", "never", "nor"})
_NEGATED_CONTRACTIONS = frozenset({"don", "doesn", "didn", "isn", "aren", "wasn", "weren", "haven", "hasn", "won",
                                   "can", "couldn", "wouldn", "shouldn", "ain"})
_COUNT_VERBS = frozenset(
    "traigo trae traemos llevo lleva llevamos tengo tenemos son eran serían serian vienen van cargo cargué cargue hay "
    "entrego bring have got carrying carry are".split()
)
_ALTERNATIVES = frozenset({"o", "u", "ó", "or", "y", "e", "and", "a", "to", "ni", "nor"})
_AFTER_CORRECTION = frozenset(  # "no, son trece", "sorry, it's thirteen", "make that thirteen"
    "son eran es serían serian fueron traigo tengo llevo are is it s that i have got we were was".split()
)
_SPEECH_FILLERS = frozenset({"eh", "ehh", "em", "emm", "mm", "mmm", "um", "uh", "uhm", "este", "hmm", "hm", "mhm", "ah",
                             "ahh", "er", "erm", "pos", "pues", "espérame", "esperame", "espera", "oye", "mmh", "mmhm",
                             "umm", "uhh", "ehm"})
_FILLER_PHRASES = tuple(tuple(p.split()) for p in ("a ver", "déjame ver", "dejame ver", "let me see", "let s see"))
_NOT_FILLER = frozenset(
    "con sin por para al hasta desde entre sobre cada with without in on for to from at by per each "
    "el la los las lo the and pero but si if que".split()
) | _NEGATIONS | _COUNT_VERBS | _ID_NOUNS | _WEIGHT_UNITS | _NUMBER_WORDS | {w for p in _CORRECTIONS for w in p}
# A number said with no unit is a count value when one of these follows it: "nope, uh, thirteen total".
_COUNT_AFTER = tuple(tuple(p.split()) for p in (  # ("en total", "de ellas" already count: "en", "de" name nothing)
    "total", "completas", "completos", "altogether", "all together", "exactas", "exactos", "justas", "justos", "now",
    "ahora",
))
_MORE = frozenset({"más", "mas", "more", "extra", "menos", "less", "fewer"})  # "y una más": 13, or 1? no count
_ARTICLES = frozenset({"un", "una", "one"})  # "bueno, un saludo": an article, a count only before a count noun
# After a correction, "un"/"una"/"one" is an article only before one of these ("bueno, un saludo", "sorry, one
# question"); before any other word it competes as a count ("perdón, un par más", "sorry, one short").
_ARTICLE_NOUNS = frozenset("saludo saludos abrazo favor momento pregunta question moment second segundo minute minuto "
                           "cosa thing".split())
# "one" is a pronoun after these: "have a good one", "the last one", "this one" (ORDER Nube-10 item 7).
_PRONOUN_BEFORE = frozenset("good great nice safe long last next other same right first this that which every "
                            "another quick easy short small big little busy slow fast tough rough quiet smooth "
                            "blessed".split())
# Right after a number these change the count, never name what it counts: "one short", "una faltante", "un par más".
_COUNT_CHANGE = frozenset("short faltante faltantes missing falta faltan suelta sueltas suelto sueltos loose adicional "
                          "adicionales additional sobrante sobrantes par pair couple".split())
# Adverbs ("-ly", "-mente" and these) after a bare number never hide it: "Thirteen honestly" competes.
_ADVERBS = frozenset("tops flat even straight nomás nomas ahorita again also too though".split())
# Said after a count, any of these adds to it or makes it a part: "doce tarimas más una suelta", "and a half".
_ADDED = frozenset("más mas plus además aparte another otra otras otro otros extra media half cacho change partial "
                   "parcial incompleta incompleto incompletas incompletos suelta sueltas suelto sueltos loose falta "
                   "faltan faltó falto faltaba missing short".split())  # "me faltó cargar una": 11 or 12?
# Second-hand values (ORDER Nube-10 item 4): before a value in its sentence, any of these makes it someone else's.
# Verbs of telling or giving, with any subject: "me dijeron doce tarimas", "the office sent me ...", "dispatch says".
_TELLING = frozenset(
    "dice dicen dijo dijeron decía decia decían decian diga digan dicho says say said told tell tells telling saying "
    "gave give gives giving sent send sends texted emailed informó informaron avisó avisaron reportó reportaron "
    "comentó comentaron contó contaron contaba reported mentioned entregó entregaron dictó dictaron leyó "
    "leyeron".split()
)
# Giving verbs that are also everyday words ("lo que pasó", "da igual"): giving only right after an object pronoun:
# "me pasó el sello", "nos mandaron doce tarimas", "le dieron el número".
_GIVING = frozenset("pasó pasaron pasa pasan dio dió dieron da dan mandó mandaron manda mandan".split())
_GIVEN_TO = frozenset("me nos le les te".split())
_GIVEN_WHAT = frozenset("lo la los las".split())  # "me lo pasó", "me las contó el muchacho"
_SCOPING = frozenset("según segun according per".split())  # "per the docs, the seal is ...", "según la carta ..."
# Documents and devices: "The sheet has twelve pallets", "En la remisión viene el sello ...", "the app shows ...".
_DOCS = frozenset(
    "docs doc documents documento documentos paperwork papeles papel papers paper sheet hoja carta guía guia remisión "
    "remision sistema system app aplicación aplicacion bol bill manifest manifiesto factura ticket pantalla screen "
    "tablet scanner portal".split()
)
# Also cargo or a name ("toallas de papel", "paper plates", "it's Bill with ..."): a document only right after one of
# _DOC_DETERMINERS ("el papel dice", "per the bill", "según la hoja").
_ARTICLE_DOCS = frozenset("papel papeles paper papers bill hoja carta sistema system ticket guía guia pantalla "
                          "screen".split())
_DOC_DETERMINERS = frozenset("el la los las the my mi su this that ese esa este esta your tu".split())
# People and places: a source unless the driver is the one acting (_DRIVER_ACTING) and nobody in the sentence counts,
# seals, tells or gives (_PARTY_ACTS, _TELLING): "El cliente contó doce tarimas", "En la planta me contaron ...".
_PARTIES = frozenset(
    "dispatch despacho despachador dispatcher shipper remitente cliente customer client montacarguista planta plant "
    "compañero compañera partner guardia guard caseta oficina office supervisor jefe boss gerente manager checker "
    "receiving receiver broker warehouse almacén almacen bodega forklift montacargas lumper lumpers loader "
    "loaders".split()
)
_SOURCE_PHRASES = tuple(tuple(p.split()) for p in ("otro chofer", "other driver", "otro operador"))  # "el otro chofer"
_PARTY_ACTS = frozenset("count counts sealed selló sellaron checked revisó reviso revisaron cargaron cargó "
                        "loaded".split())
# Said with no first person before them, these English verbs are someone else counting or sealing, before or after
# the value: "The forklift driver counted nineteen pallets", "twelve pallets, they counted them" (not "I counted").
_ACTS_BY_OTHERS = frozenset({"counted"})
_FIRST_PERSON = frozenset("i we".split())
_AUXILIARIES = frozenset("have haven had hadn t not did didn ve d just already also never".split())
# The exact ways a driver acts on a place or party (past an article): "Vengo de la planta con doce tarimas", "Traigo
# doce tarimas del cliente", "I came straight from the shipper", "Ya pasé la caseta". Anything else keeps it a source.
_DRIVER_ACTING = frozenset(
    "de del desde from for para a al to at en in con with hacia toward of pasé pase pasamos crucé cruce cruzamos "
    "passed crossed left dejé deje".split()
)
# Said anywhere, the count was not checked: "no las conté", "I didn't count them".
_COUNT_DOUBT = tuple(tuple(p.split()) for p in (
    "no las conté", "no los conté", "no las conte", "no los conte", "didn t count", "did not count", "didn t check",
    "did not check", "haven t counted", "no vi la carga", "not looked", "haven t looked",
))
# "a las tres", "at three", "cada una", "the one on the BOL": the number after these is not a count
_NOT_COUNT_BEFORE = frozenset({"las", "la", "at", "cada", "each", "the", "minus", "negative", "since", "desde"})
# Clock times said in words: "a quarter to six", "half past five", "a las ocho y diez" (the "y diez" after a time).
_TIME_BEFORE = tuple(tuple(p.split()) for p in ("quarter to", "quarter till", "past", "cuarto para"))
# English phrases with "once" (Spanish 11): "no wind for once", "all at once".
_ONCE = tuple(tuple(p.split()) for p in ("for once", "just once", "only once", "once again", "once in a while"))
_WEIGHT_WORDS = frozenset("peso pesa pesan pesamos weight weighs weigh weighing".split())
# What may stand between a weight word and its number: "peso bruto 4800", "weighs about 4,800", "pesa, como, cuatro
# mil".
_WEIGHT_GAP = frozenset(
    "es is de of del bruto bruta gross neto neta net total son are was era como about around unos unas "
    "aproximadamente approximately casi almost roughly right just justo nearly ,".split()
)
# What may stand between an id noun and its number: "trailer number is 53", "la factura es 48213", "en el andén 4".
# "No." is a lead only as the abbreviation before digits ("andén No. 4"): a correction "no" never hides a count behind a
# place noun ("en la caja, no trece", "andén no cuatro"), and neither does a comma (ORDER Nube-10 item 1a).
_ID_LEAD = frozenset("es is el la los las the de del of nº en in on a su mi my son are was era".split())
# "mi número es el 867 119 2054", "my number is ...": a possessive before "número"/"number" names the driver's own.
_POSSESSIVES = frozenset("mi my su your tu his her".split())

# Seal mentions. Spelled letter names are only the ones that are not also everyday words ("de", "te", "ese", "ve").
_SEAL_KEYWORDS = frozenset({"sello", "seal", "precinto", "marchamo"})
_SEAL_PLURALS = frozenset({"sellos", "seals", "precintos", "marchamos"})  # more than one seal: the witness has one
# The only words that may stand between the keyword and the id (ORDER Nube-9): any other word there gives None and
# the reason, the id is never read past it ("el sello de la carta es ...", "the seal on the paperwork is ...").
# "No." is the abbreviation ("sello No. MX-48213"; "el sello no es MX-48213" negates it); "de seguridad" is part of
# the keyword ("sello de seguridad").
_SEAL_LEAD_IN = frozenset("es is el the número numero núm number nº reads".split())
_SEAL_ABBREVIATIONS = frozenset({"no", "num", "núm", "nro", "n"})  # before a "."
# Said anywhere after the seal, to the end of the note (ORDER Nube-10 item 2), or before it in its sentence, any of
# these means it cannot be taken as read: "el sello MX-48213 viene roto", "... Está roto.", "they broke the seal ...",
# "le quitaron el sello ...", "falta el sello". After a negation (_NEGATED_BEFORE) they say it is fine: "no está roto",
# "nada roto", "nothing broken", "nothing missing", "no falta".
_SEAL_BAD = frozenset(
    "roto rota rotos rotas dañado dañada dañados dañadas daño daños damage violado violada violentado violentada "
    "forzado forzada abierto abierta cortado cortada quebrado quebrada alterado alterada despegado despegada falta "
    "faltan faltó falto faltaba faltante perdido perdida perdí perdi perdió perdio diferente distinto distinta broken "
    "broke damaged tampered cut opened missing lost removed gone mismatch mismatched wrong different lacks lacking "
    "loose suelto suelta flojo floja violaron violó violo cortaron rompieron rompió rompio abrieron bad mal cracked "
    "busted snapped ripped torn bent popped twisted pulled sheared quitaron quitado quitada arrancaron arrancado "
    "arrancada reventado reventada chueco chueca doblado doblada colgando desprendido desprendida cambiaron cambiado "
    "cambiada zafado zafada hole holes agujero agujeros hoyo hoyos perforado perforada tronado tronada tronó "
    "estrellado estrellada golpeado golpeada golpearon mordido mordida partieron desmontaron soldaron soldadura "
    "smashed chewed glued pried drilled welded".split()
)
_SEAL_BAD_PHRASES = tuple(tuple(p.split()) for p in (
    "se cayó", "se cayo", "se lo llevaron", "se la llevaron", "no sirve", "se zafó", "se zafo", "came off",
    "fell off", "taken off", "took it off", "is off", "s off", "was off", "se metió", "se metio", "se metieron",
    "se sale", "partido por", "partida por", "le sacaron", "tirado en", "no estaba en", "quisieron abrir", "se salió",
    "se salio", "comes off",
))
# Plain non-seal uses of a damage word: these alone never take the seal (an exact list).
_BAD_NOT_SEAL = tuple(tuple(p.split()) for p in (
    "pulled in", "pulled into", "pulled up", "pulled over", "wrong turn", "wrong way",
    "wrong exit", "bad traffic", "traffic was bad", "traffic is bad", "bad weather", "weather was bad", "too bad",
    "cut me off",
))
# Right after the id, in its sentence only: "seal MX-48213 off".
_CLAUSE_BAD = frozenset({"off"})
# Before a damage word these say it is not so, with only _NEGATION_GAP words between: "nada roto", "nothing missing".
_AFFIRMING_NEGATIONS = frozenset("nothing nada sin without ningún ningun ninguno ninguna".split())
# ... unless a condition stands before the negation: "if I'm not wrong", "si no está roto" say nothing is fine.
_CONDITIONS = frozenset("if si unless".split())
_CONDITION_GAP = frozenset("i m am it s is".split())
_WITHOUT = frozenset({"sin", "without"})  # "sin sello", "without a seal"
# "no está", "no lo tengo", "don't have it", "isn't there": a negation, these, and the end of the clause.
_MISSING_CHAIN = frozenset(
    "está esta estaba están estan hay había habia tengo tiene tienen trae traigo traemos viene vienen veo encuentro "
    "encontré encontre find found see have has had got is was are there here it lo la le one uno una".split()
)
_CLAUSE_JOINS = frozenset("y and pero but so".split())
_COPULAS = frozenset("es era".split())  # "no es VN"
# Said after a value, these take it back without a replacement; every field said before them gives None ("Mentira.",
# "olvídalo", "never mind"); so does a pronoun for what was said (_RETRACT_PRONOUNS) with, later in its clause, a word
# that puts it elsewhere (_RETRACT_PLACES): "No, ese es el de la otra caja", "ese era el de ayer", "Wait, no, that's
# from my last load".
_RETRACTIONS = tuple(tuple(p.split()) for p in (
    "mentira", "olvídalo", "olvidalo", "olvida", "olvídate", "olvidate", "olvídense", "olvidense", "borra", "táchalo",
    "tachalo", "tacha", "retiro", "retirar", "retiré", "retire lo", "no me hagas caso", "no le hagas caso",
    "no hagas caso", "no le haga caso", "nada de eso", "ese no", "esa no", "no es ese", "no es esa", "me equivoqué",
    "me equivoque", "me confundí", "me confundi", "conté mal", "conte mal", "está mal", "esta mal", "forget it",
    "forget that", "forget both", "forget all", "forget everything", "forget those", "forget about", "never mind",
    "nevermind", "ignore it", "ignore that", "ignore the", "ignore those", "disregard", "take back", "take it back",
    "take that back", "taking back", "scratch that", "scratch the", "strike that", "strike all", "cancel that",
    "none of that", "miscounted", "i messed up", "that s not it", "not that one", "wrong load", "wrong paperwork",
    "wrong trailer", "wrong number", "throw that", "throw all", "throw it", "throw those", "scratch both",
    "scratch all", "scratch those", "scratch it", "mixed up", "mixed it up", "turned around", "al revés", "al reves",
    "backwards", "no te sirve", "no les sirve", "not the count", "not the number", "no es el número",
    "no es el numero", "way off", "wrong count", "carga equivocada", "número equivocado", "numero equivocado",
))
# These take a value back only at the end of a clause: "1,400's not right.", "no va,", "no es así." ("no va a llover"
# says nothing of it).
_RETRACTIONS_AT_END = tuple(tuple(p.split()) for p in (
    "not right", "isn t right", "s not right", "no va", "no es así", "no es asi", "no es correcto", "not correct",
    "no es",
    "isn t correct",
))
_RETRACT_PRONOUNS = frozenset("ese esa eso that count conteo cuenta".split())  # "the bag count is wrong"
_RETRACT_PLACES = frozenset(
    "otra otro otras otros other another last previous anterior pasada pasado ayer yesterday old vieja viejo "
    "equivocado equivocada wrong before earlier antes entregué delivered".split()
)
# Told not to use what was said: a negation, these pronouns, then one of _DONT_USE ("don't log that", "no lo apuntes",
# "no lo tomes en cuenta", "don't go by that box count", "don't use that seal number").
_DONT_USE_WHAT = frozenset("lo la los las eso that it this those".split())
_DONT_USE = frozenset("use log write record go take enter put apuntes anotes tomes uses registres pongas escribas "
                      "captures apunten anoten tomen usen registren".split())
# Said right after a value, these say it is wrong: "the 350 is wrong", "las 34 están mal".
_WRONG = frozenset("wrong mal off incorrect incorrecto incorrecta equivocado equivocada".split())
# A part of the seal said: "seal 7731, last four", "los últimos cuatro".
_SEAL_PARTIAL = frozenset("last últimos últimas ultimos ultimas".split())
_SOMETHING = frozenset({"algo", "something"})  # right after the id: "183 something", "PR 8718 y algo"
# Anywhere after the seal, the id was not read in full: "no se alcanza a ver el resto", "rest is smudged".
_UNREAD = frozenset("resto rest borroso borrosa smudged scratched ilegible illegible unreadable faded despintado "
                    "despintada".split())
# A letter said after the digits in words: "con letra b", "with a B at the end".
_LETTER_WORDS = frozenset("letra letras letter letters".split())
# The trailer itself said to be open or broken: "la caja viene abierta", "the trailer door is open".
_TRAILER = frozenset("caja tráiler trailer remolque puerta puertas door doors".split())
_OPEN = frozenset({"open"})  # said of the trailer: "the trailer door is open"
# The seal said as a verb: "it's not sealed", "sin sellar", "came unsealed".
_SEALING = frozenset("sealed unsealed sellar sellaron sellado sellada sellados selladas".split())
# Said of the seal these say it is fine ("el sello coincide", "it matches the paperwork"); negated, that it is not
# ("no coincide", "doesn't match").
_MATCH_GAP = frozenset("con with as que to".split())  # "coincide con la carta", "same as the paperwork"
_SEAL_MATCH = frozenset(
    "coincide coinciden corresponde corresponden cuadra cuadran concuerda match matches matched matching agrees same "
    "igual mismo misma intact intacto intacta trust confío confio confiar fiar".split()
)
# Words that may stand between a negation and the word it negates: "no está roto", "has not been opened".
_NEGATION_GAP = frozenset("está esta estaba están estan viene vienen venía venia ha han has have had been be is was "
                          "are fue se le lo la it es el the a un una look looks seem seems ve veo even too tan so "
                          "muy".split())
# A negation about the cargo, not the seal: "MX-48213 no hay material peligroso".
_HAZMAT = frozenset("peligroso peligrosa peligrosos peligrosas hazmat hazardous".split())
# What may stand between such a negation and the cargo word: "no traigo material peligroso", "there is no hazmat".
_CARGO_GAP = frozenset("traigo trae hay tengo tiene llevo lleva material materiales carga cargo producto productos any "
                       "is there have has we i it de del el la los las the a an es son".split())
# Words that may stand between a negation and the keyword before it: "no traigo el sello", "there is no seal".
_SEAL_BEFORE_SKIP = frozenset(
    "el la los las lo un una the a an any ningún ningun ninguno ninguna su sus mi this that traigo trae traemos tengo "
    "tiene tienen tenemos hay había habia viene vienen venía venia llevo lleva llevamos has have had got is está esta "
    "es was there veo vi se le".split()
)
# English letter names that are not words: "em ex dash four ...", "el zee ..." (with _WORD_LETTERS).
_EN_SPELLED = {"bee": "B", "cee": "C", "dee": "D", "gee": "G", "jay": "J", "kay": "K", "pee": "P", "cue": "Q",
               "tee": "T", "vee": "V", "zee": "Z", "ex": "X", "ef": "F", "ar": "R", "why": "Y", "see": "C"}
_SPELLED = {**_EN_SPELLED, "be": "B", "ce": "C", "efe": "F", "ge": "G", "hache": "H", "jota": "J", "ka": "K",
            "ele": "L", "eme": "M", "ene": "N", "eñe": "Ñ", "pe": "P", "cu": "Q", "erre": "R", "uve": "V", "equis": "X",
            "ye": "Y", "zeta": "Z"}
_NOT_SEAL = frozenset({"OK", "NO", "SI", "YA"})
# Capitalized at the start of a sentence these are words, never seal letters: "Y pesa cuatro mil ochocientos kilos".
_SENTENCE_WORDS = frozenset({"Y", "O", "E", "U"})
# A number joined to one of these is a size, never an id: "It's a 53-foot dry van", "caja de 53-pies".
_SIZE_UNITS = frozenset("foot feet ft footer pies pie inch inches pulgadas metros meters metres".split())
# The only plain non-seal uses of _UNREAD words: anywhere else after the seal, "resto", "rest" give it None.
_UNREAD_EXCEPTIONS = tuple(tuple(p.split()) for p in (
    "el resto está bien", "el resto esta bien", "the rest is fine", "get some rest", "rest area",
))
# Letter names that are also everyday words ("a", "de", "te", "ese"): a letter only next to another spelled letter
# ("a be", "te equis", "ese eme equis", "ce a guión ..."), or after a dash at the end of a sentence ("guión de.").
_WORD_LETTERS = {"a": "A", "de": "D", "te": "T", "ese": "S", "em": "M", "el": "L"}  # "el" only before an English name
# Both a lead-in, a filler or an everyday word and a letter name: as the id's first letter, or right before a spelled
# letter that starts it, the seal gives None ('unclear seal letters'): "the seal is el em ex four ...", "sello ese eme
# equis ...", "el sello es eme equis ...", "the seal is, see, MX-48213" (ORDER Nube-10 item 6).
_LETTER_NAME_WORDS = frozenset("el es ese te a see em why".split())
# The local ASR writes a seal's letters in lowercase ("el sello es mx guión cuatro..."). Function words never are seal
# letters, whatever follows them ("el sello es de plástico", "cuatro ocho y dos uno tres").
_NOT_SEAL_LETTERS = frozenset(
    "es el la los las de del y o en con sin que se su mi tu al lo "
    "a e u ni le les me te nos mis tus sus por pa pal ya si son era fue ha he han hay va van voy da dan doy ver vez "
    "muy tan mas ese esa eso ok oh "
    "an and are as at by do for go has i if in is it its of on or per so the to up us was we yes".split()
) | _WEIGHT_UNITS | _SPEECH_FILLERS
# How an id is read over the radio: inside a seal id, before its digits, these are its letters ("Seal is Whiskey,
# India, three, oh, ..."), as is a letter said with a word that starts with it ("T as in Tom", "pe de Pablo").
_RADIO = {"alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F",
          "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J", "kilo": "K", "lima": "L", "mike": "M",
          "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T",
          "uniform": "U", "victor": "V", "whiskey": "W", "whisky": "W", "xray": "X", "yankee": "Y", "zulu": "Z"}
_AS_IN = tuple(tuple(p.split()) for p in ("as in", "for", "like", "de", "como"))
_LETTER_PAIRS = {("be", "grande"): "B", ("be", "larga"): "B", ("be", "alta"): "B", ("ve", "chica"): "V",
                 ("ve", "corta"): "V", ("ve", "baja"): "V",
                 ("doble", "ve"): "W", ("doble", "uve"): "W", ("uve", "doble"): "W", ("doble", "u"): "W",
                 ("i", "griega"): "Y", ("double", "u"): "W", ("double", "you"): "W"}
# Words that can follow a seal's last digit without being counted by it: "... uno tres traigo doce tarimas", "... tres
# y son de madera", "... tres de la caja", "... tres peso cuatro mil ...", "... tres pesa cuatro mil ...", "... tres
# sin material peligroso", "... tres buenas tardes". Any other word right after digits said in a run ("... uno tres
# doce paquetes", "... tres dos horas esperando") may be counted by the last number, and a copula may make it the
# subject ("... tres cuatro es el andén"), so the id does not read one way.
_ENDS_SEAL = frozenset(
    "el la lo los las the pero but que traigo llevo tengo cargo cargué cargue entrego hay viene trae tiene lleva has "
    "does did seems looks i we it ok okay gracias thanks thank please listo repito peso weight sin without del "
    "pesa pesan weighs weigh son ahí ahi buenas buenos buen muchas nada saludos saludo delivering bringing carrying "
    "hauling unloading loading".split()
)
_DIGIT_OF = {w: str(v) for w, v in (*_ES_ONES.items(), *_EN_ONES.items(), ("cero", 0), ("zero", 0))
             if w not in ("un", "una")}

_TOKEN = re.compile(
    r"(?P<time>\d{1,2}:\d{2}(?![\d:])(?:\s?[ap]\.?m\b\.?)?|\d{1,2}\s?(?:[ap]\.m\.|[ap]m\b|o['’]clock))"
    # "Got here at 1:45", "3 a.m.", "10 o'clock": a clock time, never a count or an id
    r"|(?P<num>\d{1,3}(?:[.,]\d{3})+(?![\w-])|\d+[.,]\d+(?![\w-]))"
    r"|(?P<word>[^\W_]+(?:[-‐][^\W_]+)*)"
    r"|(?P<stop>[.;!?…]+)"
    r"|(?P<comma>[,:])"
    r"|(?P<dash>[-‐–—])",
    flags=re.IGNORECASE,  # "8 PM", "5 AM"
)
_THOUSANDS = re.compile(r"\d{1,3}(?:[.,]\d{3})+")
_GROUPED = re.compile(r"\d+(?:[.,]\d+)+")
_LOW_PUNCT = {"stop": ".", "comma": ",", "dash": "-"}
_CLITICS = frozenset({"s", "t", "re", "ll", "ve", "d", "m"})


# Words compared inside the functions below, named so that each has a case in tests/test_voice_words.py.
_ARTICLE_WORDS = frozenset("un una a an one".split())  # never a seal digit or an id on their own
_INDEFINITE = frozenset("un una a an".split())
_EN_INDEFINITE = frozenset("a an".split())  # "a dozen", "a thousand", "fifteen hundred"
_ES_INDEFINITE = frozenset("un una".split())  # "un"/"una" are 1 only as a count: never a digit of an id
_THOUSAND = frozenset("mil thousand".split())
_REPEATS = frozenset("doble double triple".split())  # "doble ocho": 88
_AND_A_HALF = ("y media", "and a half")  # "dos docenas y media": +6
_TRAILER_ARTICLES = frozenset("la el the".split())  # "la caja viene abierta"
_DE_ARTICLES = frozenset("la el".split())  # "... tres de la caja": "de" ends the seal's digits
_AND = frozenset("y and".split())  # "a las ocho y diez", "cuarenta y ocho"
_KEYWORD_TAILS = tuple(tuple(p.split()) for p in ("de seguridad",))  # "sello de seguridad" is the keyword itself

def _tokens(text: str) -> list[tuple[str, str]]:
    """(kind, text) per token. 'twenty-four' splits into words; 'MX-48213' stays one token; 'twelve-ish' is 'twelve'
    and the hedge 'ish'. The 's' of "seal's" (and the 't' of "don't") is a 'clitic': never a seal letter."""
    out: list[tuple[str, str]] = []
    for m in _TOKEN.finditer(text):
        kind, t = m.lastgroup or "", m.group()
        ish = re.fullmatch(r"(.+?)([-‐]?)(ish)", t, flags=re.IGNORECASE) if kind == "word" else None
        if ish and (ish.group(2) or ish.group(1).isdecimal() or ish.group(1).lower() in _NUMBER_WORDS):
            # "twelve-ish", "12ish", "MX-48213-ish": the value, then the hedge "ish" ("finish", "Spanish" stay words)
            out += _tokens(ish.group(1)) + [("word", ish.group(3))]
            continue
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
    m = _EN_ONES.get(w) or _EN_TEENS.get(w) or (1 if w in _EN_INDEFINITE else None)  # "fifteen hundred"
    if m is not None and _w(ws, i + 1) == "hundred":
        j = i + 2
        t = _en_tens_part(ws, j + 1 if _w(ws, j) == "and" else j)
        return (m * 100 + t[0], t[1]) if t else (m * 100, j)
    t = _en_tens_part(ws, i)
    if t and t[0] > 20 and t[0] % 10 and _w(ws, t[1]) == "hundred":  # "forty-eight hundred kilos": 4800
        j = t[1] + 1
        r = _en_tens_part(ws, j + 1 if _w(ws, j) == "and" else j)
        return (t[0] * 100 + r[0], r[1]) if r else (t[0] * 100, j)
    return t


def _en_cardinal(ws: list[str], i: int) -> tuple[int, int] | None:
    if _w(ws, i) == "zero":
        return 0, i + 1
    if _w(ws, i) in _EN_INDEFINITE and _w(ws, i + 1) == "thousand":
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
        if _w(ws, i + 1) in _THOUSAND:
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
        elif w in _EN_INDEFINITE and _w(ws, i + 1) == "dozen":
            base = (1, i + 1)
        elif w == "media" and _w(ws, i + 1) == "docena":
            return 6, i + 2
        elif w == "half" and _w(ws, i + 1) in _EN_INDEFINITE and _w(ws, i + 2) == "dozen":
            return 6, i + 3
        elif w in _DOZENS:
            return None, i + 1  # "docenas de cajas": how many?
        else:
            return None
    v, j = base
    if _w(ws, j) not in _DOZEN | _DOZENS or v is None:
        return base
    v, j = v * 12, j + 1
    if " ".join(ws[j:j + 2]) in _AND_A_HALF:
        v, j = v + 6, j + 2
    elif " ".join(ws[j:j + 3]) in _AND_A_HALF:
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
        if lows[j] not in _ES_INDEFINITE and _number_at(lows, j, dozens=False):
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
    if kind not in ("word", "num") or text in _NOT_SEAL or lows[i] in _ES_INDEFINITE:
        return False
    n = _number_at(lows, i, dozens=False)
    if n:
        after = _w(lows, n[1])
        return not (_unit(lows, n[1]) or after in _WEIGHT_UNITS)
    nxt = _w(lows, i + 1)
    if lows[i] in _RADIO:
        return _radio_letter(toks, lows, i)
    return lows[i] in _SPELLED or (lows[i], nxt) in _LETTER_PAIRS or bool(_word_letter(lows, i)) or (
        lows[i] in _REPEATS and nxt in _DIGIT_OF) or _lowercase_letters(toks, lows, i) or bool(
        re.fullmatch(r"[A-Za-z0-9-]+", text) and (any(c.isdigit() for c in text) or (text.isupper() and len(text) <= 3))
    )


def _radio_letter(toks: list[tuple[str, str]], lows: list[str], i: int) -> bool:
    """Is the radio word at i a letter: followed, past commas, by another radio word, a spelled letter, a dash or
    digits ('Papa, Lima, six ...'; not 'Mike from dispatch')?"""
    j = i + 1
    while j < len(toks) and toks[j][0] == "comma":
        j += 1
    return j < len(toks) and (lows[j] in _RADIO or lows[j] in _SPELLED or toks[j][0] == "dash" or lows[j] in _SEAL_DASH
                              or any(ch.isdigit() for ch in toks[j][1]) or bool(_number_at(lows, j, dozens=False)))


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
    if kind not in ("word", "num") or text in _NOT_SEAL or lows[j] in _INDEFINITE:
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
            w in _REPEATS and nxt in _DIGIT_OF)
            or (w, nxt) in _LETTER_PAIRS or any(ch.isdigit() for ch in text)
            or (text.isupper() and text.isalpha() and len(text) <= 3 and text not in _NOT_SEAL and text != "I"))


def _ends_seal(lows: list[str], j: int) -> bool:
    """Does the word at j end a seal's digits cleanly (it cannot be what the last number counts)?"""
    w = lows[j]
    return (w in _ENDS_SEAL or w in _NEGATIONS or w in _NEGATED_CONTRACTIONS or w in _SPEECH_FILLERS
            or w in _SEAL_KEYWORDS or w in _ALTERNATIVES or bool(_phrase_at(lows, j, _CORRECTIONS))
            or (w == "de" and _w(lows, j + 1) in _DE_ARTICLES) or (w in _ID_NOUNS and not _unit(lows, j))
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
        if out and out[-1][-1:].isalpha() and (n := _phrase_at(lows, i, _AS_IN)) and (
                (w := _w(lows, i + n))[:1].upper() == out[-1][-1] and w.isalpha() and len(w) > 2
                and w not in _NUMBER_WORDS and w not in _SPELLED):
            i += n + 1  # "T as in Tom", "pe de Pablo": the letter said with a word that starts with it
            continue
        if low in _RADIO and not digits and _radio_letter(toks, lows, i):
            out.append(_RADIO[low])  # "Whiskey, India, three, ...": the radio alphabet
            i += 1
            continue
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
        if low in _REPEATS and nxt in _DIGIT_OF:
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
                _number_at(lows, i, dozens=False)) and not (len(low) > 1 and set(low) & set("aeiou")):
            out.append(text.upper())  # "... two one three dash b": the letter after the dash is the id's
            i += 1
            continue
        if low not in _ES_INDEFINITE and _number_at(lows, i, dozens=False):
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
            written_groups = len(groups) > 1 and all(lows[g[1]].isdecimal() for g in groups) and not spoken
            if written_groups and j < len(toks) and (toks[j][0] == "dash" or lows[j] in _SEAL_DASH):
                out += [lows[g[1]] for g in groups]  # "54 6801 dash A": digits written in groups, the id goes on
                written, i = True, j
                continue
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
        if low in _SPELLED and not (digits and low in _EN_SPELLED and out[-1] != "-") and not (
                len(text) > 1 and text.isupper()):  # "... 7731 gee thanks"; "AR 4471" is written letters, not "ar"
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
        if digits and text.isalpha() and text.isupper() and (toks[i - 1][0] == "comma" or _w(lows, i + 1) in (
                _NUMBER_NOUNS | _SEAL_ABBREVIATIONS) or any(ch.isdigit() for ch in (toks[i + 1][1] if i + 1 < len(toks)
                                                                                     else ""))):
            break  # "seal MX-48213, PO number 55123", "... BOL 77812": an acronym is never glued to the id
        if text not in _NOT_SEAL and re.fullmatch(r"[A-Za-z0-9\-]+", text) and (
                any(ch.isdigit() for ch in text) or (text.isupper() and len(text) <= 3)):
            # "MX MX guión ..." (real transcript): a letter group repeated back to back is one ASR stutter.
            # Letters only: "cuatro cuatro" stays 44. Residual risk, accepted and written down: a driver who
            # really says "MX MX 48213" against a paper reading "MX-48213" would match (a "MXMX" paper HOLDs).
            if out and out[-1].isalpha() and not text.isalpha() and text.upper().startswith(out[-1]) and (
                    text[len(out[-1]):][:1] in "-0123456789"):
                out.pop()  # "MX MX-770836": the same stutter, the group said again with the digits glued to it
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
    while j < len(toks) and toks[j][0] != "stop":
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
    not), ('', j) for anything else ('el sello no es MX-48213'). Only words of _NEGATION_GAP stand between."""
    e = j
    while e < len(lows):
        w = lows[e]
        if w in _SEAL_BAD:
            return "bad", e + 1
        if w in _SEAL_MATCH:
            return "match", e + 1
        if w not in _NEGATION_GAP:
            break
        e += 1
    return "", j


def _negated(phrase: list[str]) -> str:
    return f"negated seal in transcript: {_said(phrase)!r}"


def _reported(phrase: list[str]) -> str:
    return f"seal reported broken, missing or not matching in transcript: {_said(phrase)!r}"


def _hedged(phrase: list[str]) -> str:
    return f"hedged seal in transcript: {_said(phrase)!r}"


def _said(phrase: list[str]) -> str:
    """The words of a reason; a long stretch keeps its first 4 and last 6 words."""
    return " ".join(phrase) if len(phrase) <= 12 else " ".join(phrase[:4] + ["…"] + phrase[-6:])


def _sentence(ws: list[str], i: int) -> tuple[int, int]:
    """[start, end) of the sentence holding ws[i]: between stops ('.'), the whole note when it has none (as the local
    ASR writes it). Every scan for a hedge, a source or a negation ends here (ORDER Nube-10 item 9)."""
    a = i
    while a > 0 and ws[a - 1] != ".":
        a -= 1
    b = i
    while b < len(ws) and ws[b] != ".":
        b += 1
    return a, b


def _negated_before(ws: list[str], j: int) -> bool:
    """Is the word at j negated by what comes right before it, past only words of _NEGATION_GAP ('no falta', 'nada
    roto', 'nothing missing', "doesn't match", 'no se ha abierto')? Not after 'if'/'si': "if I'm not wrong" is no
    affirmation."""
    p = j - 1
    while p >= 0 and ws[p] in _NEGATION_GAP:
        p -= 1
    if p < 0:
        return False
    if ws[p] in _NEGATIONS or ws[p] in _AFFIRMING_NEGATIONS:
        q = p - 1
    elif ws[p] == "t" and _w(ws, p - 1) in _NEGATED_CONTRACTIONS:
        q = p - 2
    else:
        return False
    while q >= 0 and ws[q] in _CONDITION_GAP:
        q -= 1
    return not (q >= 0 and ws[q] in _CONDITIONS)


def _inside(ws: list[str], j: int, phrases: tuple[tuple[str, ...], ...]) -> bool:
    """Is the word at j part of one of these phrases ('rest area', 'pulled in')?"""
    return any(tuple(ws[j - k:j - k + len(p)]) == p for p in phrases for k, w in enumerate(p) if w == ws[j] and j >= k)


def _bad_at(ws: list[str], j: int) -> int:
    """Length of a damage word or phrase at j ('roto', 'cracked', 'se cayó', 'no sirve'), 0 if none, if it is negated
    ('nada roto', 'no está roto', 'nothing broken') or one of the plain non-seal uses of _BAD_NOT_SEAL ('pulled in')."""
    n = _phrase_at(ws, j, _SEAL_BAD_PHRASES) or (1 if ws[j] in _SEAL_BAD else 0)
    if not n or _inside(ws, j, _BAD_NOT_SEAL) or (ws[j] in _SEAL_BAD and _negated_before(ws, j)):
        return 0
    return n


def _clause_end(ws: list[str], j: int) -> int:
    while j < len(ws) and ws[j] not in (".", ","):
        j += 1
    return j


def _denied_match(ws: list[str], j: int) -> int:
    """After a negation ending before j: the end of a match word said later in the same clause, whatever words stand
    between ('no me coincide', "doesn't even match", 'no coincide con el material peligroso'), or 0."""
    return next((q + 1 for q in range(j, _clause_end(ws, j)) if ws[q] in _SEAL_MATCH), 0)


def _missing_at(toks: list[tuple[str, str]], lows: list[str], j: int) -> int:
    """'no está', "don't have it", 'no lo tengo', "isn't there": a negation at j, words of _MISSING_CHAIN, then the end
    of the clause. Its end, or 0."""
    n = _negation_at(toks, lows, j)
    q = j + n
    if not n or _w(lows, q) not in _MISSING_CHAIN:
        return 0
    while q < len(lows) and lows[q] in _MISSING_CHAIN:
        q += 1
    return q if q == len(lows) or lows[q] in (".", ",") or lows[q] in _CLAUSE_JOINS else 0


def _retraction_at(ws: list[str], j: int) -> int:
    """Length of a retraction at j: 'mentira', 'olvídalo', 'never mind', or a pronoun for what was said and, later in
    its clause, a word that puts it elsewhere ('ese es el de la otra caja', 'ese era el de ayer', "that's from my last
    load"). 0 if none."""
    if n := _phrase_at(ws, j, _RETRACTIONS):
        return n
    if (n := _phrase_at(ws, j, _RETRACTIONS_AT_END)) and _clause_end(ws, j) == j + n:
        return n  # "1,400's not right.", "..., no va."
    if ws[j] in _RETRACT_PRONOUNS:  # "that's the shipper's count", "eso me lo dictó el de la oficina": someone else's
        return next((q + 1 - j for q in range(j + 1, _clause_end(ws, j)) if ws[q] in _RETRACT_PLACES
                     or ws[q] in _PARTIES or (ws[q] in _DOCS and ws[q] not in _ARTICLE_DOCS) or ws[q] in _TELLING), 0)
    if ws[j] in _GIVEN_WHAT and (_w(ws, j + 1) in _TELLING or _w(ws, j + 1) in _GIVING or _w(ws, j + 1) in (
            _PARTY_ACTS)):
        return 2  # "Lo de los sacos me lo pasó el de báscula, él los contó": what was said came from someone else
    n = 2 if _w(ws, j) in _NEGATED_CONTRACTIONS and _w(ws, j + 1) == "t" else int(_w(ws, j) in _NEGATIONS)
    if n:  # "don't log that", "no lo apuntes", "no lo tomes en cuenta"
        q = j + n
        while _w(ws, q) in _DONT_USE_WHAT:
            q += 1
        if _w(ws, q) in _DONT_USE:
            return q + 1 - j
    if ws[j] == "ni":  # "ni las tarimas ni el sello"
        q = j + 1
        while ws[q:q + 1] and ws[q] in _ID_ARTICLES:
            q += 1
        return q + 1 - j if _w(ws, q) == "|" or _w(ws, q) in _SEAL_KEYWORDS or _is_unit(_w(ws, q)) else 0
    return 0


def _about_cargo(ws: list[str], j: int) -> bool:
    """Does the negation ending before j speak of the cargo ('no hay material peligroso', 'no hazmat', 'nothing
    hazardous'), past only words of _CARGO_GAP?"""
    while j < len(ws) and ws[j] in _CARGO_GAP:
        j += 1
    return _w(ws, j) in _HAZMAT


def _hedge_in(ws: list[str], a: int, b: int, skip: frozenset[int] | set[int] = frozenset()) -> tuple[int, int]:
    """The first hedge in ws[a:b]: (start, end), or (-1, -1). A hedge covers every value of its sentence, before or
    after it ('Creo que lo que traigo son doce tarimas', 'doce tarimas, creo', 'Six boxes, I think, the seal ...');
    one said between a seal keyword and its id (skip) is that seal's alone ('doce tarimas, el sello creo que es ...'),
    and one said right before a weight is the weight's ('maybe 14,000 pounds')."""
    return next(((j, j + n) for j in range(a, b) if j not in skip and (n := _phrase_at(ws, j, _HEDGES))
                 and not _weight_after(ws, j + n)), (-1, -1))


def _weight_after(ws: list[str], j: int) -> bool:
    """Does a weight follow at j ('maybe 14,000 pounds', 'como 38 mil libras')? A hedge said there is the weight's."""
    while _w(ws, j) in _WEIGHT_GAP:
        j += 1
    n = _number_at(ws, j, dozens=False)
    return bool(n) and _w(ws, n[1]) in _WEIGHT_UNITS


def _seal_denied_before(toks: list[tuple[str, str]], lows: list[str], k: int) -> str:
    """Why the seal said at k is denied by what its sentence says before it: a negation, 'sin' or a negated match right
    before it ('no traigo el sello', 'sin sello', 'no coincide el sello'), or a damage word anywhere before it ('they
    broke the seal', 'le quitaron el sello', 'falta el sello')."""
    a, _ = _sentence(lows, k)
    for j in range(k - 1, a - 1, -1):
        if toks[j][0] not in ("word", "clitic"):
            break
        if lows[j] in _SEAL_MATCH:  # "coincide el sello" says it is fine; "no coincide el sello" that it is not
            if _negated_before(lows, j):
                return _reported(lows[j - 1:k + 1])
            break
        if lows[j] in _NEGATIONS or (toks[j] == ("clitic", "t") and _w(lows, j - 1) in _NEGATED_CONTRACTIONS):
            return _negated(lows[j - (lows[j] == "t"):k + 1])
        if lows[j] in _WITHOUT:
            return _reported(lows[j:k + 1])
        if lows[j] not in _SEAL_BEFORE_SKIP:
            break
    for j in range(a, k):
        if n := _bad_at(lows, j):
            return _reported(lows[j:k + 1])
        if (n := _negation_at(toks, lows, j)) and (e := _denied_match(lows, j + n)) and e <= k:
            return _reported(lows[j:k + 1])
    return ""


def _hedged_before(toks: list[tuple[str, str]], lows: list[str], k: int) -> str:
    """'creo que el sello es ...', 'No sé si el sello es ...': a hedge anywhere before the keyword in its sentence."""
    a, _ = _sentence(lows, k)
    h, _ = _hedge_in(lows, a, k)
    return " ".join(lows[h:k + 1]) if h >= 0 else ""


def _source_at(ws: list[str], j: int, a: int, b: int, after: bool = False) -> int:
    """Length of a source said at j, in the sentence ws[a:b] of a value, or 0 (ORDER Nube-10 item 4).
    - a verb of telling or giving, with any subject: 'dice', 'me dijeron', 'me pasó', 'gave me', 'per', 'según';
    - a document or device not named by its number ('the sheet has ...', 'en la remisión viene ...'; not 'factura
      77812');
    - a person or place, unless the driver is the one acting ('vengo de la planta', 'traigo ... del cliente', 'I came
      straight from the shipper', 'ya pasé la caseta') and nobody counts, seals, tells or gives in the sentence.
    after=True (said after the value): not a verb of telling, mostly the driver's own ('twelve pallets, like I
    said')."""
    w = ws[j]
    if _inside(ws, j, _HEDGES):
        return 0  # "I'd say", "give or take", "según yo": the hedge's word
    if w in _SCOPING:
        return 1
    q = j - 1
    while q >= a and ws[q] in _GIVEN_WHAT:
        q -= 1
    given = q >= a and ws[q] in _GIVEN_TO  # "me pasó", "me las contó el muchacho", "nos lo dijeron"
    if w in _TELLING:
        return 0 if after and not given else 1
    if w in _GIVING:
        return 1 if given else 0
    if w in _ACTS_BY_OTHERS:  # "The forklift driver counted nineteen pallets"; not "I counted", "I haven't counted"
        p = j - 1
        while p >= a and ws[p] in _AUXILIARIES:
            p -= 1
        return 0 if p >= a and ws[p] in _FIRST_PERSON else 1
    n = _phrase_at(ws, j, _SOURCE_PHRASES) or (1 if w in _DOCS or w in _PARTIES else 0)
    if w in _ARTICLE_DOCS and _w(ws, j - 1) not in _DOC_DETERMINERS:
        return 0  # "toallas de papel", "printer paper", "it's Bill with eleven pallets"
    if not n or _numbered(ws, j):
        return 0  # "factura 77812", "la factura es 77812": the document's number
    if w in _DOCS:
        p = j - 1
        while p >= a and (ws[p] in _ID_ARTICLES or ws[p] in _MATCH_GAP):
            p -= 1
        return 0 if p >= a and ws[p] in _SEAL_MATCH and not _negated_before(ws, p) else n  # "it matches the paperwork"
    if any(ws[q] in _PARTY_ACTS or ws[q] in _TELLING or (ws[q] in _GIVING and (
            _w(ws, q - 1) in _GIVEN_TO or (_w(ws, q - 1) in _GIVEN_WHAT and _w(ws, q - 2) in _GIVEN_TO)))
           for q in range(a, b)):
        return n  # "el cliente contó ...", "en la planta me contaron ...", "the shipper counted them"
    p = j - 1
    while p >= a and ws[p] in _ID_ARTICLES:
        p -= 1
    return 0 if p >= a and ws[p] in _DRIVER_ACTING else n


def _numbered(ws: list[str], j: int) -> bool:
    """Is the document or party at j named by the number after it ('factura 77812', 'BOL number 77812')? Not by a
    count: 'me dijeron 20 tarimas', 'the sheet has 6 drums'."""
    q = j + 1
    while _w(ws, q) in _ID_LEAD or _w(ws, q) in _NUMBER_NOUNS:
        q += 1
    if _w(ws, q) == "|":
        return True  # its number was set aside as a seal's: "con sello, la factura es 48213"
    n = _number_at(ws, q, dozens=False) if _w(ws, q) not in _ES_INDEFINITE else None
    return bool(n) and not _unit(ws, n[1])


def _source_before(toks: list[tuple[str, str]], lows: list[str], k: int) -> str:
    """'según la carta porte el sello es ...', 'the BOL has seal ...', 'La oficina me pasó el sello ...': a source
    anywhere before the keyword k in its sentence (_source_at)."""
    a, b = _sentence(lows, k)
    return next((_said(lows[j:k + 1]) for j in range(a, k) if _source_at(lows, j, a, b)), "")


def _seal_denied_after(toks: list[tuple[str, str]], lows: list[str], k: int, i: int) -> str:
    """Why the seal said at k (its id read up to i) cannot be taken, from the rest of its sentence: a hedge (anywhere
    in it: '..., creo', '..., if I'm not wrong'), an approximation right after it ('... o así'), a source ('MX 55012,
    eso dice la hoja'; not 'it matches the paperwork'), a part ('..., last four', '183 something'), a letter said
    after the digits ('with a B at the end'), or a bare negation before any comma, not about the cargo; then, to the
    end of the note, a damaged, missing, non-matching or retracted seal (_seal_reported_later)."""
    comma, negated = False, ""
    j = i - 1
    while j > k and (toks[j][0] == "comma" or lows[j] in _SPEECH_FILLERS):  # the id reader steps over these
        comma, j = comma or toks[j][0] == "comma", j - 1
    a, b = _sentence(lows, k)
    b = max(b, _sentence(lows, min(i, len(lows) - 1))[1]) if i < len(lows) else b
    if (h := _hedge_in(lows, i, b))[0] >= 0:
        return _hedged(lows[k:h[1]])
    j, denial, affirmed, first = i, False, False, True
    while j < b:
        kind, text = toks[j]
        if kind in ("comma", "dash"):
            comma, j = comma or kind == "comma", j + 1
            denial = denial and kind != "comma"  # "..., no es el de la carta": a denial and a document in one clause
            continue
        if first and (n := _phrase_at(lows, j, _APPROX_AFTER)):
            return _hedged(lows[k:j + n])
        if n := _negation_at(toks, lows, j):
            what, e = _negated_word(lows, j + n)
            if what == "match" or (e := _denied_match(lows, j + n)):  # "no coincide", "no me coincide"
                return _reported(lows[k:max(e, j + n)])
            what, e = _negated_word(lows, j + n)
            denial = denial or (not what and not _about_cargo(lows, j + n))
            if not what and not comma and not negated and not _about_cargo(lows, j + n):
                negated = _negated(lows[k:j + n])
            j, first = max(e, j + n), False
            continue
        if (lows[j] in _CLAUSE_BAD and not _negated_before(lows, j)) or (denial and (
                lows[j] in _DOCS or lows[j] in _PARTIES)):  # "... is off", "..., no es el de la carta"
            return _reported(lows[k:j + 1])
        if _source_at(lows, j, a, b, after=True) and not affirmed:  # "MX 55012, eso dice la hoja"
            return f"seal said from the paperwork or someone else: {_said(lows[k:j + 1])!r}"
        affirmed = affirmed or lows[j] in _SEAL_MATCH
        if lows[j] in _SEAL_PARTIAL or (first and lows[j] in _SOMETHING):
            return f"part of a seal said in transcript: {_said(lows[k:j + 1])!r}"  # "183 something", "y algo"
        if lows[j] in _LETTER_WORDS or (kind == "word" and len(text) == 1 and text.isalpha() and (
                text.isupper() and text != "I" or _letters(toks, lows, j))):
            return f"seal letter said after the digits: {_said(lows[k:j + 1])!r}"  # "with a B at the end"
        j, first = j + 1, False
    return negated or _seal_reported_later(toks, lows, k, i)


def _seal_reported_later(toks: list[tuple[str, str]], lows: list[str], k: int, i: int) -> str:
    """A damaged, missing or non-matching seal said anywhere after the seal at k (its id read up to i), to the end of
    the note (ORDER Nube-10 item 2): '... Está roto.', "... It's broken.", '... no me coincide', '..., no está.',
    "..., don't have it.", '... cuando lo revisé ya estaba roto'."""
    for j in range(i, len(lows)):
        if lows[j] in _SEAL_KEYWORDS or lows[j] in _SEAL_PLURALS:
            break  # the next mention reads on from its own id
        if n := _bad_at(lows, j):
            return _reported(lows[k:j + n])
        if (n := _negation_at(toks, lows, j)) and (e := _denied_match(lows, j + n) or _missing_at(toks, lows, j)):
            return _reported(lows[k:e])
        q = j + n + (_w(lows, j + n) in _COPULAS)
        if n and q < len(toks) and _id_shaped(toks, lows, q) and (any(ch.isdigit() for ch in toks[q][1]) or (
                toks[q][1].isupper() and toks[q][1].isalpha())):
            return _negated(lows[k:q + 1])  # "Espera, no, no es VN", "it isn't 48231": the id said again, negated
    return ""


def _past_lead_in(toks: list[tuple[str, str]], lows: list[str], i: int) -> int:
    """Past the lead-ins after a keyword ('es', 'el', 'número', 'No.', 'de seguridad'), commas and fillers."""
    while i < len(toks):
        if lows[i] in _SEAL_ABBREVIATIONS and toks[i][0] == "word" and _w(lows, i + 1) == ".":
            i += 2  # "No. MX-48213", "núm. MX-48213"
        elif _word_letter(lows, i):
            break  # "seal is el zee ...", "seal is em ex ...": the letters L, M
        elif toks[i][0] in ("comma", "clitic") or lows[i] in _SEAL_LEAD_IN or lows[i] in _SPEECH_FILLERS:
            i += 1
        elif n := _phrase_at(lows, i, _KEYWORD_TAILS):
            i += n
        elif n := _phrase_at(lows, i, _FILLER_PHRASES):
            i += n
        else:
            break
    return i


def _spelled_at(toks: list[tuple[str, str]], lows: list[str], j: int) -> bool:
    """Is the word at j a spelled letter name ('eme', 'equis', 'em', 'zee', 'doble u'), not letters written as such
    ('AR 4471')?"""
    return not toks[j][1].isupper() and (lows[j] in _SPELLED or bool(_word_letter(lows, j))
                                         or (lows[j], _w(lows, j + 1)) in _LETTER_PAIRS)


def _unclear_letter_start(toks: list[tuple[str, str]], lows: list[str], k: int, start: int) -> str:
    """Why the id of the seal said at k, starting at start, does not read one way: its first word is also a lead-in, a
    filler or an everyday word ('ese eme equis ...', 'te equis ...', 'el zee ...', 'em ex ...', ', see, MX-48213'), or
    such a word stands right before a spelled letter that starts it ('el sello es eme equis ...', 'the seal is el em
    ex ...'). '' when it reads one way."""
    nxt = start + 1
    while nxt < len(toks) and (toks[nxt][0] == "comma" or lows[nxt] in _SPEECH_FILLERS):
        nxt += 1
    if lows[start] in _LETTER_NAME_WORDS and nxt < len(toks) and (
            _seal_like(toks, lows, nxt) or toks[nxt][0] == "dash" or lows[nxt] in _SEAL_DASH):
        return f"unclear seal letters: {_said(lows[k:nxt + 1])!r}"
    prev = start - 1
    while prev > k and (toks[prev][0] == "comma" or lows[prev] in _SPEECH_FILLERS):
        prev -= 1
    if prev > k and lows[prev] in _LETTER_NAME_WORDS and _spelled_at(toks, lows, start):
        return f"unclear seal letters: {_said(lows[k:start + 1])!r}"
    return ""


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
            if what == "match" or (not what and not comma and not _about_cargo(lows, j + n)):
                why = why or (_reported(lows[k:e]) if what else _negated(lows[k:j + n]))
            j = max(e, j + n)
            continue
        if n := _bad_at(lows, j):  # "seal's intact, nothing wrong with it" says it is fine
            why, j = why or _reported(lows[k:j + n]), j + n
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


_PHONE = re.compile(r"\d{3}-\d{3}-\d{4}|\d{3}-\d{4}")
_INTERSTATE = re.compile(r"I-\d{1,3}")  # "hit construction on I-65": a highway, never an id


def _not_an_id(toks: list[tuple[str, str]], lows: list[str], j: int) -> bool:
    """A token outside the seal mentions that is never another id: a word capitalized at the start of a sentence ('Y
    pesa ...'), a size ('53-foot'), or a phone number ('956-555-0142') unless a correction stands right before it."""
    text = toks[j][1]
    if (text in _SENTENCE_WORDS and (j == 0 or toks[j - 1][0] == "stop")) or _INTERSTATE.fullmatch(text):
        return True
    if (m := re.fullmatch(r"\d+-?([^\W\d_]+)", lows[j])) and m.group(1) in _SIZE_UNITS:
        return True
    p = j - 1
    while p >= 0 and (toks[p][0] == "comma" or lows[p] in _SPEECH_FILLERS or lows[p] in _AFTER_CORRECTION):
        p -= 1
    return bool(_PHONE.fullmatch(text)) and not _phrase_ending_at(lows, p + 1, _CORRECTIONS)


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
        if j in covered or kind not in ("word", "num") or lows[j] in _INDEFINITE or (
                lows[j] in _ID_NOUNS and not _phrase_ending_at(lows, j, _CORRECTIONS)) or _not_an_id(toks, lows, j):
            pass  # "PO 55123", "BOL 77812": the noun names the number after it
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
                for q in range(end, _sentence(lows, end)[1] if end < len(lows) else end):
                    if lows[q] not in _INDEFINITE and _number_at(lows, q, dozens=False):
                        groups, e = _number_run(lows, q)
                        read2, e2, _ = _seal_value(toks, lows, q)  # "jota, te, seis, cero, dos, uno": past commas
                        digits = re.sub(r"\D", "", read2) or "".join(str(v) for v, _, _ in groups)
                        e = max(e, e2)
                        if not (_is_unit(_w(lows, e)) or _w(lows, e) in _WEIGHT_UNITS or _weighed(lows, q)) and (
                                len(digits) >= 3):
                            value, one = " ".join(lows[j:e]), len(groups) == 1 and e2 <= groups[0][2]
                        break
        elif _number_at(lows, j, dozens=False):
            groups, e = _number_run(lows, j)
            after = _w(lows, e)
            if len(groups) > 1 and (_is_unit(after) or after in _WEIGHT_UNITS):  # "... cuatro siete 15 tarimas"
                groups, after = groups[:-1], "|"
                e = groups[-1][2]
            if all(lows[g].isdecimal() for _, g, _ in groups) and [len(lows[g]) for _, g, _ in groups] in (
                    [3, 3, 4], [3, 4]) and not _phrase_ending_at(lows, j, _CORRECTIONS):
                groups = []  # "867 119 2054": a phone number
            if groups and not (_is_unit(after) or after in _WEIGHT_UNITS or (
                    _is_filler(after) and _is_unit(_w(lows, e + 1)))) and (
                    (len(groups) >= 3 and not any(_compound(g) for g in groups))
                    or max(v for v, _, _ in groups) >= 1000):
                value = "".join(lows[s] if lows[s].isdecimal() else str(v) for v, s, _ in groups)
                one = len(groups) == 1
            e = max(e, j + 1)
        if value and not _named(lows, j) and not _weighed(lows, j):
            out.append((value, j, e, one))
        j = e
    return out


def _corrected_number(toks: list[tuple[str, str]], lows: list[str], j: int, covered: set[int],
                      mention_ends: list[int]) -> str:
    """A number said right after a correction, outside the seal mentions, with no unit, when the seal is the last thing
    said before it: 'The seal is 203. Sorry, 230.' may correct the seal (or the count). Its digits, or ''."""
    if j in covered or lows[j] in _ARTICLE_WORDS or not _number_at(lows, j, dozens=False):
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


def _names(ws: list[str], p: int) -> bool:
    """Does the id noun at p say what the number after it numbers? 'número'/'number' only after a noun ('PO number',
    "the trailer's number") or a possessive ('mi número'): 'the number is ...' names nothing."""
    if ws[p] not in _NUMBER_NOUNS:
        return True
    q = p - 2 if _w(ws, p - 1) == "s" else p - 1  # the clitic of "the trailer's number"
    return q >= 0 and (ws[q] in _NUMBERED or ws[q] in _POSSESSIVES)  # "mi número es el 867 119 2054"


def _named(ws: list[str], s: int) -> bool:
    """Is the number at s said after a noun that says what it numbers ('andén cuatro', 'trailer number is 53', 'andén
    No. 4')? Only lead words stand between them. 'No.' is a lead only as the abbreviation before digits: a correction
    'no' never is ('en la caja, no trece', 'andén no cuatro')."""
    p = s - 1
    while p >= 0:
        if ws[p] in _ID_NOUNS:
            return _names(ws, p)
        if ws[p] == "." and _w(ws, p - 1) in _SEAL_ABBREVIATIONS and _w(ws, s).isdecimal():
            p -= 2
        elif ws[p] in _ID_LEAD or any(ch.isdigit() for ch in ws[p]):
            p -= 1
        else:
            return False
    return False


def _agrees(other: str, value: str) -> bool:
    """An id said again: the same letters and digits, or the same digits with no letters ('MX-48213 ... 48213')."""
    o, v = re.sub(r"[\W_]", "", other).upper(), re.sub(r"[\W_]", "", value).upper()
    return o == v or (o.isdecimal() and o == re.sub(r"\D", "", v))


def _unread_exception(lows: list[str], j: int) -> bool:
    """Is the word at j one of the plain non-seal uses of _UNREAD_EXCEPTIONS ('El resto está bien', 'rest area')?"""
    return any(tuple(lows[j - k:j - k + len(p)]) == p for p in _UNREAD_EXCEPTIONS for k, w in enumerate(p)
               if w == lows[j] and j >= k)


def _seals(toks: list[tuple[str, str]], lows: list[str]) -> tuple[str | None, str, set[int], set[int], str, set[int]]:
    """Every seal mention: (seal or None, why None when one was said, token indices the mentions cover, indices where
    an id stopped reading one way, why no count may be taken: one was said inside a seal's digits)."""
    values: list[str] = []
    whys: list[str] = []
    unread: list[str] = []  # mentions with no id: "el sello es de plástico", "the seal is not broken"
    covered: set[int] = set()
    stuck: set[int] = set()  # where an id stopped reading one way: a count right after it does not either
    ends: list[int] = []  # where each mention ended
    mentions: list[int] = []  # where each mention started
    between: set[int] = set()  # words said between a keyword and its id: "el sello creo que es ..."
    count_why = ""
    k = 0
    while k < len(toks):
        if lows[k] not in _SEAL_KEYWORDS and lows[k] not in _SEAL_PLURALS:
            k += 1
            continue
        before = len(values), len(whys)
        mentions.append(k)
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
        between.update(range(i, start))
        why = why or (_unclear_letter_start(toks, lows, k, start) if start >= 0 else "")
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
            near = nxt + (lows[nxt] in _ID_ARTICLES and nxt + 1 < len(toks))  # "..., perdón, guión B", "perdón es el
            # ab quinientos ocho", "..., no, la B": the word right after the correction (past lead-ins, an article)
            if corrected and toks[near][0] != "stop" and (
                    (_seal_like(toks, lows, near) and toks[near][1] != "I") or toks[near][0] == "dash"
                    or lows[near] in _SEAL_DASH):
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
    if values and ends and (unread_at := next((j for j in range(min(ends), len(lows)) if lows[j] in _UNREAD
                                               and not _unread_exception(lows, j)), -1)) >= 0:
        whys.append(f"seal not read in full in transcript: {' '.join(lows[max(unread_at - 4, 0):unread_at + 1])!r}")
    for j, w in enumerate(lows):  # "la caja viene abierta", "the trailer door was opened": the seal is not intact
        if w in _TRAILER and _w(lows, j - 1) in _TRAILER_ARTICLES and j not in covered:
            bad = next((q for q in range(j + 1, _sentence(lows, j)[1]) if (lows[q] in _SEAL_BAD or lows[q] in _OPEN)
                        and not _negated_before(lows, q)), -1)
            if bad >= 0:
                whys.append(_reported(lows[j - 1:bad + 1]))
    for j, w in enumerate(lows):  # "it's not sealed", "sin sellar", "came unsealed"; "sealed" alone says nothing
        if w in _SEALING and j not in covered:
            if w == "unsealed" or _negated_before(lows, j):
                whys.append(_reported(lows[max(j - 3, 0):j + 1]))
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
        return None, "; ".join(whys), covered, stuck, count_why, between
    distinct = sorted({re.sub(r"[\W_]", "", v).upper() for v in values})
    disagree = sorted({re.sub(r"[\W_]", "", o).upper() for o, _, _, _ in others
                       if not any(_agrees(o, v) for v in values)})
    if values and (len(distinct) > 1 or disagree):
        why = f"seals disagree in transcript: {sorted(set(distinct + disagree))}"
        return None, why, covered, stuck, count_why, between
    if values and (back := next((j for j in range(mentions[0], len(lows)) if _retraction_at(lows, j)), -1)) >= 0:
        why = f"seal taken back in transcript: {_said(lows[back:back + 8])!r}"  # "..., ese era el de ayer"
        return None, why, covered, stuck, count_why, between
    if not values and unread:  # never silent: a seal was said and no id came out of it
        why = "seal said but no id read: " + ", ".join(repr(u) for u in unread)
        return None, why, covered, stuck, count_why, between
    if not values and others:
        said = ", ".join(repr(o) for o, _, _, _ in others)
        return None, f"id said with no seal keyword: {said}", covered, stuck, count_why, between
    return (values[0] if values else None), "", covered, stuck, count_why, between


# --- count -------------------------------------------------------------------------------------


def _negated_value(ws: list[str], s: int, e: int) -> list[str]:
    """The words that negate the number at ws[s:e], or []: a negation earlier in its clause with other words between,
    not about the cargo ('no, no son 18', 'no es 26'; 'no, trece' and 'no trece' are corrections), or one right after it
    that ends the clause ('las 34 no, ...', 'Las 11 skids no.')."""
    words, k = False, s - 1
    while k >= 0 and ws[k] not in (",", ".", "-", "|"):
        n = 2 if ws[k] == "t" and _w(ws, k - 1) in _NEGATED_CONTRACTIONS else int(ws[k] in _NEGATIONS)
        if n and words and not _about_cargo(ws, k + 1):
            return ws[k - (n == 2):e]
        f = _phrase_ending_at(ws, k + 1, _FILLER_PHRASES) or int(ws[k] in _SPEECH_FILLERS)
        words, k = words or not f, k - max(f, 1)
    n = 2 if _w(ws, e) in _NEGATED_CONTRACTIONS and _w(ws, e + 1) == "t" else int(_w(ws, e) in _NEGATIONS)
    if n and _w(ws, e + n) in ("", ".", ","):
        return ws[s:e + n]
    q = e
    while _w(ws, q) in _NEGATION_GAP or _w(ws, q) == "s":
        q += 1
    return ws[s:q + 1] if _w(ws, q) in _WRONG else []  # "the 350 is wrong"


def _phone_run(ws: list[str], s: int) -> range:
    """The indices of a phone number written in groups starting at s ('818 772 4410', '555 0142'), unless a
    correction stands right before it; else an empty range."""
    sizes = []
    while _w(ws, s + len(sizes)).isdecimal() and len(sizes) < 3:
        sizes.append(len(ws[s + len(sizes)]))
    for want in ([3, 3, 4], [3, 4]):
        if sizes[:len(want)] == want and not _w(ws, s + len(want)).isdecimal() and not _phrase_ending_at(
                ws, s, _CORRECTIONS):
            return range(s, s + len(want))
    return range(0)


def _after_seal(ws: list[str], n: int) -> bool:
    """Is the word at n said of a seal mention right before it, past only _NEGATION_GAP words ('| is missing')?"""
    p = n - 1
    while p >= 0 and ws[p] in _NEGATION_GAP:
        p -= 1
    return p >= 0 and ws[p] == "|"


def _weighed(ws: list[str], s: int) -> bool:
    """Is the number at s a weight said after a weight word, past only the words of _WEIGHT_GAP ('pesa cuatro mil',
    'peso bruto 4800', 'weighs about 4,800')?"""
    p = s - 1
    while p >= 0 and ws[p] in _WEIGHT_GAP:
        p -= 1
    return p >= 0 and ws[p] in _WEIGHT_WORDS


def _pronoun(ws: list[str], s: int, e: int) -> bool:
    """'one' as a pronoun: 'a good one', 'the last one', 'this one' (only after a word of _PRONOUN_BEFORE)."""
    return e == s + 1 and ws[s] == "one" and _w(ws, s - 1) in _PRONOUN_BEFORE


def _adverb(w: str) -> bool:
    """'honestly', 'realmente', 'tops': after a bare number it does not say what the number counts."""
    return (len(w) > 4 and w.endswith("ly")) or (len(w) > 6 and w.endswith("mente")) or w in _ADVERBS


def _article_noun(ws: list[str], e: int) -> bool:
    """Is the word at e a noun an article stands before ('un viaje', 'una pregunta'), and not a count noun, a count
    change, a number or an adverb?"""
    w = _w(ws, e)
    return (w == "|" or w in _ID_NOUNS or (_is_filler(w) and len(w) > 2 and w not in _SPEECH_FILLERS)) and not (
        _unit(ws, e) or w in _COUNT_CHANGE or w in _MORE or w in _NUMBER_WORDS or _adverb(w)
        or _phrase_at(ws, e, _COUNT_AFTER))



def _count(ws: list[str], stuck: frozenset[int] | set[int] = frozenset(),
           between: frozenset[int] | set[int] = frozenset()) -> tuple[int | None, str]:
    """The one count said in ws (seal mentions masked as '|'). stuck: where a seal id stopped reading one way; a count
    said right after one ("... one thousand five hundred eighty one hundred and nine pallets") does not either.
    between: words said between a seal keyword and its id ('el sello creo que es ...'), which are that seal's."""
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
        while p > 0:
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
            if head and head[1] < e and _w(ws, head[1]) in _THOUSAND:
                other.setdefault(head[0], f"self-correction: {marker!r}")
        if (_w(ws, s - 1) in _ID_NOUNS and _names(ws, s - 1)) or _w(ws, e) in _WEIGHT_UNITS:
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
    negated: list[list[str]] = []  # a value said again with a negation
    time_ends: set[int] = set()
    phones = {q for s, _, _ in bare for q in _phone_run(ws, s)}
    for s, e, v in bare:
        # Any number said with no unit, anywhere in the note, is a value for the count unless it numbers or weighs
        # something else ("andén cuatro", "factura nueve nueve ocho", "trailer number is 53", "a las tres", "pesa
        # cuatro mil ochocientos") or counts something else ("dos horas esperando"): "nope, uh, thirteen total"
        # disagrees with "twelve pallets" whether or not a correction word was recognized.
        if _named(ws, s) or s in named_ends:
            named_ends.add(e)
            continue
        nxt = _w(ws, e)
        article = e == s + 1 and ws[s] in _ARTICLES
        if _weighed(ws, s) or (article and nxt == "total") or _pronoun(ws, s, e) or s in phones or (
                ws[s] == "once" and _inside(ws, s, _ONCE)):
            continue  # "pesa cuatro mil", "un total de doce tarimas", "have a good one", "818 772 4410", "for once"
        if not article and (neg := _negated_value(ws, s, e)):
            negated.append(neg)  # "no, no son 18", "las 34 no", "the 350 is wrong": said again, negated
        if _w(ws, s - 1) in _NOT_COUNT_BEFORE or _phrase_ending_at(ws, s, _TIME_BEFORE) or s in time_ends or (
                _w(ws, s - 1) in _AND and s - 1 in time_ends):
            time_ends.add(e)  # "a las ocho y diez", "a las seis cuarenta y cinco", "a quarter to six"
            continue
        if (nxt in _MORE and not _phrase_at(ws, e, _APPROX_AFTER)) or nxt in _COUNT_CHANGE:
            # "y una más": 13, or 1? "one short", "una faltante", "dos missing": 12, or 11? Always a count.
            return None, f"unclear count in transcript: {' '.join(ws[s:e + 1])!r}"
        marker = correction_before(s)
        if article:
            # "Fue un viaje largo", "bueno, un saludo": an article before a noun that is neither counted nor a count
            # change. After a correction only the nouns of _ARTICLE_NOUNS: "perdón, un par", "sorry, one short"
            # compete (ORDER Nube-10 item 1b). "|" is a seal mention: "y un sello MX-48213" counts the seal.
            if (nxt in _ARTICLE_NOUNS or nxt == "|") if marker else _article_noun(ws, e):
                continue
        # A correction before a number makes it compete whatever follows it ("no, trece en total", "perdón, trece
        # completas"). Accepted cost: "perdón, tres horas tarde" competes too (count None, the detail says why).
        if marker:
            other.setdefault(v, f"self-correction: {marker!r}")
            continue
        if not _phrase_at(ws, e, _COUNT_AFTER) and _is_filler(nxt) and len(nxt) > 2 and nxt not in _SPEECH_FILLERS and (
                not _adverb(nxt)):
            continue  # "dos horas esperando", "two hours late": it counts something else ("thirteen honestly" does not)
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
        a, b = _sentence(ws, s)
        src = next((n for n in range(a, s) if _source_at(ws, n, a, b)), -1)
        affirmed = False
        for n in range(u, b):
            affirmed = affirmed or ws[n] in _SEAL_MATCH  # "twelve pallets, matches the bill"
            if src < 0 and not affirmed and _source_at(ws, n, a, b, after=True):
                src = n
        if src >= 0:
            # "según la factura son doce tarimas", "the bol says fourteen ...", "doce tarimas, eso dice el papel",
            # "En la planta me contaron doce tarimas"
            return None, f"count said from the paperwork or someone else: {_said(ws[min(src, s):max(src + 1, u)])!r}"
        if any(_phrase_at(ws, n, _COUNT_DOUBT) for n in range(len(ws))):  # "no sé, no las conté"
            return None, f"count not checked in transcript: {said!r}"
        thousands = next((n for n in range(s, u) if ws[n] in _THOUSAND), -1)
        if thousands >= 0 and (_WEIGHT_WORDS & set(ws[a:s]) or (
                v is not None and v >= 10000 and _number_at(ws, thousands + 1, dozens=False))):
            # "pesa veinte mil doce tarimas", "forty thousand twelve pallets": a weight and a count, or one number?
            return None, f"count runs into a weight: {_said(ws[a:u])!r}"
        if any(p < s and all(w in _NUMBER_WORDS or w in _AND for w in ws[p:s]) for p in stuck):
            return None, f"count runs into the seal's digits: {' '.join(ws[min(stuck, key=lambda p: abs(s - p)):u])!r}"
        q = u + 1 if _w(ws, u) == "," else u
        if _phrase_ending_at(ws, s, _APPROX_BEFORE) or _phrase_at(ws, q, _APPROX_AFTER):
            return None, f"approximate count in transcript: {said!r}"
        # a hedge anywhere in its sentence: "Creo que lo que traigo son doce tarimas", "..., creo"
        h, he = _hedge_in(ws, a, b, between)
        if h >= 0:
            return None, f"hedged count in transcript: {_said(ws[h:u] if h < s else ws[s:he])!r}"
        for k in range(s - 1, a - 1, -1):  # a negation in its clause before it: "no traigo doce tarimas"
            if ws[k] in (",", "-", "|"):
                break
            n = 2 if ws[k] == "t" and _w(ws, k - 1) in _NEGATED_CONTRACTIONS else int(ws[k] in _NEGATIONS)
            if n and not _about_cargo(ws, k + 1):
                return None, f"negated count in transcript: {_said(ws[k - (n == 2):u])!r}"
        n = 2 if _w(ws, u) in _NEGATED_CONTRACTIONS and _w(ws, u + 1) == "t" else int(_w(ws, u) in _NEGATIONS)
        if n and _w(ws, u + n) in ("", ".", ","):
            negated.append(ws[s:u + n])  # "Las 11 skids no, I miscounted"
        q = u
        while _w(ws, q) in _NEGATION_GAP or _w(ws, q) == "s":
            q += 1
        if _w(ws, q) in _WRONG:
            negated.append(ws[s:q + 1])  # "twelve pallets is wrong"
        # said after it in its sentence, more than the count: "doce tarimas más una suelta", "and a box", "y media"
        for n in range(u, b):
            if _phrase_at(ws, n, _CORRECTIONS) or _phrase_at(ws, n, _APPROX_AFTER):
                break  # "doce tarimas, más bien trece", "más o menos": said for what they are
            seal_next = ws[n] in _SEAL_BAD and (_w(ws, n + 1) == "|" or (
                _w(ws, n + 1) in _ID_ARTICLES | _EN_INDEFINITE and _w(ws, n + 2) == "|") or _after_seal(ws, n))
            if (ws[n] in _ADDED and not _negated_before(ws, n) and not seal_next) or (
                    ws[n] in _INDEFINITE and (
                    _unit(ws, n + 1) or (_is_filler(_w(ws, n + 1)) and _unit(ws, n + 2)))):
                # not "no falta nada", "nothing missing", nor "falta el sello": that is the seal's
                return None, f"more than the count said in transcript: {_said(ws[s:n + 1])!r}"
        if v is None:
            return None, f"unclear count in transcript: {said!r}"

    values = {v for _, _, v in counts if v is not None} | set(other)
    taken_back = next((r for r in range(min(s for s, _, _ in counts), len(ws)) if _retraction_at(ws, r)), -1)
    if len(values) == 1 and negated:  # one value, said again negated: "Son 18 tarimas, no, no son 18"
        return None, f"negated count in transcript: {_said(negated[0])!r}"
    if len(values) == 1 and taken_back >= 0:  # "Doce tarimas. Mentira.", "..., no, no son 18, me equivoqué"
        return None, f"count taken back in transcript: {_said(ws[taken_back:taken_back + 8])!r}"
    if len(values) == 1:
        return values.pop(), ""
    notes = sorted({n for n in other.values() if n})
    why = f"counts disagree in transcript: {sorted(values)}"
    return None, why + (f" ({'; '.join(notes)})" if notes else "")


def _parse_voice(transcript: str) -> tuple[int | None, str, str | None, str]:
    toks = _tokens(transcript)
    lows = [_LOW_PUNCT.get(k, t.lower()) for k, t in toks]
    seal, seal_why, covered, stuck, inside, between = _seals(toks, lows)
    count, why = _count(["|" if k in covered else w for k, w in enumerate(lows)], stuck, between)
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
