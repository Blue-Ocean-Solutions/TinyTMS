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

# "payets", "pajets", "payettes": how Muse Voice Transcribe wrote "pallets" said in Spanish, three runs over the
# same audio (real transcripts, tests/data/real-voice-es-*).
_UNITS = (
    r"pallets?|palets?|pal[eé]s|payets?|pajets?|payettes?|paletas?|tarimas?|skids?|cajas?|boxes|box|bultos?|"
    r"piezas?|pieces?"
)
# A spoken dash inside a seal id ("MX guión cuatro ocho...", same real transcript); "guion" is the RAE 2010 spelling.
_SEAL_DASH = {"guión", "guion"}
_WORDS: dict[str, int] = {
    # es
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintiún": 21, "veintidós": 22, "veintidos": 22, "veintitrés": 23, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiséis": 26, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30, "cuarenta": 40,
    "cincuenta": 50,
    # en
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}
_DIGIT_WORDS = {k: v for k, v in _WORDS.items() if v <= 9 and k not in ("un", "una")}
_NUM = r"\d+|[a-záéíóúñ]+(?:[\s-]+(?:y\s+)?[a-záéíóúñ]+)?"


def _word_number(s: str) -> int | None:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    toks = [t for t in re.split(r"[\s-]+", s) if t and t != "y"]
    if not toks or any(t not in _WORDS for t in toks):
        return None
    if len(toks) == 1:
        return _WORDS[toks[0]]
    tens, ones = _WORDS[toks[0]], _WORDS[toks[1]]
    if len(toks) == 2 and tens >= 20 and tens % 10 == 0 and 1 <= ones <= 9:
        return tens + ones
    return None


def parse_count(transcript: str) -> tuple[int | None, str]:
    """Unique number directly followed (within one word) by a unit noun; else None + why."""
    text = transcript.lower()
    found: set[int] = set()
    for m in re.finditer(rf"\b({_NUM})\s+(?:[a-záéíóúñ]+\s+)?(?:{_UNITS})\b", text):
        phrase = m.group(1)
        n = _word_number(phrase)
        if n is None:  # e.g. "son doce" -> try the last word alone
            n = _word_number(phrase.split()[-1])
        if n is not None:
            found.add(n)
    if len(found) == 1:
        return found.pop(), ""
    if not found:
        return None, "no count found in transcript"
    return None, f"ambiguous counts in transcript: {sorted(found)}"


def parse_seal(transcript: str) -> str | None:
    """Tokens after 'sello'/'seal'/'precinto' that look like a seal id (digit words -> digits)."""
    m = re.search(
        r"\b(?:sello|seal|precinto|marchamo)\b(?:\s+(?:number|no\.?|n[uú]mero|num\.?|#))?"
        r"(?:\s+(?:es|is|:))?\s*:?\s*(.+)",
        transcript,
        flags=re.I,
    )
    if not m:
        return None
    out: list[str] = []
    for tok in re.split(r"[\s,]+", m.group(1)):
        t = tok.strip(".;:")
        low = t.lower()
        if low in _DIGIT_WORDS:
            out.append(str(_DIGIT_WORDS[low]))
        elif low in _SEAL_DASH and out:
            out.append("-")
        elif re.fullmatch(r"[A-Za-z0-9\-]+", t) and (any(ch.isdigit() for ch in t) or (t.isupper() and len(t) <= 3)):
            # "MX MX guión ..." (real transcript): a letter group repeated back to back is one ASR stutter.
            # Letters only: "cuatro cuatro" stays 44. Residual risk, accepted and written down: a driver who
            # really says "MX MX 48213" against a paper reading "MX-48213" would match (a "MXMX" paper HOLDs).
            if not (t.isalpha() and out and out[-1] == t):
                out.append(t)
        else:
            break
        if tok.endswith((".", ";")):
            break
    seal = "".join(out)
    return seal if any(ch.isdigit() for ch in seal) else None


def voice_from_transcript(
    data: dict[str, Any], *, model: str, provider: str, source_sha256: str
) -> VoiceWitness:
    meta = dict(model=model, provider=provider, source_sha256=source_sha256)
    """Measured ASR shape: {sessionId, transcript, audioDurationMs, turns[]}. Empty transcript -> OK, count None."""
    if not isinstance(data, dict) or not isinstance(data.get("transcript"), str):
        keys = sorted(data)[:12] if isinstance(data, dict) else type(data).__name__
        return VoiceWitness(status="ERROR", detail=f"no 'transcript' string; keys={keys}", **meta)
    text = data["transcript"]
    count, why = parse_count(text)
    if not text.strip():
        why = "empty transcript"
    lang = data.get("language") if isinstance(data.get("language"), str) else None
    dur = data.get("audioDurationMs") if _is_int(data.get("audioDurationMs")) else None
    return VoiceWitness(
        status="OK", transcript=text, language=lang, count=count, seal=parse_seal(text), detail=why,
        audio_duration_ms=dur, **meta
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
