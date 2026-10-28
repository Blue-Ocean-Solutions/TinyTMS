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
    """Text from a Responses-style payload: output_text, or output[].content[].text."""
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        for c in item.get("content") or []:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "".join(parts) if parts else None


def parse_json_text(text: str) -> Any:
    """json.loads tolerant of a ```json fence around the object."""
    t = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, flags=re.S)
    return json.loads(m.group(1) if m else t)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# --- vision --------------------------------------------------------------------------------

_DETECTION_KEYS = ("detections", "instances", "masks", "objects")


def vision_from_sam(
    data: dict[str, Any], *, model: str, provider: str, source_sha256: str, concept: str = "pallet"
) -> VisionWitness:
    meta = dict(model=model, provider=provider, source_sha256=source_sha256, concept=concept)
    if _is_int(data.get("count")):
        return VisionWitness(status="OK", count=data["count"], boxes=_boxes(data), **meta)
    # Look for a detection list at top level or inside output items.
    candidates: list[dict[str, Any]] = [data] + [
        i for i in (data.get("output") or []) if isinstance(i, dict)
    ]
    for c in candidates:
        for k in _DETECTION_KEYS:
            if isinstance(c.get(k), list):
                return VisionWitness(
                    status="OK", count=len(c[k]), boxes=_boxes({"boxes": _det_boxes(c[k])}), **meta
                )
    return VisionWitness(
        status="ERROR",
        detail=f"unrecognized SAM response shape; top-level keys={sorted(data)[:12]}",
        **meta,
    )


def _det_boxes(dets: list[Any]) -> list[Any]:
    out = []
    for d in dets:
        if isinstance(d, dict):
            b = d.get("box") or d.get("bbox")
            if b is not None:
                out.append(b)
    return out


def _boxes(data: dict[str, Any]) -> tuple[tuple[float, float, float, float], ...]:
    out = []
    for b in data.get("boxes") or []:
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
            out.append(tuple(float(x) for x in b))
    return tuple(out)  # type: ignore[return-value]


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

_UNITS = (
    r"pallets?|palets?|pal[eé]s|paletas?|tarimas?|skids?|cajas?|boxes|box|bultos?|piezas?|pieces?"
)
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
        elif re.fullmatch(r"[A-Za-z0-9\-]+", t) and (any(ch.isdigit() for ch in t) or (t.isupper() and len(t) <= 3)):
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
    text = data.get("transcript") if isinstance(data.get("transcript"), str) else None
    if text is None and isinstance(data.get("text"), str):
        text = data["text"]
    if text is None:
        text = output_text(data)
    if text is None:
        return VoiceWitness(status="ERROR", detail=f"no transcript; keys={sorted(data)[:12]}", **meta)
    count, why = parse_count(text)
    lang = data.get("language") if isinstance(data.get("language"), str) else None
    return VoiceWitness(
        status="OK", transcript=text, language=lang, count=count, seal=parse_seal(text), detail=why, **meta
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
        w = vision_from_sam(data, concept=data.get("concept", "pallet"), **meta)
    elif kind == "doc":
        w = doc_from_fields(data, **meta)
    else:
        w = voice_from_transcript(data, **meta)
    return type(w)(**{**_fields(w), "detail": "; ".join(x for x in (note, w.detail) if x)})


def _fields(w: Witness) -> dict[str, Any]:
    from dataclasses import fields

    return {f.name: getattr(w, f.name) for f in fields(w)}
