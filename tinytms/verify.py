"""The judge: deterministic, fail-closed three-way check. Models explain; this decides.

Rule order (fixed):
  1. every witness OK, else HOLD with NO_MODEL:<w> / ERROR:<w> (stops here: nothing to compare)
  2. doc.pieces == vision.count == voice.count, else COUNT_MISMATCH (missing count -> <W>_NO_COUNT)
  3. doc.seal and voice.seal present and equal, else SEAL_MISSING / SEAL_MISMATCH
  4. doc.weight_kg <= limits.max_weight_kg, else OVERWEIGHT (missing weight -> WEIGHT_MISSING)
  5. doc.hazmat is True -> HAZMAT_REVIEW
  6. no reasons -> ACCEPT
Rules 2-5 all run so the HOLD lists every discrepancy, not just the first.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from .witness import DocWitness, VisionWitness, VoiceWitness

Decision = Literal["ACCEPT", "HOLD"]
Check = Literal["pass", "fail", "skipped"]

DEFAULT_MAX_WEIGHT_KG = 24000.0


@dataclass(frozen=True)
class Limits:
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reasons: tuple[str, ...]
    checks: dict[str, Check] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"decision": self.decision, "reasons": list(self.reasons), "checks": dict(self.checks)}


def normalize_seal(seal: str | None) -> str | None:
    """Uppercase alphanumerics only, so 'mx-48 213' == 'MX48213'. Empty -> None."""
    if seal is None:
        return None
    s = re.sub(r"[^0-9A-Za-z]", "", str(seal)).upper()
    return s or None


def _valid_count(n: object) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def judge(
    vision: VisionWitness, doc: DocWitness, voice: VoiceWitness, limits: Limits = Limits()
) -> Verdict:
    checks: dict[str, Check] = {
        "witnesses": "pass",
        "count": "skipped",
        "seal": "skipped",
        "weight": "skipped",
        "hazmat": "skipped",
    }

    # 1. witnesses
    reasons: list[str] = []
    for name, w in (("vision", vision), ("doc", doc), ("voice", voice)):
        if w.status != "OK":
            tag = "NO_MODEL" if w.status == "NO_MODEL" else "ERROR"
            reasons.append(f"{tag}:{name}")
    if reasons:
        checks["witnesses"] = "fail"
        return Verdict("HOLD", tuple(reasons), checks)

    # 2. count
    count_reasons: list[str] = []
    if not _valid_count(doc.pieces):
        count_reasons.append("DOC_NO_PIECES")
    if not _valid_count(vision.count):
        count_reasons.append("VISION_NO_COUNT")
    if not _valid_count(voice.count):
        count_reasons.append("VOICE_NO_COUNT")
    if not count_reasons and not (doc.pieces == vision.count == voice.count):
        count_reasons.append(
            f"COUNT_MISMATCH doc={doc.pieces} vision={vision.count} voice={voice.count}"
        )
    checks["count"] = "fail" if count_reasons else "pass"
    reasons += count_reasons

    # 3. seal
    ds, vs = normalize_seal(doc.seal), normalize_seal(voice.seal)
    if ds is None or vs is None:
        missing = ",".join(n for n, s in (("doc", ds), ("voice", vs)) if s is None)
        reasons.append(f"SEAL_MISSING {missing}")
        checks["seal"] = "fail"
    elif ds != vs:
        reasons.append(f"SEAL_MISMATCH doc={ds} voice={vs}")
        checks["seal"] = "fail"
    else:
        checks["seal"] = "pass"

    # 4. weight (fail-closed on missing / non-finite)
    w = doc.weight_kg
    if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w) or w < 0:
        reasons.append("WEIGHT_MISSING")
        checks["weight"] = "fail"
    elif w > limits.max_weight_kg:
        reasons.append(f"OVERWEIGHT doc={w:g}kg max={limits.max_weight_kg:g}kg")
        checks["weight"] = "fail"
    else:
        checks["weight"] = "pass"

    # 5. hazmat
    if doc.hazmat is True:
        reasons.append("HAZMAT_REVIEW")
        checks["hazmat"] = "fail"
    else:
        checks["hazmat"] = "pass"

    # 6. decision
    return Verdict("HOLD" if reasons else "ACCEPT", tuple(reasons), checks)
