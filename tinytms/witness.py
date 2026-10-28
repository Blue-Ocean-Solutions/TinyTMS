"""Witness records. A witness reports what one model saw; it never decides."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["OK", "NO_MODEL", "ERROR"]
Provider = Literal["meta-api", "local-mlx", "FIXTURE", "none"]

STATUSES: tuple[str, ...] = ("OK", "NO_MODEL", "ERROR")


@dataclass(frozen=True, kw_only=True)
class _Witness:
    status: Status
    model: str = ""
    provider: Provider = "none"
    source_sha256: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"bad witness status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # tuples -> lists so the dict is JSON-canonical
        return {k: (_lists(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def _lists(v: Any) -> Any:
    return [_lists(x) for x in v] if isinstance(v, (tuple, list)) else v


@dataclass(frozen=True, kw_only=True)
class VisionWitness(_Witness):
    """SAM-style concept segmentation of the load photo."""

    concept: str = "pallet"
    count: int | None = None
    # (x1, y1, x2, y2) or, from SAM 3.1, (x1, y1, x2, y2, confidence); pixels of the image that was sent
    boxes: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    image_w: int | None = None
    image_h: int | None = None


@dataclass(frozen=True, kw_only=True)
class DocWitness(_Witness):
    """Structured fields read from the BOL / Carta de Porte."""

    pieces: int | None = None
    weight_kg: float | None = None
    seal: str | None = None
    shipper: str | None = None
    consignee: str | None = None
    hazmat: bool | None = None


@dataclass(frozen=True, kw_only=True)
class VoiceWitness(_Witness):
    """Driver voice note: transcript plus count/seal parsed from it."""

    transcript: str = ""
    language: str | None = None
    count: int | None = None
    seal: str | None = None
    audio_duration_ms: int | None = None


Witness = VisionWitness | DocWitness | VoiceWitness


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
