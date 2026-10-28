"""Regenerate fixtures/: hand-written FIXTURE witnesses + tiny placeholder media (1x1 PNG, silent WAV).

Nothing here is a model output. Run from the repo root: python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import struct
import wave
import zlib
from pathlib import Path

PROV = "FIXTURE hand-written 2026-10-28, not a model output"
ROOT = Path(__file__).resolve().parent.parent / "fixtures"

# name: (vision count, doc fields, driver transcript, language)
CASES = {
    "accept": (12, {"pieces": 12, "weight_kg": 8450.5, "seal": "MX-48213", "hazmat": False},
               "Traigo doce tarimas, el sello es MX-48213.", "es"),
    "count_mismatch": (11, {"pieces": 12, "weight_kg": 8450.5, "seal": "MX-48213", "hazmat": False},
                       "Son doce tarimas y el sello es MX-48213.", "es"),
    "seal_mismatch": (12, {"pieces": 12, "weight_kg": 8450.5, "seal": "MX-48213", "hazmat": False},
                      "Traigo 12 tarimas, sello MX-48231.", "es"),
    "overweight": (12, {"pieces": 12, "weight_kg": 26500.0, "seal": "MX-48213", "hazmat": False},
                   "Twelve pallets, seal number MX-48213.", "en"),
}


def png(path: Path, rgb: tuple[int, int, int]) -> None:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" + bytes(rgb))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def wav(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)


def dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    for i, (name, (count, doc, transcript, lang)) in enumerate(CASES.items()):
        d = ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        note = f"case {name}: synthetic data, placeholder media"
        dump(d / "vision.FIXTURE.json",
             {"witness": "vision", "provenance": PROV, "note": note, "concept": "pallet", "count": count})
        dump(d / "doc.FIXTURE.json",
             {"witness": "doc", "provenance": PROV, "note": note, **doc,
              "shipper": "Transportes Ejemplo SA de CV (synthetic)",
              "consignee": "Bodega Demo LLC (synthetic)"})
        dump(d / "voice.FIXTURE.json",
             {"witness": "voice", "provenance": PROV, "note": note, "language": lang, "transcript": transcript})
        png(d / "load.FIXTURE.png", (40 * i, 120, 60))
        png(d / "doc.FIXTURE.png", (250, 250, 250 - i))
        wav(d / "voice.FIXTURE.wav")


if __name__ == "__main__":
    main()
