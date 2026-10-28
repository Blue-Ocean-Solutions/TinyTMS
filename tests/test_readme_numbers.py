"""Numbers policy guard: any README line quoting ms / FPS / % / tok/s must cite receipts/."""

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
NUMBER = re.compile(r"\d+(\.\d+)?\s*(ms|FPS|fps|%|tok/s)")


def offending_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if NUMBER.search(line) and "receipts/" not in line]


def test_guard_bites():
    assert offending_lines("Runs at 126 FPS on metal") == ["Runs at 126 FPS on metal"]
    assert offending_lines("7.9 ms per frame") and offending_lines("99% accurate") and offending_lines("40 tok/s")
    assert offending_lines("40 tok/s (receipts/local-vlm-20261030T120000Z.json)") == []


def test_readme_has_no_unreceipted_numbers():
    bad = offending_lines(README.read_text(encoding="utf-8"))
    assert bad == [], "README quotes performance numbers without a receipts/ link:\n" + "\n".join(bad)
