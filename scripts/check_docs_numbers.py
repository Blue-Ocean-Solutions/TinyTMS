"""Numbers guard for every Markdown we publish: README.md and docs/**/*.md.

    python scripts/check_docs_numbers.py [--root .]      # exit 0 clean, 1 with the offending lines

The README guard (tests/test_readme_numbers.py) extended to docs/ and made a little stricter: a line that quotes
a time (ms or s), frame rate, percentage or token rate must cite a file in receipts/ on the same line, and every
receipts/ path cited on that line must exist (a glob such as receipts/live-sam-*.json must match a file).
A bare `receipts/` or a receipt that is not in the tree is not a citation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NUMBER = re.compile(r"\d+(\.\d+)?\s*(ms|FPS|fps|%|tok/s)")  # same pattern as tests/test_readme_numbers.py
SECONDS = re.compile(r"(?<![\w.])\d+(\.\d+)?\s?s\b")
RECEIPT = re.compile(r"receipts/([A-Za-z0-9._*-]*)")


def problems(text: str, root: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not (NUMBER.search(line) or SECONDS.search(line)):
            continue
        names = [m.group(1).rstrip(".") for m in RECEIPT.finditer(line)]
        names = list(dict.fromkeys(name for name in names if name))  # a Markdown link names its file twice
        if not names:
            out.append((n, "figure without a receipts/ file on the same line"))
            continue
        missing = [name for name in names if not any((root / "receipts").glob(name))]
        if missing:
            out.append((n, "cites receipts that are not in the tree: " + ", ".join(f"receipts/{x}" for x in missing)))
    return out


def markdown_files(root: Path) -> list[Path]:
    files = [root / "README.md"] if (root / "README.md").is_file() else []
    return files + sorted(p for p in (root / "docs").rglob("*.md") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args(argv)
    root = Path(args.root)
    bad = 0
    for path in markdown_files(root):
        lines = path.read_text(encoding="utf-8").splitlines()
        for n, why in problems("\n".join(lines), root):
            print(f"{path.relative_to(root).as_posix()}:{n}: {why}: {lines[n - 1].strip()}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
