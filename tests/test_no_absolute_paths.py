"""Guard: nothing under receipts/ or tests/data/ may carry a path from the machine it was made on.

Same class as the numbers guard: a receipt is evidence for everyone, not a map of someone's laptop.
"""

from pathlib import Path

from tinytms.receipt import MACHINE_PATH_PREFIXES

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("receipts", "tests/data")


def offending_files(root: Path, dirs: tuple[str, ...] = SCAN_DIRS) -> list[tuple[str, str]]:
    bad: list[tuple[str, str]] = []
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue  # binary (image, audio): not a place for paths
            for prefix in MACHINE_PATH_PREFIXES:
                if prefix in text:
                    bad.append((p.relative_to(root).as_posix(), prefix))
    return bad


def test_guard_bites(tmp_path):
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "r.json").write_text('{"command": "/Users/someone/x"}', encoding="utf-8")
    assert offending_files(tmp_path) == [("receipts/r.json", "/Users/")]


def test_guard_ignores_binaries_and_clean_text(tmp_path):
    data = tmp_path / "tests" / "data"
    data.mkdir(parents=True)
    (data / "a.bin").write_bytes(b"\xff\xfe/Users/")
    (data / "ok.json").write_text('{"image": "evals/public/x.jpg"}', encoding="utf-8")
    assert offending_files(tmp_path) == []


def test_repo_receipts_and_test_data_are_clean():
    bad = offending_files(ROOT)
    assert bad == [], "machine paths found:\n" + "\n".join(f"{f}: {p}" for f, p in bad)
