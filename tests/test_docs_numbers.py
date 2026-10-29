"""Numbers guard for every published Markdown (README.md and docs/): a figure must cite an existing receipt."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("check_docs_numbers", ROOT / "scripts" / "check_docs_numbers.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _root_with_receipts(tmp_path: Path, *names: str) -> Path:
    (tmp_path / "receipts").mkdir()
    for n in names:
        (tmp_path / "receipts" / n).write_text("{}", encoding="utf-8")
    return tmp_path


def test_flags_each_unit_without_a_receipt(tmp_path):
    m = _script()
    for line in ("7.9 ms per frame", "Runs at 126 FPS", "99% accurate", "40 tok/s", "loads in 1.2 s"):
        assert [n for n, _ in m.problems(line, tmp_path)] == [1], line


def test_receipt_on_the_same_line_passes(tmp_path):
    m = _script()
    root = _root_with_receipts(tmp_path, "local-vlm-1.json", "live-sam-a.json")
    assert m.problems("30 tok/s (`receipts/local-vlm-1.json`)", root) == []
    assert m.problems("SAM answered in 0.47 s, [receipt](../receipts/live-sam-a.json).", root) == []
    assert m.problems("every SAM call took under 2 s: `receipts/live-sam-*.json`", root) == []


def test_receipt_on_another_line_does_not_count(tmp_path):
    m = _script()
    root = _root_with_receipts(tmp_path, "r.json")
    assert [n for n, _ in m.problems("see receipts/r.json\n40 tok/s on metal", root)] == [2]


def test_bare_folder_or_missing_file_does_not_count(tmp_path):
    m = _script()
    root = _root_with_receipts(tmp_path, "r.json")
    assert m.problems("40 tok/s, see `receipts/`", root)
    assert m.problems("40 tok/s (receipts/local-vlm-20261030T120000Z.json)", root)
    assert m.problems("40 tok/s (receipts/nothing-*.json)", root)
    assert m.problems("40 tok/s (receipts/r.json, receipts/typo.json)", root), "every cited receipt must exist"


def test_plain_text_and_hashes_are_not_figures(tmp_path):
    m = _script()
    for line in ("12 pieces, 4800 kg, seal MX-48213", "the three sha256s", "in 5 steps", "Ed25519 receipts", "100 GB"):
        assert m.problems(line, tmp_path) == [], line


def test_main_exit_codes_and_report(tmp_path, capsys):
    m = _script()
    root = _root_with_receipts(tmp_path, "r.json")
    (root / "README.md").write_text("Fast: 3 ms (receipts/r.json)\n", encoding="utf-8")
    (root / "docs" / "sub").mkdir(parents=True)
    (root / "docs" / "ok.md").write_text("12 pallets on paper\n", encoding="utf-8")
    assert m.main(["--root", str(root)]) == 0
    (root / "docs" / "sub" / "bad.md").write_text("intro\nRuns at 99%\n", encoding="utf-8")
    assert m.main(["--root", str(root)]) == 1
    assert "docs/sub/bad.md:2:" in capsys.readouterr().out


def test_the_committed_docs_pass():
    m = _script()
    files = m.markdown_files(ROOT)
    names = {p.relative_to(ROOT).as_posix() for p in files}
    assert {"README.md", "docs/SUBMISSION.md", "docs/VIDEO-OUTLINE.md"} <= names
    assert m.main(["--root", str(ROOT)]) == 0


def test_ci_runs_the_checker():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python scripts/check_docs_numbers.py" in ci
