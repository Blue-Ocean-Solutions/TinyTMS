"""docs/MEASURED.md is generated from receipts/ by scripts/receipts_table.py and must never drift from them.

Every receipt shows up exactly once: verified (signature checked with the public key the receipt carries) in the
main table, or in "not verified" with the reason. Nothing is dropped silently. Output is deterministic.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from tinytms.receipt import canonical_json, find_machine_paths, sign_receipt, verify_receipt

REPO = Path(__file__).resolve().parent.parent
RECEIPTS = REPO / "receipts"
MEASURED = REPO / "docs" / "MEASURED.md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


table = _load("receipts_table", REPO / "scripts" / "receipts_table.py")
docs_guard = _load("check_docs_numbers", REPO / "scripts" / "check_docs_numbers.py")


def _signed(tmp: Path, name: str, body: dict) -> Path:
    p = tmp / name
    p.write_bytes(canonical_json(sign_receipt(body, SigningKey(bytes(range(32))))) + b"\n")
    return p


def _live_body(utc: str, s: float = 0.5) -> dict:
    return {"kind": "live-witness", "witness": "sam", "model": "sam-3.1", "provider": "meta-api", "http": 200,
            "s": s, "utc": utc, "result": {"status": "OK", "count": 3, "concept": "pallet"}}


def test_committed_table_is_not_stale():
    """The anti-stale check CI runs: regenerate from receipts/ and compare byte for byte."""
    assert MEASURED.is_file(), "docs/MEASURED.md missing: run python scripts/receipts_table.py"
    assert MEASURED.read_text(encoding="utf-8") == table.render(RECEIPTS), \
        "docs/MEASURED.md is stale: run python scripts/receipts_table.py"


def test_every_repo_receipt_is_listed_exactly_once():
    text = table.render(RECEIPTS)
    for p in sorted(RECEIPTS.glob("*.json")):
        assert text.count(f"](../receipts/{p.name})") == 1, p.name
    rows = table.collect(RECEIPTS)
    assert {r.name for r in rows} == {p.name for p in RECEIPTS.glob("*.json")}
    for r in rows:  # verified means what verify_receipt says, with the receipt's own key
        assert r.verified == verify_receipt((RECEIPTS / r.name).read_bytes()), r.name
    assert [r.name for r in rows if not r.verified] == ["llamadas.json"]  # the call ledger is not signed


def test_a_tampered_receipt_lands_in_not_verified(tmp_path):
    good = _signed(tmp_path, "live-a.json", _live_body("2026-10-28T10:00:00.000000Z"))
    bad = tmp_path / "live-b.json"
    raw = good.read_bytes()
    i = raw.index(b'"count":3') + len(b'"count":')
    bad.write_bytes(raw[:i] + b"4" + raw[i + 1:])  # one byte of the signed body: 3 -> 4
    assert json.loads(bad.read_bytes())["result"]["count"] == 4 and not verify_receipt(bad.read_bytes())
    text = table.render(tmp_path)
    verified, _, not_verified = text.partition("## Not verified")
    assert "](../receipts/live-a.json)" in verified and "](../receipts/live-a.json)" not in not_verified
    assert "](../receipts/live-b.json)" in not_verified and "](../receipts/live-b.json)" not in verified
    assert "signature does not verify" in not_verified


@pytest.mark.parametrize("content,why", [
    (b"{not json", "not JSON"),
    (b"[1, 2]", "not a JSON object"),
    (b'{"kind": "checkin", "utc": "2026-10-28T10:00:00Z"}', "no signature"),
])
def test_unreadable_or_unsigned_files_are_listed_with_the_reason(tmp_path, content, why):
    (tmp_path / "x.json").write_bytes(content)
    _, _, not_verified = table.render(tmp_path).partition("## Not verified")
    assert "](../receipts/x.json)" in not_verified and why in not_verified


def test_output_is_byte_identical_and_sorted_by_utc(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for d, order in ((a, (3, 1, 2)), (b, (2, 3, 1))):  # same receipts, written in a different order
        for n in order:
            _signed(d, f"live-{n}.json", _live_body(f"2026-10-28T1{n}:00:00.000000Z", s=n / 10))
    shutil.copy(RECEIPTS / "llamadas.json", a / "llamadas.json")
    shutil.copy(RECEIPTS / "llamadas.json", b / "llamadas.json")
    first, second = table.render(a), table.render(a)
    assert first == second == table.render(b)
    assert first.index("live-1.json") < first.index("live-2.json") < first.index("live-3.json")


def test_generated_table_passes_the_docs_numbers_guard():
    text = table.render(RECEIPTS)
    assert docs_guard.problems(text, REPO) == []
    assert any(docs_guard.SECONDS.search(line) for line in text.splitlines()), "expected timings in the table"


def test_a_machine_path_cannot_reach_a_receipt_or_the_table(tmp_path):
    """The existing guard still holds: sign_receipt redacts machine paths (and counts them) before signing."""
    body = {**_live_body("2026-10-28T10:00:00.000000Z"), "command": "python /Users/someone/x.py --out /tmp/y"}
    p = _signed(tmp_path, "live-path.json", body)
    receipt = json.loads(p.read_bytes())
    assert find_machine_paths(receipt) == [] and receipt["redacted_machine_paths"] == 1
    assert verify_receipt(p.read_bytes())
    text = table.render(tmp_path)
    assert "/Users/" not in text and "/tmp/" not in text
    assert find_machine_paths(table.render(RECEIPTS)) == []


def test_check_mode_fails_on_a_stale_file(tmp_path):
    _signed(tmp_path, "live-1.json", _live_body("2026-10-28T10:00:00.000000Z"))
    out = tmp_path / "MEASURED.md"
    assert table.main(["--receipts", str(tmp_path), "--out", str(out)]) == 0
    assert table.main(["--receipts", str(tmp_path), "--out", str(out), "--check"]) == 0
    out.write_text(out.read_text(encoding="utf-8").replace("0.5", "0.4"), encoding="utf-8")  # a hand edit
    assert table.main(["--receipts", str(tmp_path), "--out", str(out), "--check"]) == 1
