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


TEST_SK = SigningKey(bytes(range(32)))  # a test seat: its receipts verify only where TEST_KNOWN is passed
TEST_KNOWN = {TEST_SK.verify_key.encode().hex(): "test seat"}


def _signed(tmp: Path, name: str, body: dict) -> Path:
    p = tmp / name
    p.write_bytes(canonical_json(sign_receipt(body, TEST_SK)) + b"\n")
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
    text = table.render(tmp_path, TEST_KNOWN)
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
    first, second = table.render(a, TEST_KNOWN), table.render(a, TEST_KNOWN)
    assert first == second == table.render(b, TEST_KNOWN)
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
    text = table.render(tmp_path, TEST_KNOWN)
    assert "/Users/" not in text and "/tmp/" not in text
    assert find_machine_paths(table.render(RECEIPTS)) == []


def test_check_mode_fails_on_a_stale_file(tmp_path):
    _signed(tmp_path, "live-1.json", _live_body("2026-10-28T10:00:00.000000Z"))
    out, signers = tmp_path / "MEASURED.md", tmp_path / "signers.txt"
    signers.write_text(f"{next(iter(TEST_KNOWN))}  # test seat\n", encoding="utf-8")
    args = ["--receipts", str(tmp_path), "--out", str(out), "--known-signers", str(signers)]
    assert table.main(args) == 0 and "0.5 s" in out.read_text(encoding="utf-8")
    assert table.main([*args, "--check"]) == 0
    out.write_text(out.read_text(encoding="utf-8").replace("0.5", "0.4"), encoding="utf-8")  # a hand edit
    assert table.main([*args, "--check"]) == 1


def test_a_valid_signature_by_an_unknown_key_is_not_verified(tmp_path):
    """L1's forgery (2026-10-30): a real receipt's body re-signed with a fresh key. The signature checks, the signer
    is not in keys/known_signers.txt, so the row goes to "Not verified" as an unknown signer, never into the table."""
    src = json.loads((RECEIPTS / "20261028T153041947742Z-HOLD.json").read_bytes())
    body = {k: v for k, v in src.items() if k not in ("signature", "public_key")}
    forged = sign_receipt(body, SigningKey.generate())
    (tmp_path / "20261028T153041947742Z-HOLD.json").write_bytes(canonical_json(forged) + b"\n")
    assert verify_receipt(canonical_json(forged))  # integrity holds: this is about origin
    rows = table.collect(tmp_path)
    assert [(r.name, r.verified) for r in rows] == [("20261028T153041947742Z-HOLD.json", False)]
    verified, _, not_verified = table.render(tmp_path).partition("## Not verified")
    assert "](../receipts/20261028T153041947742Z-HOLD.json)" in not_verified
    assert "](../receipts/20261028T153041947742Z-HOLD.json)" not in verified
    assert "unknown signer" in not_verified and forged["public_key"][:16] in not_verified
