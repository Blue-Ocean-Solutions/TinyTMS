"""receipts/INDEX.md and GET /receipts/: every signed receipt in one table, verified, with its signer named.

Receipts are made here with throwaway keys in tmp dirs; the real tree is only read. No network, no model."""

import importlib.util
import io
import json
import socket
import subprocess
from pathlib import Path
from wsgiref.util import setup_testing_defaults

import httpx
import pytest
from nacl.signing import SigningKey

from tinytms import receipts_index as ri
from tinytms.receipt import find_machine_paths, public_key_hex, sign_receipt, write_receipt
from tinytms.web import make_app

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "receipts_index.py"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)
    monkeypatch.setattr(socket.socket, "bind", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


def _checkin_body(decision: str, reasons: list[str]) -> dict:
    return {"kind": "checkin", "utc": "2026-10-30T12:00:00.000000Z", "mode": "fixture",
            "models": {w: {"model": f"FIXTURE:{w}.FIXTURE.json", "provider": "FIXTURE"} for w in ("vision", "doc", "voice")},
            "witnesses": {w: {"status": "OK"} for w in ("vision", "doc", "voice")},
            "verdict": {"decision": decision, "reasons": reasons, "checks": {}}}


@pytest.fixture
def tree(tmp_path):
    """receipts/ with two good receipts and keys/known_signers.txt naming their key."""
    sk = SigningKey.generate()
    rdir = tmp_path / "receipts"
    write_receipt(sign_receipt(_checkin_body("ACCEPT", []), sk), rdir, "a-ACCEPT")
    write_receipt(sign_receipt(_checkin_body("HOLD", ["COUNT_MISMATCH doc=12 vision=11"]), sk), rdir, "b-HOLD")
    (rdir / "llamadas.json").write_text("[]")
    (rdir / ".gitkeep").write_text("")
    known = tmp_path / "known_signers.txt"
    known.write_text(f"# test keys\n{public_key_hex(sk)}  # test seat\n")
    return tmp_path, rdir, known, sk


def run(rdir: Path, known: Path, *args: str) -> int:
    return ri.main(["--receipts-dir", str(rdir), "--known-signers", str(known), *args])


def test_index_lists_every_receipt_verified_and_skips_the_ledger(tree):
    _, rdir, known, sk = tree
    assert run(rdir, known) == 0
    text = (rdir / "INDEX.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| [")]
    assert [r.split("]")[0][3:] for r in rows] == ["a-ACCEPT.json", "b-HOLD.json"]  # by file name
    assert "llamadas" not in text and ".gitkeep" not in text
    assert all("✓" in r for r in rows)
    assert public_key_hex(sk)[:8] in rows[0] and "test seat" in rows[0]
    assert "COUNT_MISMATCH doc=12 vision=11" in rows[1]
    assert run(rdir, known, "--check") == 0


def test_tampered_receipt_is_a_cross_and_fails_check(tree):
    _, rdir, known, _ = tree
    p = rdir / "b-HOLD.json"
    r = json.loads(p.read_text())
    r["verdict"]["decision"] = "ACCEPT"  # the judge said HOLD
    p.write_text(json.dumps(r))
    assert run(rdir, known) == 1  # written, but not clean
    row = next(line for line in (rdir / "INDEX.md").read_text().splitlines() if "b-HOLD.json" in line)
    assert "✗ bad signature" in row
    assert run(rdir, known, "--check") == 1


def test_unknown_signer_fails_check_unless_allowed_locally(tree):
    _, rdir, known, _ = tree
    stranger = SigningKey.generate()
    write_receipt(sign_receipt(_checkin_body("ACCEPT", []), stranger), rdir, "c-ACCEPT")
    run(rdir, known, "--allow-unknown-signers")
    row = next(line for line in (rdir / "INDEX.md").read_text().splitlines() if "c-ACCEPT.json" in line)
    assert "✗ unknown signer" in row and public_key_hex(stranger)[:8] in row
    assert run(rdir, known, "--check") == 1
    assert run(rdir, known, "--check", "--allow-unknown-signers") == 0


def test_check_catches_a_stale_index_and_writing_is_idempotent(tree):
    _, rdir, known, sk = tree
    assert run(rdir, known, "--check") == 1  # no INDEX.md yet
    assert run(rdir, known) == 0
    first = (rdir / "INDEX.md").read_bytes()
    assert run(rdir, known) == 0 and (rdir / "INDEX.md").read_bytes() == first
    assert run(rdir, known, "--check") == 0
    write_receipt(sign_receipt(_checkin_body("ACCEPT", []), sk), rdir, "d-ACCEPT")
    assert run(rdir, known, "--check") == 1  # a new receipt makes the index stale


def test_known_signers_file_parses_keys_and_seat_labels(tmp_path):
    k = tmp_path / "ks.txt"
    k.write_text("# header\n\n" + "ab" * 32 + "  # seat one\n" + "CD" * 32 + "\n")
    assert ri.load_known_signers(k) == {"ab" * 32: "seat one", "cd" * 32: ""}
    k.write_text("not-a-key # seat\n")
    with pytest.raises(ValueError):
        ri.load_known_signers(k)


# --- the real tree ----------------------------------------------------------------------------------


def test_repository_index_is_fresh_and_every_receipt_verifies():
    assert ri.main(["--check"]) == 0


def test_known_signers_are_the_three_seats_the_readme_names():
    known = ri.load_known_signers(REPO / "keys" / "known_signers.txt")
    assert sorted(k[:8] for k in known) == ["37f9717d", "50be0ced", "a100b16d"]
    assert all(known.values())  # every key says which seat it is


def test_local_checkin_row_shows_the_voice_exactly_as_the_receipt_says():
    text = (REPO / "receipts" / "INDEX.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if "20261028T165434299357Z-HOLD.json" in line)
    assert "voice: local-mlx · NO_MODEL" in row
    assert "NO_MODEL:voice" in row


def test_index_quotes_no_figure_without_a_receipt_and_no_machine_path():
    spec = importlib.util.spec_from_file_location("check_docs_numbers", REPO / "scripts" / "check_docs_numbers.py")
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    text = (REPO / "receipts" / "INDEX.md").read_text(encoding="utf-8")
    assert guard.problems(text, REPO) == []
    assert find_machine_paths(text) == []


def test_ci_runs_the_check_with_known_signers_only():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "python scripts/receipts_index.py --check" in ci
    assert "--allow-unknown-signers" not in ci


def test_readme_points_at_the_index():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    measured = readme.split("## What is measured", 1)[1].split("\n## ", 1)[0]
    assert "receipts/INDEX.md" in measured


def test_known_signers_is_tracked_and_private_keys_stay_ignored():
    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO).returncode == 0

    assert not ignored("keys/known_signers.txt")
    assert ignored("keys/ed25519.key") and ignored("keys/ed25519.pub")


def test_script_is_a_thin_wrapper():
    assert SCRIPT.is_file() and "tinytms.receipts_index" in SCRIPT.read_text()


# --- web --------------------------------------------------------------------------------------------


def call(app, path, host="127.0.0.1"):
    env = {"REQUEST_METHOD": "GET", "PATH_INFO": path, "wsgi.input": io.BytesIO(b""), "CONTENT_LENGTH": "0"}
    setup_testing_defaults(env)
    env["HTTP_HOST"] = host
    got = {}

    def start(status, headers):
        got["status"], got["headers"] = status, dict(headers)

    data = b"".join(app(env, start))
    return got["status"], got["headers"], data.decode()


def test_web_receipts_page_is_the_index_as_escaped_html(tree):
    tmp, rdir, known, sk = tree
    body = _checkin_body("HOLD", ["<script>alert(1)</script>"])
    write_receipt(sign_receipt(body, sk), rdir, "e-HOLD")
    app = make_app(receipts_dir=rdir, keys_dir=tmp / "keys", known_signers=known)
    status, headers, page = call(app, "/receipts/")
    assert status == "200 OK" and headers["Content-Type"].startswith("text/html")
    for name in ("a-ACCEPT.json", "b-HOLD.json", "e-HOLD.json"):
        assert f'href="/receipts/{name}"' in page
    assert "<script" not in page.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "✓" in page and "test seat" in page
    status, _, data = call(app, "/receipts/a-ACCEPT.json")  # the JSON route still answers
    assert status == "200 OK" and json.loads(data)["verdict"]["decision"] == "ACCEPT"


def test_web_receipts_page_refuses_a_foreign_host(tree):
    tmp, rdir, known, _ = tree
    app = make_app(receipts_dir=rdir, keys_dir=tmp / "keys", known_signers=known)
    status, _, _ = call(app, "/receipts/", host="evil.example")
    assert status == "403 Forbidden"
