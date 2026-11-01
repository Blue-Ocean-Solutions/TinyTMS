"""README "Verify a receipt", run exactly as written in a fresh clone (no keys/ed25519.pub: that key is local).

The snippet must say True for every committed receipt, signed by a seat in keys/known_signers.txt, and False for
the same body re-signed by any other key. Before 2026-11-01 it pinned keys/ed25519.pub: FileNotFoundError in a
clean clone, and False for all of them once a fixture run had created a local key.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from tinytms.receipt import canonical_json, sign_receipt

REPO = Path(__file__).resolve().parent.parent


def readme_snippet() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    section = text[text.index("Verify a receipt"):]
    (code,) = re.findall(r"```python\n(.*?)```", section, flags=re.S)[:1]
    assert "<file>" in code and "ed25519.pub" not in code
    return code


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    dst = tmp_path_factory.mktemp("clone") / "TinyTMS"
    try:
        subprocess.run(["git", "clone", "--quiet", str(REPO), str(dst)], check=True, capture_output=True, timeout=120)
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"cannot make a clone here: {e}")
    assert not (dst / "keys" / "ed25519.pub").exists()  # what a judge has: no local key
    return dst


def run_snippet(clone: Path, name: str, monkeypatch, capsys) -> str:
    monkeypatch.chdir(clone)
    exec(compile(readme_snippet().replace("<file>", Path(name).stem), "README.md", "exec"), {})
    return capsys.readouterr().out.strip()


def test_snippet_says_true_for_every_committed_receipt(clone, monkeypatch, capsys):
    names = sorted(p.name for p in (clone / "receipts").glob("*.json") if p.name != "llamadas.json")
    assert len(names) >= 17
    assert {n: run_snippet(clone, n, monkeypatch, capsys) for n in names} == {n: "True" for n in names}


def test_snippet_says_false_for_a_known_body_signed_by_another_key(clone, monkeypatch, capsys):
    name = "20261028T153041947742Z-HOLD.json"
    body = {k: v for k, v in json.loads((clone / "receipts" / name).read_bytes()).items()
            if k not in ("signature", "public_key")}
    forged = clone / "receipts" / "forged-copy.json"
    forged.write_bytes(canonical_json(sign_receipt(body, SigningKey.generate())))
    assert run_snippet(clone, forged.name, monkeypatch, capsys) == "False"
