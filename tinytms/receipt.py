"""Signed receipts: canonical JSON (sorted keys, no spaces, UTF-8) + Ed25519 (PyNaCl).

The signature covers every field except "signature" itself, including "public_key".
A receipt that carries its own key proves integrity; pass trusted_public_key to also pin origin.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

from . import __version__

RECEIPT_VERSION = "tinytms-receipt/1"
PRIVATE_KEY_FILE = "ed25519.key"
PUBLIC_KEY_FILE = "ed25519.pub"

# A receipt describes a run, never the machine it ran on: no user home, temp or volume paths.
# One definition for the finder (tests, guards) and the redactor (errors, receipts).
MACHINE_PATH_PREFIXES = ("/Users/", "/private/", "/home/", "/Volumes/", "/var/folders/", "/tmp/", "/root/")
REDACTED = "<machine-path>"
# From a prefix to the end of the line or to a closing quote/paren: paths carry spaces on macOS
# ("Mobile Documents"), so stopping at the first space would leak the rest.
_MACHINE_PATH_RE = re.compile("(?:" + "|".join(re.escape(p) for p in MACHINE_PATH_PREFIXES) + r")[^\"'\n)]*")


def redact_machine_paths(text: str) -> str:
    """Replace every machine path in free text with REDACTED; clean text comes back unchanged."""
    return _MACHINE_PATH_RE.sub(REDACTED, text)


def find_machine_paths(obj: Any) -> list[str]:
    """Every string anywhere in obj (keys included) that carries a machine path prefix."""
    found: list[str] = []
    if isinstance(obj, str):
        if any(p in obj for p in MACHINE_PATH_PREFIXES):
            found.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += find_machine_paths(k)
            found += find_machine_paths(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found += find_machine_paths(v)
    return found


def redact_body(obj: Any) -> tuple[Any, int]:
    """Copy of obj with every machine path redacted (keys included) and how many strings changed."""
    if isinstance(obj, str):
        new = redact_machine_paths(obj)
        return new, int(new != obj)
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        n = 0
        for k, v in obj.items():
            k2, nk = redact_body(k)
            v2, nv = redact_body(v)
            out[k2] = v2
            n += nk + nv
        return out, n
    if isinstance(obj, (list, tuple)):
        items = [redact_body(v) for v in obj]
        return [v for v, _ in items], sum(c for _, c in items)
    return obj, 0


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def load_or_create_key(keys_dir: str | Path = "keys") -> SigningKey:
    """Load keys/ed25519.key (hex seed) or create it with mode 0600. Never prints the key."""
    d = Path(keys_dir)
    priv = d / PRIVATE_KEY_FILE
    if priv.exists():
        return SigningKey(bytes.fromhex(priv.read_text().strip()))
    d.mkdir(parents=True, exist_ok=True)
    sk = SigningKey.generate()
    fd = os.open(priv, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(bytes(sk).hex() + "\n")
    (d / PUBLIC_KEY_FILE).write_text(public_key_hex(sk) + "\n")
    return sk


def public_key_hex(sk: SigningKey) -> str:
    return sk.verify_key.encode().hex()


def sign_receipt(body: dict[str, Any], sk: SigningKey) -> dict[str, Any]:
    """Return body + version/tinytms/public_key + signature (hex)."""
    if "signature" in body:
        raise ValueError("body already has a signature")
    # Machine paths never enter a receipt, and a path in free text (a model's words, a document's
    # remarks) must not block the receipt either: redact, count, sign. The count makes the edit visible.
    unsigned, redacted = redact_body(dict(body))
    if redacted:
        unsigned["redacted_machine_paths"] = redacted
    unsigned.setdefault("version", RECEIPT_VERSION)
    unsigned.setdefault("tinytms", __version__)
    unsigned["public_key"] = public_key_hex(sk)
    sig = sk.sign(canonical_json(unsigned)).signature
    return {**unsigned, "signature": sig.hex()}


def verify_receipt(
    receipt: dict[str, Any] | str | bytes, trusted_public_key: str | None = None
) -> bool:
    """True only if the signature is valid (and the key matches trusted_public_key if given)."""
    try:
        if isinstance(receipt, (str, bytes)):
            receipt = json.loads(receipt)
        if not isinstance(receipt, dict):
            return False
        body = {k: v for k, v in receipt.items() if k != "signature"}
        pk = body.get("public_key")
        sig = receipt.get("signature")
        if not isinstance(pk, str) or not isinstance(sig, str):
            return False
        if trusted_public_key is not None and pk != trusted_public_key:
            return False
        VerifyKey(bytes.fromhex(pk)).verify(canonical_json(body), bytes.fromhex(sig))
        return True
    except (BadSignatureError, ValueError, TypeError, UnicodeDecodeError):
        return False


def write_receipt(receipt: dict[str, Any], receipts_dir: str | Path, stem: str) -> Path:
    d = Path(receipts_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}.json"
    i = 1
    while path.exists():
        path = d / f"{stem}-{i}.json"
        i += 1
    path.write_bytes(canonical_json(receipt) + b"\n")
    return path
