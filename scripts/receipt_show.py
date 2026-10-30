"""One screen per signed receipt: the signature first, then what answered each witness and what the judge said.

    python scripts/receipt_show.py receipts/<file>.json [--trusted-key <hex>]

Exit 0 when the signature verifies (and, with --trusted-key, was made by that key); 1 when it does not, and the
body is then not shown as a receipt; 3 when the file cannot be read. Seconds come only from the receipt: an API
witness's call record (`calls.<witness>.s`) or a local answer's `wall_s=` in its detail; anything else prints `-`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytms.receipt import verify_receipt  # noqa: E402

WITNESSES = ("vision", "doc", "voice")
WALL_S = re.compile(r"\bwall_s=([0-9]+(?:\.[0-9]+)?)\b")


def seconds(receipt: dict[str, Any], name: str) -> str:
    """The seconds the receipt itself records for this witness, or '-'."""
    w = receipt.get("witnesses", {}).get(name, {})
    call = receipt.get("calls", {}).get(name)
    if w.get("provider") == "meta-api" and isinstance(call, dict) and call.get("s") is not None:
        return str(call["s"])
    m = WALL_S.search(w.get("detail") or "")
    return m.group(1) if w.get("provider") == "local-mlx" and m else "-"


def fields(name: str, w: dict[str, Any]) -> str:
    """What the witness reported, only when it answered (status OK)."""
    if w.get("status") != "OK":
        return ""
    if name == "doc":
        return (f"pieces={w.get('pieces')} weight_kg={w.get('weight_kg')} seal={w.get('seal')} "
                f"hazmat={w.get('hazmat')}")
    out = f"count={w.get('count')}"
    return out + (f" seal={w.get('seal')}" if name == "voice" else "")


def witness_line(receipt: dict[str, Any], name: str) -> str:
    w = receipt.get("witnesses", {}).get(name, {})
    parts = [f"{name:<6}", str(w.get("status", "-")), str(w.get("provider", "-")), w.get("model") or "-",
             f"{seconds(receipt, name)} s"]
    f = fields(name, w)
    return "  ".join(parts + ([f] if f else []))


def render(receipt: dict[str, Any]) -> list[str]:
    lines = [f"kind: {receipt.get('kind', '-')}  mode: {receipt.get('mode', '-')}  utc: {receipt.get('utc', '-')}"]
    for k, v in receipt.get("inputs", {}).items():
        lines.append(f"input {k:<5}  {v.get('name')}  sha256 {str(v.get('sha256', ''))[:12]}")
    if "input" in receipt:
        lines.append(f"input  {receipt['input'].get('path')}  sha256 {str(receipt['input'].get('sha256', ''))[:12]}")
    for name in WITNESSES:
        if name in receipt.get("witnesses", {}):
            lines.append(witness_line(receipt, name))
            detail = receipt["witnesses"][name].get("detail")
            if detail:
                lines.append(f"        detail: {detail[:160]}")
    calls = receipt.get("calls", {})
    for name, c in sorted(calls.items(), key=lambda kv: WITNESSES.index(kv[0]) if kv[0] in WITNESSES else 9):
        lines.append(f"call {name:<6}  {c.get('model')}  http {c.get('http')}  {c.get('status')}  {c.get('s')} s")
    for name, f in receipt.get("fallback", {}).items():
        lines.append(f"fallback {name}: -> {f.get('to')} ({str(f.get('why', ''))[:120]})")
    if receipt.get("kind") not in (None, "checkin"):
        for k in ("witness", "model", "provider", "s", "status", "command"):
            if k in receipt:
                lines.append(f"{k}: {receipt[k]}")
        if isinstance(receipt.get("result"), dict):
            r = receipt["result"]
            lines.append("result: " + "  ".join(f"{k}={r[k]}" for k in ("status", "count", "pieces", "seal",
                                                                          "transcript") if k in r))
    v = receipt.get("verdict")
    if isinstance(v, dict):
        lines.append(f"verdict: {v.get('decision')}")
        lines += [f"  reason: {r}" for r in v.get("reasons", [])]
    e = receipt.get("explanation")
    if isinstance(e, dict):
        lines.append(f"explanation [{e.get('status')}] {e.get('model')} ({e.get('language')}, {e.get('s')} s): "
                     f"{e.get('text') or e.get('detail') or ''}")
    if "redacted_machine_paths" in receipt:
        lines.append(f"redacted_machine_paths: {receipt['redacted_machine_paths']}")
    lines.append(f"signer: {str(receipt.get('public_key', ''))[:16]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("receipt")
    ap.add_argument("--trusted-key", help="hex Ed25519 public key the receipt must be signed by")
    args = ap.parse_args(argv)
    path = Path(args.receipt)
    try:
        raw = path.read_bytes()
    except OSError as e:
        print(f"ERROR: cannot read {path.name}: {type(e).__name__}", file=sys.stderr)
        return 3
    if not verify_receipt(raw, trusted_public_key=args.trusted_key):
        print(f"{path.name}: signature FAILED" + (" for the trusted key" if args.trusted_key else ""))
        return 1
    print(f"{path.name}: signature OK" + (" (trusted key)" if args.trusted_key else ""))
    print("\n".join(render(json.loads(raw))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
