"""receipts/INDEX.md: every signed receipt in one table, its signature verified and its signer named.

    python scripts/receipts_index.py            # (re)write receipts/INDEX.md; exit 1 if any receipt is not clean
    python scripts/receipts_index.py --check    # exit 1 if INDEX.md is stale or any receipt is not clean (CI)

A receipt is clean when its Ed25519 signature verifies AND its public key is listed in keys/known_signers.txt
(one hex key per line, a comment naming the seat). A valid signature by an unlisted key proves integrity, not
origin: its row says "✗ unknown signer". --allow-unknown-signers lets such rows pass, for local use only, never CI.
The same rows back the web page GET /receipts/. Receipts are only read, never touched.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .receipt import verify_receipt

REPO = Path(__file__).resolve().parent.parent
INDEX_NAME = "INDEX.md"
SKIP = {"llamadas.json"}  # the API call ledger is a log, not a signed receipt
HEX_KEY = re.compile(r"[0-9a-fA-F]{64}")
COLUMNS = ("file", "kind", "witness / mode", "model · provider", "utc", "result", "verify", "signer")


@dataclass(frozen=True)
class Row:
    file: str
    kind: str
    what: str
    model: str
    utc: str
    result: str
    verify: str  # "✓", "✗ bad signature", "✗ unknown signer", "✗ unreadable"
    signer: str
    signature_ok: bool
    known_signer: bool

    def cells(self) -> tuple[str, ...]:
        return (self.file, self.kind, self.what, self.model, self.utc, self.result, self.verify, self.signer)


def load_known_signers(path: str | Path) -> dict[str, str]:
    """hex public key (lower case) -> seat label (the comment on its line). Malformed lines are an error."""
    known: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        key, _, label = line.partition("#")
        key = key.strip()
        if not key:
            continue
        if not HEX_KEY.fullmatch(key):
            raise ValueError(f"{Path(path).name}:{n}: not a 64-hex Ed25519 public key")
        known[key.lower()] = label.strip()
    return known


def _models(r: dict[str, Any]) -> str:
    """Per witness: model · provider, plus the witness status when it is not OK (exactly as the receipt says)."""
    out = []
    for w, m in r.get("models", {}).items():
        parts = [p for p in (m.get("model"), m.get("provider")) if p]
        status = r.get("witnesses", {}).get(w, {}).get("status")
        if status and status != "OK":
            parts.append(status)
        out.append(f"{w}: " + " · ".join(parts))
    return "; ".join(out)


def _fields(d: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(f"{k}={d[k]}" for k in keys if k in d)


def _summary(r: dict[str, Any]) -> tuple[str, str, str]:
    """(witness / mode, model · provider, result) for one receipt, by kind."""
    kind = r.get("kind")
    if kind == "checkin":
        v = r.get("verdict", {})
        result = " · ".join([v.get("decision", "?"), ", ".join(v.get("reasons", [])) or "no reasons"])
        if "explanation" in r:
            e = r["explanation"]
            result += f" · explanation {e.get('status')} ({e.get('language', '?')})"
        return f"check-in · {r.get('mode', '?')}", _models(r), result
    if kind == "explain":
        e, v, src = r.get("explanation", {}), r.get("verdict", {}), r.get("source_receipt", {})
        model = " · ".join(p for p in (e.get("model"), e.get("provider")) if p)
        decision = v.get("decision", "?") if isinstance(v, dict) else str(v)
        return (f"explain · {e.get('language', '?')}", model,
                f"{e.get('status', '?')} · explains {src.get('name', '?')} ({decision})")
    if kind == "live-witness":
        res = r.get("result", {})
        status = res.get("status", "?")
        seen = _fields(res, ("concept", "count", "pieces", "weight_kg", "seal"))
        detail = res.get("detail") or ""
        result = " · ".join(p for p in (f"HTTP {r.get('http')}", status, seen, detail) if p)
        return f"live · {r.get('witness', '?')}", " · ".join(p for p in (r.get("model"), r.get("provider")) if p), result
    if kind == "local-vlm-measure":
        m = r.get("model", {})
        model = " · ".join(p for p in (m.get("id"), m.get("provider")) if p) if isinstance(m, dict) else str(m)
        n = len(r.get("runs", []))
        return "local VLM measure", model, f"{r.get('status', '?')} · {n} run{'s' if n != 1 else ''}; figures in the receipt"
    return str(kind or "?"), "", ""


def build_rows(receipts_dir: str | Path, known: dict[str, str]) -> list[Row]:
    rows: list[Row] = []
    for p in sorted(Path(receipts_dir).glob("*.json"), key=lambda q: q.name):
        if p.name in SKIP:
            continue
        try:
            r = json.loads(p.read_bytes())
            if not isinstance(r, dict):
                raise ValueError("not a JSON object")
        except (ValueError, UnicodeDecodeError):
            rows.append(Row(p.name, "?", "", "", "", "", "✗ unreadable", "", False, False))
            continue
        pk = r.get("public_key") if isinstance(r.get("public_key"), str) else ""
        ok = verify_receipt(r)
        is_known = pk.lower() in known
        verify = "✓" if ok and is_known else ("✗ bad signature" if not ok else "✗ unknown signer")
        label = known.get(pk.lower(), "")
        signer = f"{pk[:8]} {label}".strip() if pk else ""
        what, model, result = _summary(r)
        rows.append(Row(p.name, str(r.get("kind", "?")), what, model, str(r.get("utc", "")), result, verify, signer,
                        ok, is_known))
    return rows


def clean(row: Row, allow_unknown_signers: bool = False) -> bool:
    return row.signature_ok and (row.known_signer or allow_unknown_signers)


def _md_cell(s: str) -> str:
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown(rows: list[Row]) -> str:
    """Deterministic: same receipts and signers -> same bytes (no generation time)."""
    lines = [
        "# Receipts index",
        "",
        "Generated by `python scripts/receipts_index.py`; CI runs it with `--check`, so this file cannot go stale.",
        "Do not edit by hand. Each row is one signed receipt in this folder, in file-name order. `verify` is ✓ when",
        "the Ed25519 signature checks and the signer is listed in `keys/known_signers.txt`. Figures (times, token",
        "rates) stay inside the receipts; this table only says what each one is.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for row in rows:
        file, *rest = row.cells()
        lines.append("| " + " | ".join([f"[{_md_cell(file)}]({file})"] + [_md_cell(c) for c in rest]) + " |")
    lines += ["", f"{sum(clean(r) for r in rows)} of {len(rows)} receipts verify with a known signer.", ""]
    return "\n".join(lines)


def render_html_table(rows: list[Row]) -> str:
    """The same table as HTML (text escaped, no script); each file links to GET /receipts/<name>."""
    e = html.escape
    head = "".join(f"<th>{e(c)}</th>" for c in COLUMNS)
    body = []
    for row in rows:
        file, *rest = row.cells()
        css = "OK" if clean(row) else "ERROR"
        cells = [f'<td><a href="/receipts/{e(file)}">{e(file)}</a></td>']
        cells += [f'<td class="{css}">{e(c)}</td>' if c is row.verify else f"<td>{e(c)}</td>" for c in rest]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="receipts_index", description=__doc__.splitlines()[0])
    ap.add_argument("--receipts-dir", default=str(REPO / "receipts"))
    ap.add_argument("--known-signers", default=str(REPO / "keys" / "known_signers.txt"))
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if INDEX.md is stale or a row fails")
    ap.add_argument("--allow-unknown-signers", action="store_true",
                    help="local use only, never in CI: a valid signature by an unlisted key does not fail")
    args = ap.parse_args(argv)
    rdir = Path(args.receipts_dir)
    try:
        known = load_known_signers(args.known_signers)
    except (OSError, ValueError) as e:
        print(f"ERROR: known signers: {e}", file=sys.stderr)
        return 1
    rows = build_rows(rdir, known)
    text = render_markdown(rows)
    index = rdir / INDEX_NAME
    bad = [r for r in rows if not clean(r, args.allow_unknown_signers)]
    for r in bad:
        print(f"{r.verify}: {r.file} ({r.signer or 'no public key'})", file=sys.stderr)
    if args.check:
        current = index.read_text(encoding="utf-8") if index.is_file() else None
        if current != text:
            print(f"{INDEX_NAME} is stale: run python scripts/receipts_index.py", file=sys.stderr)
            return 1
    else:
        index.write_text(text, encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
