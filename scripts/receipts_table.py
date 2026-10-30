"""Build docs/MEASURED.md from receipts/: one row per receipt, its signature checked with the key it carries.

    python scripts/receipts_table.py            # write docs/MEASURED.md
    python scripts/receipts_table.py --check    # exit 1 if docs/MEASURED.md differs from what receipts/ says
    python scripts/receipts_table.py --out FILE # write somewhere else (CI writes a temp file and diffs it)

No figure in the output is typed by hand: each one comes from a receipt and sits on the same line as its link, so
scripts/check_docs_numbers.py accepts it. A receipt is verified when its Ed25519 signature checks with the key it
carries AND that key is in keys/known_signers.txt (the allowlist receipts/INDEX.md uses too). Anything else (bad
JSON, no signature, a changed byte, a valid signature by an unknown key) is listed under "Not verified" with the
reason, never dropped. Output is deterministic: sorted by utc, then file name.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinytms.receipt import verify_receipt  # noqa: E402
from tinytms.receipts_index import load_known_signers  # noqa: E402

KNOWN_SIGNERS = ROOT / "keys" / "known_signers.txt"

WITNESS_ORDER = ("vision", "doc", "voice")


@dataclass(frozen=True)
class Row:
    name: str
    verified: bool
    why: str = ""  # why not verified
    utc: str = ""
    kind: str = ""
    model: str = ""
    http: str = ""
    seconds: str = ""
    result: str = ""
    key: str = ""


def _num(x: Any) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _secs(x: Any) -> str:
    return f"{_num(x)} s" if isinstance(x, (int, float)) and not isinstance(x, bool) else "—"


def _cell(s: str) -> str:
    return " ".join(str(s).split()).replace("|", "\\|") or "—"


def _model(model: Any, provider: Any) -> str:
    return f"{model or 'none'} ({provider})" if provider else str(model or "none")


def _fields(r: dict[str, Any], keys: tuple[str, ...]) -> str:
    def show(k: str, v: Any) -> str:
        if k == "weight_kg" and isinstance(v, (int, float)):
            return f"{_num(v)} kg"
        return f"{k} {'—' if v is None else json.dumps(v, ensure_ascii=False).strip(chr(34))}"

    return ", ".join(show(k, r.get(k)) for k in keys if k in r)


def _witness_result(r: Any) -> str:
    if not isinstance(r, dict):
        return "—"
    if r.get("status") != "OK":
        return f"{r.get('status')}: {r.get('detail', '')}"
    if "concept" in r:
        return f"count {_num(r.get('count'))} ({r.get('concept')})"
    if "pieces" in r:
        return _fields(r, ("pieces", "weight_kg", "seal", "hazmat"))
    out = _fields(r, ("count", "seal"))
    return out + (f" ({r['detail']})" if r.get("detail") else "")


def _per_witness(d: dict[str, Any], key: str, fmt) -> str:
    parts = [f"{w} {fmt(d[w][key])}" for w in (*WITNESS_ORDER, "explain") if isinstance(d.get(w), dict)
             and key in d[w]]
    return " · ".join(parts) or "—"


def _summarise(d: dict[str, Any]) -> dict[str, str]:
    kind = d.get("kind")
    if kind == "live-witness":
        return {"kind": f"live · {d.get('witness')}", "model": _model(d.get("model"), d.get("provider")),
                "http": _num(d.get("http", "—")), "seconds": _secs(d.get("s")),
                "result": _witness_result(d.get("result"))}
    if kind == "checkin":
        models = d.get("models") or {}
        provs = {m.get("provider") for m in models.values() if isinstance(m, dict)}
        if len(provs) == 1:
            names = ", ".join(f"{w} {models[w].get('model') or 'none'}" for w in WITNESS_ORDER if w in models)
            model = f"{names} ({provs.pop()})"
        else:
            model = ", ".join(f"{w} {_model(models[w].get('model'), models[w].get('provider'))}"
                              for w in WITNESS_ORDER if w in models)
        calls = dict(d.get("calls") or {})
        ex = d.get("explanation")
        if isinstance(ex, dict):
            calls["explain"] = ex
        v = d.get("verdict") or {}
        result = f"{v.get('decision')}: {', '.join(v.get('reasons') or []) or 'no reasons'}"
        if isinstance(ex, dict):
            result += f"; explanation {ex.get('status')} ({ex.get('language')})"
        return {"kind": f"check-in · {d.get('mode')}", "model": model,
                "http": _per_witness(calls, "http", _num), "seconds": _per_witness(calls, "s", _secs),
                "result": result}
    if kind == "explain":
        ex = d.get("explanation") or {}
        src = (d.get("source_receipt") or {}).get("name", "?")
        result = f"explanation {ex.get('status')} ({ex.get('language')}) for {src}"
        if ex.get("detail"):
            result += f": {ex['detail']}"
        usage = ex.get("usage") or {}
        if "output_tokens" in usage:
            reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            result += f"; output tokens {usage['output_tokens']}" + (f" (reasoning {reasoning})" if reasoning else "")
        return {"kind": "explain", "model": _model(ex.get("model"), ex.get("provider")),
                "http": _num(ex.get("http", "—")), "seconds": _secs(ex.get("s")), "result": result}
    if kind == "local-vlm-measure":
        m = d.get("model") or {}
        load = d.get("load") or {}
        runs = [r for r in d.get("runs") or [] if isinstance(r, dict)]
        secs = [f"load {_secs(load.get('load_s_in_worker'))}"] + [f"run {_secs(r.get('wall_s'))}" for r in runs]
        result = "; ".join(
            f"{r.get('generation_tps', 0):.2f} tok/s, {r.get('prompt_tokens')} prompt + {r.get('generation_tokens')}"
            f" generated tokens, peak {r.get('peak_memory_gb', 0):.2f} GB" for r in runs)
        return {"kind": "local VLM measure", "model": f"{_model(m.get('id'), m.get('provider'))}, mlx_vlm "
                f"{d.get('mlx_vlm_version')}", "http": "—", "seconds": " · ".join(secs),
                "result": f"{d.get('status')}: {result}" if result else str(d.get("status"))}
    return {"kind": str(kind or "?"), "model": _model(d.get("model"), d.get("provider")),
            "http": _num(d.get("http", "—")), "seconds": _secs(d.get("s")), "result": "—"}


def _why_not(raw: bytes, d: Any) -> str:
    if not isinstance(d, dict):
        return "not a JSON object"
    if not isinstance(d.get("signature"), str) or not isinstance(d.get("public_key"), str):
        if isinstance(d.get("llamadas"), list):
            per = {}
            for c in d["llamadas"]:
                if isinstance(c, dict):
                    per[str(c.get("modelo"))] = per.get(str(c.get("modelo")), 0) + 1
            calls = ", ".join(f"{m} {n}" for m, n in sorted(per.items()))
            return f"no signature: the API call ledger ({len(d['llamadas'])} calls: {calls})"
        return "no signature"
    return "signature does not verify with the public key it carries"


def collect(receipts_dir: str | Path, known: dict[str, str] | None = None) -> list[Row]:
    """known: hex public key -> seat (default: keys/known_signers.txt)."""
    known = load_known_signers(KNOWN_SIGNERS) if known is None else known
    rows = []
    for p in sorted(Path(receipts_dir).glob("*.json")):
        raw = p.read_bytes()
        try:
            d = json.loads(raw)
        except ValueError:
            rows.append(Row(p.name, False, "not JSON"))
            continue
        if isinstance(d, dict) and verify_receipt(raw):
            key = str(d["public_key"]).lower()
            if key in known:
                rows.append(Row(p.name, True, utc=str(d.get("utc", "")), key=key[:16], **_summarise(d)))
            else:
                rows.append(Row(p.name, False, f"unknown signer: the signature checks with key {key[:16]}…, which is "
                                f"not in keys/known_signers.txt", utc=str(d.get("utc", ""))))
        else:
            utc = str(d.get("utc", "")) if isinstance(d, dict) else ""
            rows.append(Row(p.name, False, _why_not(raw, d), utc=utc))
    return sorted(rows, key=lambda r: (not r.verified, r.utc if r.verified else "", r.name))


def render(receipts_dir: str | Path, known: dict[str, str] | None = None) -> str:
    known = load_known_signers(KNOWN_SIGNERS) if known is None else known
    rows = collect(receipts_dir, known)
    ok = [r for r in rows if r.verified]
    bad = [r for r in rows if not r.verified]
    out = [
        "# What is measured",
        "",
        "<!-- Generated by `python scripts/receipts_table.py` from the receipts folder. Do not edit by hand:",
        "     CI regenerates this file and fails if it differs. -->",
        "",
        "One row per verified receipt, oldest first. *Verified* means the Ed25519 signature checks with the public key",
        "the receipt carries (integrity) and that key is listed in keys/known_signers.txt (origin). *key* is the first",
        "16 hex digits of it.",
        "Every figure on a row comes from the receipt linked at the end of that row. A result is what the receipt",
        "recorded when it was signed; parsers have changed since (tests/test_real_data.py re-reads the same answers).",
        "",
        "| UTC | kind | model (provider) | HTTP | seconds | result | key | receipt |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        cells = (r.utc, r.kind, r.model, r.http, r.seconds, r.result, f"`{r.key}`")
        out.append("| " + " | ".join(_cell(c) for c in cells) + f" | [{r.name}](../receipts/{r.name}) |")
    keys: dict[str, int] = {}
    for r in ok:
        keys[r.key] = keys.get(r.key, 0) + 1
    out += ["", "## Signing keys", ""]
    seats = {k[:16]: label for k, label in known.items()}
    out += [f"- `{k}` ({seats.get(k, '?')}): {n} receipt{'s' if n != 1 else ''}" for k, n in sorted(keys.items())]
    out += ["", "## Not verified", ""]
    if bad:
        out += ["Listed so nothing in the receipts folder goes unseen. These are not evidence of a measurement.", "",
                "| file | why |", "|---|---|"]
        out += [f"| [{r.name}](../receipts/{r.name}) | {_cell(r.why)} |" for r in bad]
    else:
        out.append("None.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--receipts", default=str(ROOT / "receipts"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "MEASURED.md"))
    ap.add_argument("--known-signers", default=str(KNOWN_SIGNERS), help="allowlist of signer public keys")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if --out differs from a fresh render")
    args = ap.parse_args(argv)
    text = render(args.receipts, load_known_signers(args.known_signers))
    out = Path(args.out)
    if args.check:
        current = out.read_text(encoding="utf-8") if out.is_file() else ""
        if current != text:
            print(f"{out.name} is stale: run python scripts/receipts_table.py", file=sys.stderr)
            return 1
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
