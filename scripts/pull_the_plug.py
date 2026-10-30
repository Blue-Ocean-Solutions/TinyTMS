"""Pull the plug: the same check-in twice on the same three inputs, through the API and then without it.

    python scripts/pull_the_plug.py --load <photo> --doc <document photo> --voice <wav> [--receipts-dir receipts]

Run 1 is `tinytms.cli checkin --mode api`. Run 2 is `--mode auto` with TINYTMS_META_BASE_URL pointed at a closed
port on 127.0.0.1, inside that child process only: the API is unreachable for it, the network stack is not touched.
Each run is the real CLI in its own process (the client reads TINYTMS_META_BASE_URL once, at import) and writes its
own signed receipt; the side-by-side summary is read back from those two receipts after their signatures verify
and their signer is found in keys/known_signers.txt (--allow-unknown-signers: local use only).

With MODEL_API_KEY set, run 1 spends three Meta API calls, logged to the call ledger like every other; run 2's
attempts at the closed port are logged there too, as transport failures.

Nothing is faked. Without a Meta key both runs lose the API to NoKey; without the local worker or Glimmer (as in the
cloud) run 2's paper and photo are NO_MODEL or ERROR and the summary says so; voice is always NO_MODEL locally.
Exit 0 when both receipts are written and verify with a known signer, 1 when one does not, 3 when a run is an
operational error.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from receipt_show import KNOWN_SIGNERS, WITNESSES, check, fields, seconds  # noqa: E402

sys.path.insert(0, str(REPO))
from tinytms.receipts_index import load_known_signers  # noqa: E402
from tinytms.verify import DEFAULT_MAX_WEIGHT_KG  # noqa: E402


def closed_local_url() -> str:
    """http://127.0.0.1:<port>/v1 for a port that was free a moment ago and has nothing listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def run_cli(mode: str, args: argparse.Namespace, extra_env: dict[str, str]) -> tuple[int, Path | None, str]:
    cmd = [sys.executable, "-m", "tinytms.cli", "checkin", "--mode", mode, "--load", args.load, "--doc", args.doc,
           "--voice", args.voice, "--concept", args.concept, "--max-weight-kg", str(args.max_weight_kg),
           "--receipts-dir", args.receipts_dir, "--keys-dir", args.keys_dir]
    if args.calls_log:
        cmd += ["--calls-log", args.calls_log]
    env = {**os.environ, **extra_env}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    path = next((Path(line.split("receipt: ", 1)[1].strip()) for line in p.stdout.splitlines()
                 if line.startswith("  receipt: ")), None)
    return p.returncode, path, p.stderr.strip()


def cell(receipt: dict[str, Any], name: str) -> str:
    w = receipt["witnesses"][name]
    out = f"{w['status']} {w['provider']} {w.get('model') or '-'} {seconds(receipt, name)} s"
    f = fields(name, w)
    out += f" {f}" if f else ""
    if name in receipt.get("fallback", {}):
        out += f" (fallback: {receipt['fallback'][name]['why'][:60]})"
    return out


def side_by_side(r1: dict[str, Any], p1: Path, r2: dict[str, Any], p2: Path, url: str) -> list[str]:
    lines = [f"run 1 · --mode api  |  run 2 · --mode auto, API unreachable ({url})"]
    for k, v in r1.get("inputs", {}).items():
        lines.append(f"input {k:<5} {v['name']} sha256 {v['sha256'][:12]}")
    for name in WITNESSES:
        lines.append(f"{name:<7} {cell(r1, name)} | {cell(r2, name)}")
    v1, v2 = r1["verdict"], r2["verdict"]
    lines.append(f"verdict {v1['decision']} {'; '.join(v1['reasons'])} | {v2['decision']} {'; '.join(v2['reasons'])}")
    lines.append(f"receipt {p1} | {p2}")
    local = [n for n in ("vision", "doc") if r2["witnesses"][n]["provider"] == "local-mlx"]
    if any(r2["witnesses"][n]["status"] != "OK" for n in local):
        lines.append("note: the Mac did not answer here (no local worker or no Glimmer on this machine); those "
                     "witnesses are NO_MODEL/ERROR and nothing was guessed.")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--load", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--concept", default="pallet")
    ap.add_argument("--max-weight-kg", type=float, default=DEFAULT_MAX_WEIGHT_KG)
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--keys-dir", default=os.environ.get("TINYTMS_KEYS_DIR", "keys"))
    ap.add_argument("--calls-log")
    ap.add_argument("--unreachable-url", help="base URL for run 2 (default: a closed port on 127.0.0.1)")
    ap.add_argument("--known-signers", default=str(KNOWN_SIGNERS),
                    help="allowlist of signer keys (default: %(default)s)")
    ap.add_argument("--allow-unknown-signers", action="store_true", help="local use only: accept receipts signed by a "
                    "key that is not in the list")
    args = ap.parse_args(argv)
    url = args.unreachable_url or closed_local_url()
    try:
        known = load_known_signers(args.known_signers)
    except (OSError, ValueError) as e:
        print(f"ERROR: known signers: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    receipts: list[tuple[dict[str, Any], Path]] = []
    for label, mode, env in (("run 1", "api", {}), ("run 2", "auto", {"TINYTMS_META_BASE_URL": url})):
        code, path, err = run_cli(mode, args, env)
        if code not in (0, 2) or path is None:
            print(f"{label} (--mode {mode}) failed, exit {code}: {err}", file=sys.stderr)
            return 3
        raw = path.read_bytes()
        ok, line = check(raw, known, allow_unknown=args.allow_unknown_signers)
        if not ok:
            print(f"{label}: receipt {path}: {line}", file=sys.stderr)
            return 1
        receipts.append((json.loads(raw), path))
    (r1, p1), (r2, p2) = receipts
    print("\n".join(side_by_side(r1, p1, r2, p2, url)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
