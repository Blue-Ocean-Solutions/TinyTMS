"""Measure Muse Glimmer 30B 4-bit on this Mac: load time, generation tokens/s, raw output on one image.

NOT run by the build lane. For the cockpit / L1, on a session with Metal access:

    TINYTMS_LOCAL_PYTHON=/path/to/venv-with-glimmer-mlx_vlm/bin/python \
      python scripts/measure_local_vlm.py --image fixtures/accept/doc.FIXTURE.png

Writes receipts/local-vlm-<utc>.json (signed) with the exact command, date, interpreter, mlx_vlm
version and the library-reported stats. On failure the receipt records the literal error and the
exit code is 3; nothing is estimated or filled in.
"""

from __future__ import annotations

import argparse
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytms._local import LocalModelError, local_python  # noqa: E402
from tinytms.local_vlm import GLIMMER_MODEL_ID, GLIMMER_REPO, LocalVLM, mlx_vlm_version  # noqa: E402
from tinytms.receipt import load_or_create_key, sign_receipt, write_receipt  # noqa: E402
from tinytms.witness import sha256_file  # noqa: E402

DEFAULT_PROMPT = "Describe this image in one sentence."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=1, help="generations after one load")
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--keys-dir", default="keys")
    a = ap.parse_args()

    now = datetime.now(timezone.utc)
    py = local_python()
    body: dict = {
        "kind": "local-vlm-measure",
        "utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "command": " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
        "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0], "node": platform.node()},
        "model": {"repo": GLIMMER_REPO, "id": GLIMMER_MODEL_ID, "provider": "local-mlx"},
        "interpreter": py,
        "mlx_vlm_version": mlx_vlm_version(py),
        "image": {"name": Path(a.image).name, "sha256": sha256_file(a.image)},
        "prompt": a.prompt,
        "max_tokens": a.max_tokens,
    }
    rc = 0
    vlm = LocalVLM(python=py)
    try:
        t0 = time.perf_counter()
        info = vlm.load()
        body["load"] = {"load_s_in_worker": info.load_s,
                        "load_s_incl_process_start": round(time.perf_counter() - t0, 3),
                        "model_path": info.model_path, "worker_python": info.python,
                        "package_version": info.package_version}
        runs = []
        for _ in range(a.runs):
            g = vlm.describe(a.image, a.prompt, a.max_tokens)
            runs.append({"text": g.text, "prompt_tokens": g.prompt_tokens,
                         "generation_tokens": g.generation_tokens, "generation_tps": g.generation_tps,
                         "peak_memory_gb": g.peak_memory_gb, "wall_s": g.wall_s})
        body["runs"] = runs
        body["status"] = "OK"
    except LocalModelError as e:
        body["status"] = "ERROR"
        body["error"] = f"{type(e).__name__}: {e}"
        rc = 3
    finally:
        vlm.close()

    receipt = sign_receipt(body, load_or_create_key(a.keys_dir))
    stem = "local-vlm-" + now.strftime("%Y%m%dT%H%M%SZ") + ("" if rc == 0 else "-ERROR")
    path = write_receipt(receipt, a.receipts_dir, stem)
    print(f"{body['status']}  mlx_vlm={body['mlx_vlm_version']}  receipt: {path}")
    if rc:
        print(f"  {body['error']}", file=sys.stderr)
    else:
        for r in body["runs"]:
            print(f"  load_s={body['load']['load_s_in_worker']} gen_tps={r['generation_tps']} "
                  f"tokens={r['generation_tokens']} wall_s={r['wall_s']}")
            print(f"  text: {r['text'][:300]!r}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
