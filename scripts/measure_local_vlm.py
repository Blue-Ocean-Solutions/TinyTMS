"""Measure Muse Glimmer 30B 4-bit on this Mac: load time, generation tokens/s, raw output on one image.

NOT run by the build lane. For the cockpit / L1, on a session with Metal access:

    TINYTMS_LOCAL_PYTHON=/path/to/venv-with-glimmer-mlx_vlm/bin/python \
      python scripts/measure_local_vlm.py --image evals/public/boxes-pallet.jpg

Writes receipts/local-vlm-<utc>.json (signed) with the command relative to the repo, the date, the
interpreter by variable name and version, the mlx_vlm version and the library-reported stats. The
receipt describes the run, never the machine: no user, temp or volume paths (sign_receipt refuses them).
On failure the receipt records the literal error and the exit code is 3; nothing is estimated.
"""

from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinytms._local import LocalModelError, local_python  # noqa: E402
from tinytms.local_vlm import GLIMMER_MODEL_ID, GLIMMER_REPO, LocalVLM, mlx_vlm_version  # noqa: E402
from tinytms.receipt import load_or_create_key, sign_receipt, write_receipt  # noqa: E402
from tinytms.witness import sha256_file  # noqa: E402

DEFAULT_PROMPT = "Describe this image in one sentence."
DEFAULT_MAX_TOKENS = 128
DEFAULT_RUNS = 1
INTERPRETER_ENV = "TINYTMS_LOCAL_PYTHON"


def python_version(py: str) -> str:
    """Version of the worker interpreter, asked of the interpreter itself; never its path."""
    try:
        out = subprocess.run(
            [py, "-c", "import platform; print(platform.python_version())"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _repo_relative(path: str | Path, repo_root: str | Path) -> str:
    """Path relative to the repo; a file outside the repo is named only by its basename."""
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def receipt_body(
    *,
    repo_root: str | Path,
    image: str | Path,
    prompt: str,
    max_tokens: int,
    runs_requested: int,
    interpreter_env: str,
    interpreter_version: str,
    mlx_vlm_version: str,
    load: dict[str, Any] | None,
    runs: list[dict[str, Any]],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    """The receipt body: what ran, on what input, with what result. No machine paths anywhere."""
    rel = _repo_relative(image, repo_root)
    command = f"python scripts/measure_local_vlm.py --image {shlex.quote(rel)}"
    if prompt != DEFAULT_PROMPT:
        command += f" --prompt {shlex.quote(prompt)}"
    if max_tokens != DEFAULT_MAX_TOKENS:
        command += f" --max-tokens {max_tokens}"
    if runs_requested != DEFAULT_RUNS:
        command += f" --runs {runs_requested}"

    body: dict[str, Any] = {
        "kind": "local-vlm-measure",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "command": command,
        "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0]},
        "model": {"repo": GLIMMER_REPO, "id": GLIMMER_MODEL_ID, "provider": "local-mlx"},
        "interpreter": {"env": interpreter_env, "python": interpreter_version},
        "mlx_vlm_version": mlx_vlm_version,
        "image": {"path": rel, "sha256": sha256_file(image)},
        "prompt": prompt,
        "max_tokens": max_tokens,
        "status": status,
    }
    if load is not None:
        body["model"]["snapshot"] = Path(str(load.get("model_path", ""))).name or "unknown"
        body["load"] = {
            "load_s_in_worker": load.get("load_s_in_worker"),
            "load_s_incl_process_start": load.get("load_s_incl_process_start"),
            "package_version": load.get("package_version"),
        }
    if runs:
        body["runs"] = runs
    if error:
        body["error"] = error
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="generations after one load")
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--keys-dir", default="keys")
    a = ap.parse_args()

    now = datetime.now(timezone.utc)
    py = local_python()
    load: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []
    status, error, rc = "OK", None, 0
    vlm = LocalVLM(python=py)
    try:
        t0 = time.perf_counter()
        info = vlm.load()
        load = {"load_s_in_worker": info.load_s,
                "load_s_incl_process_start": round(time.perf_counter() - t0, 3),
                "model_path": info.model_path, "package_version": info.package_version}
        for _ in range(a.runs):
            g = vlm.describe(a.image, a.prompt, a.max_tokens)
            runs.append({"text": g.text, "prompt_tokens": g.prompt_tokens,
                         "generation_tokens": g.generation_tokens, "generation_tps": g.generation_tps,
                         "peak_memory_gb": g.peak_memory_gb, "wall_s": g.wall_s})
    except LocalModelError as e:
        status, error, rc = "ERROR", f"{type(e).__name__}: {e}", 3
    finally:
        vlm.close()

    body = receipt_body(
        repo_root=ROOT, image=a.image, prompt=a.prompt, max_tokens=a.max_tokens, runs_requested=a.runs,
        interpreter_env=INTERPRETER_ENV, interpreter_version=python_version(py),
        mlx_vlm_version=mlx_vlm_version(py), load=load, runs=runs, status=status, error=error,
    )
    receipt = sign_receipt(body, load_or_create_key(a.keys_dir))
    stem = "local-vlm-" + now.strftime("%Y%m%dT%H%M%SZ") + ("" if rc == 0 else "-ERROR")
    path = write_receipt(receipt, a.receipts_dir, stem)
    print(f"{status}  mlx_vlm={body['mlx_vlm_version']}  receipt: {path}")
    if rc:
        print(f"  {error}", file=sys.stderr)
    else:
        for r in runs:
            print(f"  load_s={body['load']['load_s_in_worker']} gen_tps={r['generation_tps']} "
                  f"tokens={r['generation_tokens']} wall_s={r['wall_s']}")
            print(f"  text: {r['text'][:300]!r}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
