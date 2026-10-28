"""Standalone MLX worker (stdlib + mlx_vlm / mlx_lm only; does NOT import tinytms).

Run as a separate process so a failing import (e.g. "No Metal device available" aborts the
interpreter, or "model type not supported") cannot take the caller down, and so a different
interpreter can be used via TINYTMS_LOCAL_PYTHON.

    python _mlx_worker.py --kind vlm|lm --model /local/snapshot/dir

Protocol: one JSON object per line on the ORIGINAL stdout; library chatter goes to stderr.
  -> {"event":"hello", python, package, package_version}
  -> {"event":"loaded", load_s} | {"event":"error", stage, error}
  <- {"op":"describe", "image": path, "prompt": str, "max_tokens": int}      (vlm)
  <- {"op":"generate", "prompt": str, "max_tokens": int}                    (lm)
  -> {"event":"result", text, prompt_tokens, generation_tokens, generation_tps, peak_memory_gb, wall_s}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version

_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)  # anything printed by libraries lands on stderr
sys.stdout = sys.stderr


def emit(**kw: object) -> None:
    _proto.write(json.dumps(kw) + "\n")
    _proto.flush()


def _attr(obj: object, name: str) -> object:
    v = getattr(obj, name, None)
    return v if isinstance(v, (int, float)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("vlm", "lm"), required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    pkg = "mlx-vlm" if a.kind == "vlm" else "mlx-lm"
    try:
        pkg_v = version(pkg)
    except PackageNotFoundError:
        pkg_v = None
    emit(event="hello", python=sys.version.split()[0], executable=sys.executable,
         package=pkg, package_version=pkg_v)
    if pkg_v is None:
        emit(event="error", stage="import", error=f"{pkg} not installed in {sys.executable}")
        return 1

    try:
        if a.kind == "vlm":
            from mlx_vlm import generate, load  # type: ignore
            from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
        else:
            from mlx_lm import load, stream_generate  # type: ignore
    except Exception as e:  # noqa: BLE001 - report literally
        emit(event="error", stage="import", error=f"{type(e).__name__}: {e}")
        return 1

    t0 = time.perf_counter()
    try:
        model, proc = load(a.model)
    except Exception as e:  # noqa: BLE001
        emit(event="error", stage="load", error=f"{type(e).__name__}: {e}",
             trace=traceback.format_exc()[-2000:])
        return 1
    emit(event="loaded", load_s=round(time.perf_counter() - t0, 3))

    config = getattr(model, "config", None)
    if a.kind == "vlm":
        try:
            from mlx_vlm.utils import load_config  # type: ignore

            config = load_config(a.model)
        except Exception:  # noqa: BLE001 - fall back to model.config
            pass

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            t1 = time.perf_counter()
            max_tokens = int(req.get("max_tokens", 256))
            if a.kind == "vlm" and req.get("op") == "describe":
                prompt = apply_chat_template(proc, config, req["prompt"], num_images=1)
                out = generate(model, proc, prompt, [req["image"]], max_tokens=max_tokens,
                               verbose=False)
                text = out if isinstance(out, str) else getattr(out, "text", None)
                stats = None if isinstance(out, str) else out
            elif a.kind == "lm" and req.get("op") == "generate":
                msgs = [{"role": "user", "content": req["prompt"]}]
                prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
                text, stats = "", None
                for r in stream_generate(model, proc, prompt, max_tokens=max_tokens):
                    text += r.text
                    stats = r
            else:
                emit(event="error", stage="request", error=f"bad op for {a.kind}: {req.get('op')!r}")
                continue
            if not isinstance(text, str):
                emit(event="error", stage="generate", error=f"no text in result {type(out).__name__}")
                continue
            gen_tokens = _attr(stats, "generation_tokens")
            emit(event="result", text=text,
                 prompt_tokens=_attr(stats, "prompt_tokens"),
                 generation_tokens=gen_tokens,
                 generation_tps=_attr(stats, "generation_tps"),
                 peak_memory_gb=_attr(stats, "peak_memory"),
                 wall_s=round(time.perf_counter() - t1, 3),
                 # Reasoning models spend tokens in an analysis channel first: when the budget runs out
                 # the text is that analysis, not an answer. The caller must not parse it.
                 truncated=bool(gen_tokens is not None and gen_tokens >= max_tokens))
        except Exception as e:  # noqa: BLE001
            emit(event="error", stage="generate", error=f"{type(e).__name__}: {e}",
                 trace=traceback.format_exc()[-2000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
