"""Local vision-language model: Muse Glimmer 30B (4-bit) via mlx_vlm, in an isolated process.

- Never downloads: if the model is not in the HF cache -> LocalModelError.
- mlx_vlm is imported only inside the worker process, with the interpreter from
  TINYTMS_LOCAL_PYTHON (default: this one). Status measured 2026-10-28 by L1: the house venv's
  mlx_vlm does not load Glimmer ("muse_glimmer not supported"); point TINYTMS_LOCAL_PYTHON at a
  venv whose mlx_vlm does.
"""

from __future__ import annotations

from pathlib import Path

from ._local import (
    Generation,
    LoadInfo,
    LocalModelError,
    Worker,
    local_python,
    package_version,
    resolve_cached_model,
)
from .adapters import doc_from_fields, parse_json_text
from .meta_api import DOC_PROMPT
from .witness import DocWitness, VisionWitness, sha256_file

GLIMMER_REPO = "mlx-community/Muse-Glimmer-30B-4bit"
GLIMMER_MODEL_ID = "muse-glimmer-30b-4bit"
PROVIDER = "local-mlx"
PATH_ENV = "TINYTMS_GLIMMER_PATH"

COUNT_PROMPT = (
    'Count the {concept}s fully or partly visible in this image. Return ONLY JSON: {{"count": <integer>}}. '
    'If you cannot count them reliably return {{"count": null}}.'
)

__all__ = ["LocalVLM", "LocalModelError", "mlx_vlm_version", "vision_local", "doc_local"]


def mlx_vlm_version(python: str | None = None) -> str | None:
    """mlx-vlm version in the local interpreter (metadata only; does not import mlx)."""
    return package_version("mlx-vlm", python)


class LocalVLM:
    def __init__(self, repo: str = GLIMMER_REPO, python: str | None = None,
                 load_timeout: float = 900.0, gen_timeout: float = 600.0) -> None:
        self.repo, self.python = repo, python or local_python()
        self.load_timeout, self.gen_timeout = load_timeout, gen_timeout
        self._w: Worker | None = None
        self.info: LoadInfo | None = None

    def load(self) -> LoadInfo:
        path = resolve_cached_model(self.repo, PATH_ENV)  # raises if not cached
        self._w = Worker("vlm", path, self.python)
        try:
            self.info = self._w.start(self.load_timeout)
        except LocalModelError:
            self._w.close_hard()
            self._w = None
            raise
        return self.info

    def describe(self, image: str | Path, prompt: str, max_tokens: int = 256) -> Generation:
        if self._w is None:
            raise LocalModelError("call load() first")
        return self._w.request(
            {"op": "describe", "image": str(Path(image).resolve()), "prompt": prompt,
             "max_tokens": max_tokens},
            self.gen_timeout,
        )

    def close(self) -> None:
        if self._w is not None:
            self._w.close()
            self._w = None

    def __enter__(self) -> LocalVLM:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def vision_local(vlm: LocalVLM, image: str | Path, concept: str = "pallet") -> VisionWitness:
    meta = dict(model=f"{GLIMMER_MODEL_ID}@local", provider=PROVIDER,
                source_sha256=sha256_file(image), concept=concept)
    try:
        g = vlm.describe(image, COUNT_PROMPT.format(concept=concept), max_tokens=32)
        data = parse_json_text(g.text)
    except (LocalModelError, ValueError) as e:
        return VisionWitness(status="ERROR", detail=f"{type(e).__name__}: {e}", **meta)
    n = data.get("count") if isinstance(data, dict) else None
    if not isinstance(n, int) or isinstance(n, bool):
        return VisionWitness(status="OK", count=None, detail=f"no count in output: {g.text[:200]!r}", **meta)
    return VisionWitness(status="OK", count=n, detail=f"wall_s={g.wall_s}", **meta)


def doc_local(vlm: LocalVLM, image: str | Path) -> DocWitness:
    meta = dict(model=f"{GLIMMER_MODEL_ID}@local", provider=PROVIDER, source_sha256=sha256_file(image))
    try:
        g = vlm.describe(image, DOC_PROMPT, max_tokens=256)
        data = parse_json_text(g.text)
    except (LocalModelError, ValueError) as e:
        return DocWitness(status="ERROR", detail=f"{type(e).__name__}: {e}", **meta)
    if not isinstance(data, dict):
        return DocWitness(status="ERROR", detail="output JSON is not an object", **meta)
    return doc_from_fields(data, **meta)
