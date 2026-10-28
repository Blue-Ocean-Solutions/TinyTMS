"""Local vision-language model: Muse Glimmer 30B (4-bit) via mlx_vlm, in an isolated process.

- Never downloads: if the model is not in the HF cache -> LocalModelError.
- mlx_vlm is imported only inside the worker process, with the interpreter from
  TINYTMS_LOCAL_PYTHON (default: this one). Status measured 2026-10-28 by L1: the house venv's
  mlx_vlm does not load Glimmer ("muse_glimmer not supported"); point TINYTMS_LOCAL_PYTHON at a
  venv whose mlx_vlm does.
"""

from __future__ import annotations

import re
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
    "How many {concept}s are fully or partly visible in this image? "
    "Answer with a single integer and nothing else. If you cannot tell, answer: unknown"
)
_SINGLE_INT = re.compile(r"\s*(\d{1,4})\s*\.?\s*")

# Glimmer reasons in an analysis channel before its final answer. Measured 2026-10-28 (L1, M5 Max, mlx_vlm
# 0.7.10): the count answer took 241 tokens of analysis before the final "1"; the document JSON needed more
# than 512. With the budget exhausted the worker returns the analysis (" to=self<|message|>" + the prompt
# echoed + reasoning), which must never be parsed as an answer. Budgets leave room; truncation is an ERROR.
VISION_MAX_TOKENS = 512
DOC_MAX_TOKENS = 2048
ANALYSIS_MARKERS = ("<|message|>", "to=self")


def is_incomplete(g: Generation) -> bool:
    """True when the generation ran out of budget or still carries analysis-channel residue."""
    head = g.text.lstrip()[:40]
    return bool(g.truncated) or any(m in head for m in ANALYSIS_MARKERS)


def _incomplete_detail(g: Generation, budget: int) -> str:
    return (f"incomplete: generation stopped in the analysis channel ({g.generation_tokens} of {budget} tokens); "
            f"raw={g.text[:160]!r}")


def parse_single_integer(text: str) -> int | None:
    """The answer only if it IS a single integer (an optional final period allowed); prose -> None."""
    m = _SINGLE_INT.fullmatch(text)
    return int(m.group(1)) if m else None


__all__ = ["LocalVLM", "LocalModelError", "mlx_vlm_version", "vision_local", "doc_local", "parse_single_integer",
           "is_incomplete", "VISION_MAX_TOKENS", "DOC_MAX_TOKENS"]


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
        g = vlm.describe(image, COUNT_PROMPT.format(concept=concept), max_tokens=VISION_MAX_TOKENS)
    except LocalModelError as e:
        return VisionWitness(status="ERROR", detail=f"{type(e).__name__}: {e}", **meta)
    if is_incomplete(g):
        return VisionWitness(status="ERROR", detail=_incomplete_detail(g, VISION_MAX_TOKENS), **meta)
    n = parse_single_integer(g.text)
    if n is None:
        return VisionWitness(status="OK", count=None, detail=f"not a single integer: {g.text[:200]!r}", **meta)
    return VisionWitness(status="OK", count=n, detail=f"answer={g.text.strip()!r} wall_s={g.wall_s}", **meta)


def doc_local(vlm: LocalVLM, image: str | Path) -> DocWitness:
    meta = dict(model=f"{GLIMMER_MODEL_ID}@local", provider=PROVIDER, source_sha256=sha256_file(image))
    try:
        g = vlm.describe(image, DOC_PROMPT, max_tokens=DOC_MAX_TOKENS)
    except LocalModelError as e:
        return DocWitness(status="ERROR", detail=f"{type(e).__name__}: {e}", **meta)
    if is_incomplete(g):
        return DocWitness(status="ERROR", detail=_incomplete_detail(g, DOC_MAX_TOKENS), **meta)
    try:
        data = parse_json_text(g.text)
    except ValueError as e:
        return DocWitness(status="ERROR", detail=f"{type(e).__name__}: {e}", **meta)
    if not isinstance(data, dict):
        return DocWitness(status="ERROR", detail="output JSON is not an object", **meta)
    return doc_from_fields(data, **meta)
