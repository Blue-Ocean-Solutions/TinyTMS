"""Local text explainer: Llama 3.1 8B Instruct (4-bit) via mlx_lm, in an isolated process.

It turns the judge's reasons into a short message in the driver's language. It never decides:
the verdict is fixed before it runs and does not change if it fails. Same discipline as
local_vlm: no download, explicit LocalModelError, nothing simulated.
"""

from __future__ import annotations

from ._local import Generation, LoadInfo, LocalModelError, Worker, local_python, package_version, resolve_cached_model

LLAMA_REPO = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
LLAMA_MODEL_ID = "meta-llama-3.1-8b-instruct-4bit"
PATH_ENV = "TINYTMS_LLAMA_PATH"

EXPLAIN_PROMPT = (
    "A freight check-in was put ON HOLD by a deterministic rule engine. Explain to the truck "
    "driver, in {language}, in at most three short sentences, what does not match and what to do "
    "next. Use only these reasons; do not add facts, numbers or a different decision.\n"
    "Reasons:\n{reasons}"
)

__all__ = ["LocalLLM", "LocalModelError", "mlx_lm_version", "explain_prompt"]


def mlx_lm_version(python: str | None = None) -> str | None:
    return package_version("mlx-lm", python)


def explain_prompt(reasons: list[str] | tuple[str, ...], language: str = "Spanish") -> str:
    return EXPLAIN_PROMPT.format(language=language, reasons="\n".join(f"- {r}" for r in reasons))


class LocalLLM:
    def __init__(self, repo: str = LLAMA_REPO, python: str | None = None,
                 load_timeout: float = 300.0, gen_timeout: float = 120.0) -> None:
        self.repo, self.python = repo, python or local_python()
        self.load_timeout, self.gen_timeout = load_timeout, gen_timeout
        self._w: Worker | None = None
        self.info: LoadInfo | None = None

    def load(self) -> LoadInfo:
        path = resolve_cached_model(self.repo, PATH_ENV)
        self._w = Worker("lm", path, self.python)
        try:
            self.info = self._w.start(self.load_timeout)
        except LocalModelError:
            self._w.close_hard()
            self._w = None
            raise
        return self.info

    def generate(self, prompt: str, max_tokens: int = 160) -> Generation:
        if self._w is None:
            raise LocalModelError("call load() first")
        return self._w.request({"op": "generate", "prompt": prompt, "max_tokens": max_tokens},
                               self.gen_timeout)

    def explain(self, reasons: list[str] | tuple[str, ...], language: str = "Spanish") -> Generation:
        return self.generate(explain_prompt(reasons, language))

    def close(self) -> None:
        if self._w is not None:
            self._w.close()
            self._w = None

    def __enter__(self) -> LocalLLM:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
