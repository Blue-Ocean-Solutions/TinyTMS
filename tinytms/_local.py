"""Shared plumbing for local MLX models: offline cache lookup + an isolated worker process."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PYTHON_ENV = "TINYTMS_LOCAL_PYTHON"
WORKER = Path(__file__).with_name("_mlx_worker.py")


class LocalModelError(RuntimeError):
    """Explicit failure of a local model (not cached, import/load crash, timeout...)."""


class ModelNotCached(LocalModelError):
    """The weights are not on disk; nothing was downloaded."""


def local_python() -> str:
    return os.environ.get(PYTHON_ENV) or sys.executable


def hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_cached_model(repo: str, override_env: str | None = None) -> Path:
    """Local snapshot dir for an HF repo, or LocalModelError. Never downloads."""
    if override_env and os.environ.get(override_env):
        p = Path(os.environ[override_env])
        if (p / "config.json").is_file():
            return p
        raise ModelNotCached(f"{override_env}={p} has no config.json")
    root = hf_hub_cache() / ("models--" + repo.replace("/", "--"))
    snaps = root / "snapshots"
    ref = root / "refs" / "main"
    candidates: list[Path] = []
    if ref.is_file():
        candidates.append(snaps / ref.read_text().strip())
    if snaps.is_dir():
        candidates += sorted(p for p in snaps.iterdir() if p.is_dir())
    for c in candidates:
        if (c / "config.json").is_file() and any(c.glob("*.safetensors")):
            return c
    raise ModelNotCached(
        f"model {repo} not in local cache ({root}); this adapter does not download. "
        f"Fetch it beforehand or set {override_env or 'HF_HUB_CACHE'}."
    )


def package_version(package: str, python: str | None = None, timeout: float = 30.0) -> str | None:
    """Version of `package` in the target interpreter, via metadata only (no import of mlx)."""
    code = (
        "import importlib.metadata as m, sys\n"
        f"try: print(m.version({package!r}))\n"
        "except m.PackageNotFoundError: sys.exit(1)\n"
    )
    try:
        r = subprocess.run([python or local_python(), "-I", "-c", code],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


@dataclass(frozen=True)
class LoadInfo:
    model_path: str
    python: str
    package: str
    package_version: str | None
    load_s: float


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int | None
    generation_tokens: int | None
    generation_tps: float | None
    peak_memory_gb: float | None
    wall_s: float


class Worker:
    """One long-lived _mlx_worker.py process speaking JSON lines."""

    def __init__(self, kind: str, model_path: Path, python: str | None = None) -> None:
        self.kind, self.model_path, self.python = kind, model_path, python or local_python()
        self._p: subprocess.Popen[str] | None = None
        self._q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stderr: list[str] = []
        self.hello: dict[str, Any] = {}

    def start(self, timeout: float) -> LoadInfo:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        cmd = [self.python, "-I", str(WORKER), "--kind", self.kind, "--model", str(self.model_path)]
        try:
            self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=env, bufsize=1)
        except OSError as e:
            raise LocalModelError(f"cannot start {self.python}: {e}") from e
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()
        self.hello = self._next(timeout, expect="hello")
        loaded = self._next(timeout, expect="loaded")
        return LoadInfo(str(self.model_path), self.hello.get("executable", self.python),
                        self.hello.get("package", ""), self.hello.get("package_version"),
                        float(loaded["load_s"]))

    def request(self, req: dict[str, Any], timeout: float) -> Generation:
        if self._p is None or self._p.stdin is None or self._p.poll() is not None:
            raise LocalModelError("worker not running; call load() first")
        self._p.stdin.write(json.dumps(req) + "\n")
        self._p.stdin.flush()
        r = self._next(timeout, expect="result")
        return Generation(r["text"], r.get("prompt_tokens"), r.get("generation_tokens"),
                          r.get("generation_tps"), r.get("peak_memory_gb"), float(r["wall_s"]))

    def close(self) -> None:
        if self._p is not None:
            if self._p.stdin:
                self._p.stdin.close()
            try:
                self._p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._p.kill()
            self._p = None

    def _pump_out(self) -> None:
        assert self._p and self._p.stdout
        for line in self._p.stdout:
            try:
                self._q.put(json.loads(line))
            except ValueError:
                self._stderr.append("[stdout] " + line)
        self._q.put(None)

    def _pump_err(self) -> None:
        assert self._p and self._p.stderr
        for line in self._p.stderr:
            self._stderr.append(line)
            del self._stderr[:-40]

    def _next(self, timeout: float, expect: str) -> dict[str, Any]:
        try:
            msg = self._q.get(timeout=timeout)
        except queue.Empty:
            self.close_hard()
            raise LocalModelError(f"timeout after {timeout}s waiting for '{expect}'") from None
        if msg is None:
            rc = self._p.wait() if self._p else None
            tail = "".join(self._stderr[-8:]).strip()
            raise LocalModelError(f"worker exited (rc={rc}) before '{expect}': {tail[-800:]}")
        if msg.get("event") == "error":
            raise LocalModelError(f"{msg.get('stage')}: {msg.get('error')}")
        if msg.get("event") != expect:
            raise LocalModelError(f"protocol: expected '{expect}', got {msg.get('event')!r}")
        return msg

    def close_hard(self) -> None:
        if self._p is not None:
            self._p.kill()
            self._p = None
