"""Local adapters: explicit errors, isolation, no download. Never loads a real model."""

import subprocess
import sys
from pathlib import Path

import pytest

from tinytms._local import LocalModelError, ModelNotCached, resolve_cached_model
from tinytms.local_llm import LocalLLM, explain_prompt
from tinytms.local_vlm import GLIMMER_REPO, LocalVLM, mlx_vlm_version

REPO = Path(__file__).resolve().parent.parent


def test_not_cached_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("TINYTMS_GLIMMER_PATH", raising=False)
    with pytest.raises(ModelNotCached, match="does not download"):
        LocalVLM().load()
    with pytest.raises(ModelNotCached):
        LocalLLM().load()
    assert list(tmp_path.iterdir()) == []  # nothing fetched


def test_resolves_hf_snapshot_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    root = tmp_path / ("models--" + GLIMMER_REPO.replace("/", "--"))
    snap = root / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("abc123")
    (snap / "config.json").write_text("{}")
    with pytest.raises(ModelNotCached):  # incomplete: no weights
        resolve_cached_model(GLIMMER_REPO)
    (snap / "model.safetensors").write_bytes(b"")
    assert resolve_cached_model(GLIMMER_REPO) == snap


def test_worker_failure_is_reported_not_fatal(tmp_path, monkeypatch):
    # A bogus "model" dir: whatever happens in the worker (mlx_vlm missing, no Metal device abort,
    # unparseable config), the caller gets LocalModelError and keeps running.
    fake = tmp_path / "fake-model"
    fake.mkdir()
    (fake / "config.json").write_text('{"model_type": "FIXTURE-not-a-model"}')
    monkeypatch.setenv("TINYTMS_GLIMMER_PATH", str(fake))
    vlm = LocalVLM(load_timeout=120)
    with pytest.raises(LocalModelError) as ei:
        vlm.load()
    assert str(ei.value)  # literal reason carried through


def test_alternate_interpreter_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TINYTMS_LOCAL_PYTHON", str(tmp_path / "no-such-python"))
    monkeypatch.setenv("TINYTMS_GLIMMER_PATH", str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(LocalModelError, match="cannot start"):
        LocalVLM().load()
    assert mlx_vlm_version() is None


def test_version_probe_does_not_import_mlx():
    # metadata-only probe: must return quickly and never abort, installed or not
    v = mlx_vlm_version(sys.executable)
    assert v is None or isinstance(v, str)


def test_explain_prompt_carries_only_reasons():
    p = explain_prompt(["COUNT_MISMATCH doc=12 vision=11 voice=12"], "Spanish")
    assert "COUNT_MISMATCH doc=12 vision=11 voice=12" in p and "Spanish" in p


def test_live_eval_without_key_exits_3(tmp_path):
    import os

    env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
    p = subprocess.run([sys.executable, "scripts/live_eval.py", "--receipts-dir", str(tmp_path / "r")],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 3 and "MODEL_API_KEY" in p.stderr
    assert not (tmp_path / "r").exists()
