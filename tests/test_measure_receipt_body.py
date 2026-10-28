"""The local-VLM measurement receipt must describe the run without describing the machine."""

import importlib.util
from pathlib import Path

from tinytms.receipt import MACHINE_PATH_PREFIXES, canonical_json

ROOT = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("measure_local_vlm", ROOT / "scripts" / "measure_local_vlm.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_receipt_body_has_no_machine_paths(tmp_path):
    m = _script()
    repo = tmp_path / "repo"
    (repo / "evals" / "public").mkdir(parents=True)
    img = repo / "evals" / "public" / "x.jpg"
    img.write_bytes(b"\xff\xd8fake-jpeg")

    body = m.receipt_body(
        repo_root=repo,
        image=img,
        prompt=m.DEFAULT_PROMPT,
        max_tokens=128,
        runs_requested=1,
        interpreter_env="TINYTMS_LOCAL_PYTHON",
        interpreter_version="3.12.0",
        mlx_vlm_version="0.7.10",
        load={
            "load_s_in_worker": 1.0,
            "load_s_incl_process_start": 2.0,
            "model_path": "/Users/someone/.cache/huggingface/hub/models--x/snapshots/abc123",
            "worker_python": "/Users/someone/venv/bin/python",
            "package_version": "0.7.10",
        },
        runs=[{"text": "t", "prompt_tokens": 1, "generation_tokens": 2, "generation_tps": 3.0,
               "peak_memory_gb": 4.0, "wall_s": 5.0}],
        status="OK",
    )

    text = canonical_json(body).decode("utf-8")
    assert not any(p in text for p in MACHINE_PATH_PREFIXES), text
    assert body["image"] == {"path": "evals/public/x.jpg", "sha256": body["image"]["sha256"]}
    assert len(body["image"]["sha256"]) == 64
    assert body["command"] == "python scripts/measure_local_vlm.py --image evals/public/x.jpg"
    assert body["model"]["snapshot"] == "abc123"
    assert body["interpreter"] == {"env": "TINYTMS_LOCAL_PYTHON", "python": "3.12.0"}
    assert "model_path" not in body["load"] and "worker_python" not in body["load"]
    assert body["load"]["package_version"] == "0.7.10"
    assert body["status"] == "OK" and body["runs"][0]["generation_tps"] == 3.0


def test_receipt_body_records_non_default_arguments(tmp_path):
    m = _script()
    repo = tmp_path / "repo"
    (repo / "evals").mkdir(parents=True)
    img = repo / "evals" / "y.png"
    img.write_bytes(b"\x89PNGfake")
    body = m.receipt_body(
        repo_root=repo, image=img, prompt="Count the boxes.", max_tokens=64, runs_requested=3,
        interpreter_env="TINYTMS_LOCAL_PYTHON", interpreter_version="3.12.0", mlx_vlm_version="0.7.10",
        load=None, runs=[], status="ERROR", error="LocalModelError: no model",
    )
    assert body["command"] == "python scripts/measure_local_vlm.py --image evals/y.png --prompt 'Count the boxes.' --max-tokens 64 --runs 3"
    assert body["status"] == "ERROR" and body["error"] == "LocalModelError: no model"
    assert "load" not in body
