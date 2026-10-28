"""Glimmer reasons in an analysis channel before it answers. A generation cut by max_tokens is the
analysis (' to=self<|message|>' + the prompt echoed + reasoning), never the answer: it must be an
ERROR, not a parsed count or a JSON attempt. Real outputs measured on 2026-10-28 are the test data."""

import json
from pathlib import Path

from tinytms._local import Generation
from tinytms.local_vlm import DOC_MAX_TOKENS, VISION_MAX_TOKENS, doc_local, is_incomplete, vision_local

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "tests" / "data" / "real-local-glimmer-truncated-20261028.json").read_text(encoding="utf-8"))
CASES = {(c["witness"], c["truncated"]): c for c in DATA["cases"]}
LOAD = ROOT / "evals" / "public" / "boxes-pallet.jpg"
DOC = ROOT / "evals" / "doc" / "carta_sintetica.png"


class FakeVLM:
    def __init__(self, generation: Generation) -> None:
        self.generation, self.calls = generation, []

    def describe(self, image, prompt, max_tokens=256):
        self.calls.append({"image": str(image), "prompt": prompt, "max_tokens": max_tokens})
        return self.generation


def gen(case: dict, **override) -> Generation:
    kw = dict(text=case["text"], prompt_tokens=1242, generation_tokens=case["generation_tokens"],
              generation_tps=30.0, peak_memory_gb=20.7, wall_s=5.0, truncated=case["truncated"])
    kw.update(override)
    return Generation(**kw)


def test_truncated_vision_output_is_error_not_a_count():
    w = vision_local(FakeVLM(gen(CASES[("vision", True)])), LOAD)
    assert w.status == "ERROR" and w.count is None
    assert "incomplete" in w.detail and "to=self" in w.detail


def test_truncated_doc_output_is_error_not_json():
    w = doc_local(FakeVLM(gen(CASES[("doc", True)])), DOC)
    assert w.status == "ERROR" and w.pieces is None and w.seal is None
    assert "incomplete" in w.detail and "JSONDecodeError" not in w.detail


def test_complete_final_answer_parses():
    w = vision_local(FakeVLM(gen(CASES[("vision", False)])), LOAD)
    assert w.status == "OK" and w.count == 1


def test_analysis_marker_counts_as_incomplete_even_without_the_flag():
    g = gen(CASES[("vision", True)], generation_tokens=5, truncated=False)
    assert is_incomplete(g)
    assert vision_local(FakeVLM(g), LOAD).status == "ERROR"


def test_generation_defaults_to_not_truncated():
    g = Generation(text="1", prompt_tokens=1, generation_tokens=1, generation_tps=1.0, peak_memory_gb=1.0, wall_s=0.1)
    assert g.truncated is False and not is_incomplete(g)


def test_local_budgets_leave_room_for_the_analysis_channel():
    assert VISION_MAX_TOKENS >= 512 and DOC_MAX_TOKENS >= 2048
    f = FakeVLM(gen(CASES[("vision", False)]))
    vision_local(f, LOAD)
    assert f.calls[0]["max_tokens"] == VISION_MAX_TOKENS
    f = FakeVLM(Generation(text='{"pieces": 12, "weight_kg": 4800, "seal": "MX-48213", "shipper": "a", "consignee": "b", "hazmat": false}',
                           prompt_tokens=1, generation_tokens=40, generation_tps=1.0, peak_memory_gb=1.0, wall_s=0.1))
    w = doc_local(f, DOC)
    assert f.calls[0]["max_tokens"] == DOC_MAX_TOKENS and w.status == "OK" and w.pieces == 12
