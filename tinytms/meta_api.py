"""Meta Model API client (POST {base}/responses, Bearer MODEL_API_KEY from the environment).

No key -> NoKey (the caller turns it into a NO_MODEL witness). Any transport/HTTP/JSON error ->
ApiResult(status="ERROR", detail=...). This module never fabricates a model output.

The request bodies below are an ASSUMPTION (Responses-style "input" with typed content parts) and
must be checked against the live API by the cockpit with a real key; see ENTREGA.md.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

BASE_URL = os.environ.get("TINYTMS_META_BASE_URL", "https://api.meta.ai/v1")
KEY_ENV = "MODEL_API_KEY"

# Model ids are placeholders until confirmed against the live API (override by env).
SAM_MODEL = os.environ.get("TINYTMS_SAM_MODEL", "sam-3.1")
SPARK_MODEL = os.environ.get("TINYTMS_SPARK_MODEL", "muse-spark-1.3")
VOICE_MODEL = os.environ.get("TINYTMS_VOICE_MODEL", "muse-voice-transcribe")

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0)

DOC_PROMPT = (
    "You are reading a freight document (bill of lading or Mexican Carta de Porte). "
    "Return ONLY a JSON object with exactly these keys: "
    '"pieces" (integer total handling units, or null), "weight_kg" (number, gross kg, or null), '
    '"seal" (string seal number, or null), "shipper" (string or null), "consignee" (string or null), '
    '"hazmat" (true, false, or null if not stated). Use null for anything not legible. Do not guess.'
)


class NoKey(RuntimeError):
    """MODEL_API_KEY is not set: there is no model to ask."""


@dataclass(frozen=True)
class ApiResult:
    status: Literal["OK", "ERROR"]
    model: str
    data: dict[str, Any] | None = None
    http_status: int | None = None
    detail: str = ""
    provider: str = field(default="meta-api")


def _data_url(path: str | Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_sam_request(image: str | Path, concept: str, model: str = SAM_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": _data_url(image)},
                    {"type": "input_text", "text": concept},
                ],
            }
        ],
    }


def build_spark_doc_request(image: str | Path, model: str = SPARK_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": _data_url(image)},
                    {"type": "input_text", "text": DOC_PROMPT},
                ],
            }
        ],
    }


def build_voice_request(audio: str | Path, model: str = VOICE_MODEL) -> dict[str, Any]:
    fmt = Path(audio).suffix.lstrip(".").lower() or "wav"
    b64 = base64.b64encode(Path(audio).read_bytes()).decode("ascii")
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}],
            }
        ],
    }


class MetaClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(KEY_ENV, "")
        if not key.strip():
            raise NoKey(f"{KEY_ENV} is not set")
        self._key = key.strip()
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )

    def __repr__(self) -> str:  # never leak the key
        return f"MetaClient(base_url={str(self._client.base_url)!r})"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MetaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _redact(self, s: str) -> str:
        return s.replace(self._key, "<redacted>")

    def responses(self, payload: dict[str, Any]) -> ApiResult:
        model = str(payload.get("model", ""))
        try:
            r = self._client.post("/responses", json=payload)
        except httpx.TimeoutException as e:
            return ApiResult("ERROR", model, detail=f"timeout: {type(e).__name__}")
        except httpx.HTTPError as e:
            return ApiResult("ERROR", model, detail=self._redact(f"transport: {type(e).__name__}: {e}"))
        if r.status_code != 200:
            body = self._redact(r.text[:500])
            return ApiResult("ERROR", model, http_status=r.status_code, detail=f"HTTP {r.status_code}: {body}")
        try:
            data = r.json()
        except ValueError:
            return ApiResult("ERROR", model, http_status=200, detail="response is not JSON")
        if not isinstance(data, dict):
            return ApiResult("ERROR", model, http_status=200, detail="response JSON is not an object")
        return ApiResult("OK", model, data=data, http_status=200)

    def sam_segment(self, image: str | Path, concept: str = "pallet") -> ApiResult:
        return self.responses(build_sam_request(image, concept))

    def spark_extract_document(self, image: str | Path) -> ApiResult:
        return self.responses(build_spark_doc_request(image))

    def voice_transcribe(self, audio: str | Path) -> ApiResult:
        return self.responses(build_voice_request(audio))


# Module-level conveniences: raise NoKey when the environment has no key.
def sam_segment(image: str | Path, concept: str = "pallet") -> ApiResult:
    with MetaClient() as c:
        return c.sam_segment(image, concept)


def spark_extract_document(image: str | Path) -> ApiResult:
    with MetaClient() as c:
        return c.spark_extract_document(image)


def voice_transcribe(audio: str | Path) -> ApiResult:
    with MetaClient() as c:
        return c.voice_transcribe(audio)
