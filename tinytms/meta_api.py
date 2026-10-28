"""Meta Model API client, built to the shapes MEASURED on 2026-10-28 (see FORMAS-MEDIDAS.md upstream).

- SAM 3.1:   POST {base}/responses, "stream": true, Accept: text/event-stream; read SSE line by line.
- Spark 1.3: POST {base}/responses, no stream, max_output_tokens 800; text in output[type=message].
- Voice:     POST {base}/asr/transcribe, multipart: part "request" (JSON) + part "audio" (WAV mono PCM 16/24 kHz).

No key -> NoKey (the caller turns it into a NO_MODEL witness). Any transport/HTTP/parse error ->
ApiResult(status="ERROR", detail=...) with the key redacted. This module never fabricates a model output.
Images are downscaled to <= 1280 px on the long side before sending; the sha256 of the ORIGINAL file is kept.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx

BASE_URL = os.environ.get("TINYTMS_META_BASE_URL", "https://api.meta.ai/v1")
KEY_ENV = "MODEL_API_KEY"

SAM_MODEL = os.environ.get("TINYTMS_SAM_MODEL", "sam-3.1")
SPARK_MODEL = os.environ.get("TINYTMS_SPARK_MODEL", "muse-spark-1.3")
VOICE_MODEL = os.environ.get("TINYTMS_VOICE_MODEL", "muse-voice-transcribe-1.0")

# Reasoning is spent first (API minimum 16). 800 was measured too short on 2026-10-28: 748 reasoning tokens,
# status "incomplete", JSON cut (receipts/live-spark-20261028T152338Z.json). 1600 leaves room.
SPARK_MAX_OUTPUT_TOKENS = int(os.environ.get("TINYTMS_SPARK_MAX_OUTPUT_TOKENS", "1600"))
MAX_IMAGE_SIDE = 1280
VOICE_RATES = (16000, 24000)
VOICE_LANGUAGE_BIAS = ["Spanish", "English"]

# Per-model call caps for one call log (credits are real money). Over the cap -> ERROR, no request sent.
DEFAULT_CAPS: dict[str, int] = {SAM_MODEL: 60, SPARK_MODEL: 60, VOICE_MODEL: 20}
# Optional per-lane caps on the same ledger: TINYTMS_LANE names the lane (stamped on every record) and
# TINYTMS_LANE_CAPS="sam-3.1=20,muse-spark-1.3=25" caps that lane's own records only.
LANE_ENV, LANE_CAPS_ENV = "TINYTMS_LANE", "TINYTMS_LANE_CAPS"


def parse_caps(spec: str) -> dict[str, int]:
    """'model=n,model=n' -> {model: n}. A malformed entry raises ValueError: a cap is never silently dropped."""
    out: dict[str, int] = {}
    for part in (x.strip() for x in spec.split(",")):
        if not part:
            continue
        model, sep, n = part.rpartition("=")
        if not sep or not model.strip() or not n.strip().isdigit():
            raise ValueError(f"bad cap entry {part!r}; expected model=integer")
        out[model.strip()] = int(n)
    return out

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


class MediaError(ValueError):
    """An input file cannot be turned into what the API accepts (bad image, non-RIFF audio, ...)."""


@dataclass(frozen=True)
class ApiResult:
    status: Literal["OK", "ERROR"]
    model: str
    data: dict[str, Any] | None = None  # parsed JSON body (Spark, Voice)
    http_status: int | None = None
    detail: str = ""
    provider: str = field(default="meta-api")
    raw_text: str | None = None  # SAM: the SSE body as received (contains mask data; never write it whole)
    seconds: float | None = None
    utc: str = ""
    request: dict[str, Any] | None = None  # the request WITHOUT base64 payloads
    media: dict[str, Any] | None = None  # what was sent vs. the original (sha256, size, conversion)


def is_transport_failure(r: ApiResult) -> bool:
    """The API was unreachable or answered non-200 (network, timeout, HTTP). A 200 with a bad body,
    a media error or a budget refusal is NOT a transport failure: those are answers, not outages."""
    if r.status != "ERROR":
        return False
    if r.http_status is not None:
        return r.http_status != 200
    return r.detail.startswith(("timeout:", "transport:"))


# --- media ---------------------------------------------------------------------------------


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def prepare_image(path: str | Path, max_side: int = MAX_IMAGE_SIDE) -> tuple[bytes, str, dict[str, Any]]:
    """JPEG/PNG bytes to send (downscaled with cv2 if the long side > max_side), mime, and media info."""
    import cv2
    import numpy as np

    raw = Path(path).read_bytes()
    if raw[:3] == b"\xff\xd8\xff":
        mime, ext = "image/jpeg", ".jpg"
    elif raw[:8] == b"\x89PNG\r\n\x1a\n":
        mime, ext = "image/png", ".png"
    else:
        raise MediaError(f"{Path(path).name}: not a JPEG or PNG")
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise MediaError(f"{Path(path).name}: cv2 cannot decode the image")
    h, w = img.shape[:2]
    info: dict[str, Any] = {"original_sha256": _sha(raw), "original_wh": [w, h], "mime": mime}
    out = raw
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == ".jpg" else []
        ok, enc = cv2.imencode(ext, small, params)
        if not ok:
            raise MediaError(f"{Path(path).name}: cv2 cannot re-encode the resized image")
        out, w, h = enc.tobytes(), nw, nh
    info.update(sent_sha256=_sha(out), sent_wh=[w, h], sent_bytes=len(out), resized=out is not raw)
    return out, mime, info


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def prepare_wav(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    """WAV bytes the ASR accepts: RIFF, mono, PCM16, 16 or 24 kHz. Converts with wave + numpy if needed."""
    import numpy as np

    raw = Path(path).read_bytes()
    info: dict[str, Any] = {"original_sha256": _sha(raw)}
    try:
        with wave.open(io.BytesIO(raw)) as w:
            ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            comp = w.getcomptype()
            frames = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise MediaError(f"{Path(path).name}: not a RIFF/PCM WAV ({e}); convert it first") from e
    if comp != "NONE":
        raise MediaError(f"{Path(path).name}: compressed WAV ({comp}) not supported")
    info["original"] = {"channels": ch, "sample_width": sw, "rate": rate, "frames": n}
    if ch == 1 and sw == 2 and rate in VOICE_RATES:
        info.update(converted=False, sent_sha256=info["original_sha256"], sent={"channels": 1, "rate": rate})
        return raw, info
    if sw == 1:
        x = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        x = np.where(v & 0x800000, v - 0x1000000, v).astype(np.float64) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise MediaError(f"{Path(path).name}: unsupported sample width {sw}")
    x = x.reshape(-1, ch).mean(axis=1)
    target = rate if rate in VOICE_RATES else 16000
    if target != rate and len(x):
        m = max(1, round(len(x) * target / rate))
        x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(target)
        o.writeframes(pcm)
    out = buf.getvalue()
    info.update(converted=True, sent_sha256=_sha(out), sent={"channels": 1, "rate": target})
    return out, info


# --- request bodies ------------------------------------------------------------------------


def _message(text: str, image_url: str) -> list[dict[str, Any]]:
    return [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": text},
        {"type": "input_image", "image_url": image_url},
    ]}]


def build_sam_request(image_url: str, concept: str, model: str = SAM_MODEL) -> dict[str, Any]:
    return {"model": model, "stream": True, "metadata": {"include_confidence": "true"},
            "input": _message(concept, image_url)}


def build_spark_doc_request(image_url: str, model: str = SPARK_MODEL, prompt: str = DOC_PROMPT) -> dict[str, Any]:
    return {"model": model, "max_output_tokens": SPARK_MAX_OUTPUT_TOKENS, "input": _message(prompt, image_url)}


def build_voice_request(model: str = VOICE_MODEL) -> dict[str, Any]:
    return {"model": model, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK", "languageBias": list(VOICE_LANGUAGE_BIAS)}


_DATA_URL = re.compile(r"data:([\w/+.-]+);base64,[A-Za-z0-9+/=]+")


def redact_request(obj: Any) -> Any:
    """Copy of a request body with every data: URL replaced by '<data:mime;base64 N chars sha256=…>'."""
    if isinstance(obj, dict):
        return {k: redact_request(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_request(v) for v in obj]
    if isinstance(obj, str):
        return _DATA_URL.sub(lambda m: f"<data:{m.group(1)};base64 {len(m.group(0))} chars "
                             f"sha256={_sha(m.group(0).encode())[:16]}…>", obj)
    return obj


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --- client --------------------------------------------------------------------------------


class MetaClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
        call_log: str | Path | None = None,
        caps: dict[str, int] | None = None,
        origin: str = "",
        lane: str | None = None,
        lane_caps: dict[str, int] | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(KEY_ENV, "")
        if not key.strip():
            raise NoKey(f"{KEY_ENV} is not set")
        self._key = key.strip()
        self._call_log = Path(call_log) if call_log else None
        self._caps = DEFAULT_CAPS if caps is None else caps
        self._origin = origin
        self._lane = (os.environ.get(LANE_ENV, "") if lane is None else lane).strip()
        self._lane_caps = parse_caps(os.environ.get(LANE_CAPS_ENV, "")) if lane_caps is None else lane_caps
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {self._key}"},
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

    # call log + budget ------------------------------------------------------------------

    def _logged(self) -> list[dict[str, Any]]:
        if self._call_log is None or not self._call_log.exists():
            return []
        try:
            data = json.loads(self._call_log.read_text(encoding="utf-8"))
        except ValueError:
            return []
        return data.get("llamadas", []) if isinstance(data, dict) else []

    def _ledger_lanes(self) -> dict[str, Any]:
        try:
            data = json.loads(self._call_log.read_text(encoding="utf-8")) if self._call_log else {}
        except (OSError, ValueError):
            return {}
        lanes = data.get("lanes") if isinstance(data, dict) else None
        return lanes if isinstance(lanes, dict) else {}

    def _over_cap(self, model: str) -> str:
        if self._call_log is None:
            return ""
        logged = self._logged()
        cap = self._caps.get(model)
        n = sum(1 for c in logged if c.get("modelo") == model)
        if cap is not None and n >= cap:
            return f"budget cap reached for {model}: {n}/{cap} calls in {self._call_log.name}"
        cap = self._lane_caps.get(model) if self._lane else None
        n = sum(1 for c in logged if c.get("modelo") == model and c.get("lane") == self._lane)
        if cap is not None and n >= cap:
            return f"lane budget cap reached for {model}: {n}/{cap} calls by {self._lane} in {self._call_log.name}"
        return ""

    def _log(self, model: str, endpoint: str, utc: str, http: int | None, s: float, note: str = "") -> None:
        if self._call_log is None:
            return
        calls = self._logged()
        rec = {"modelo": model, "endpoint": endpoint, "utc": utc, "http": http, "s": s}
        if self._origin:
            rec["origen"] = self._origin
        if self._lane:
            rec["lane"] = self._lane
        if note:
            rec["nota"] = self._redact(note)[:300]
        calls.append(rec)
        counts: dict[str, int] = {}
        lanes: dict[str, dict[str, Any]] = {}
        for c in calls:
            counts[c["modelo"]] = counts.get(c["modelo"], 0) + 1
            if c.get("lane"):
                lc = lanes.setdefault(c["lane"], {}).setdefault("cuenta", {})
                lc[c["modelo"]] = lc.get(c["modelo"], 0) + 1
        prev = self._ledger_lanes()
        for name, v in lanes.items():
            caps = self._lane_caps if name == self._lane and self._lane_caps else prev.get(name, {}).get("tope")
            if caps:
                v["tope"] = caps
        self._call_log.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {"tope": self._caps, "cuenta": counts}
        if lanes:
            body["lanes"] = lanes
        body["llamadas"] = calls
        self._call_log.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # transport --------------------------------------------------------------------------

    def _send(self, model: str, endpoint: str, request: dict[str, Any] | None, media: dict[str, Any] | None,
              stream: bool, **kw: Any) -> ApiResult:
        base = dict(request=request, media=media)
        why = self._over_cap(model)
        if why:
            return ApiResult("ERROR", model, detail=why, **base)
        utc, t0 = _utc(), time.perf_counter()
        status: int | None = None
        try:
            if stream:
                lines: list[str] = []
                with self._client.stream("POST", endpoint, **kw) as r:
                    status = r.status_code
                    if status == 200:
                        for line in r.iter_lines():
                            lines.append(line)
                        text = "\n".join(lines)
                    else:
                        r.read()
                        text = r.text
            else:
                r = self._client.post(endpoint, **kw)
                status, text = r.status_code, r.text
        except httpx.TimeoutException as e:
            s = round(time.perf_counter() - t0, 3)
            self._log(model, endpoint, utc, status, s, f"timeout: {type(e).__name__}")
            return ApiResult("ERROR", model, detail=f"timeout: {type(e).__name__}", seconds=s, utc=utc, **base)
        except httpx.HTTPError as e:
            s = round(time.perf_counter() - t0, 3)
            d = self._redact(f"transport: {type(e).__name__}: {e}")
            self._log(model, endpoint, utc, status, s, d)
            return ApiResult("ERROR", model, detail=d, seconds=s, utc=utc, **base)
        s = round(time.perf_counter() - t0, 3)
        timing = dict(seconds=s, utc=utc, http_status=status)
        if status != 200:
            body = self._redact(text[:500])
            self._log(model, endpoint, utc, status, s, body)
            return ApiResult("ERROR", model, detail=f"HTTP {status}: {body}", **timing, **base)
        self._log(model, endpoint, utc, status, s)
        if stream:
            return ApiResult("OK", model, raw_text=text, **timing, **base)
        try:
            data = json.loads(text)
        except ValueError:
            return ApiResult("ERROR", model, detail="response is not JSON", raw_text=self._redact(text[:500]),
                             **timing, **base)
        if not isinstance(data, dict):
            return ApiResult("ERROR", model, detail="response JSON is not an object", **timing, **base)
        return ApiResult("OK", model, data=data, **timing, **base)

    def responses(self, payload: dict[str, Any], media: dict[str, Any] | None = None) -> ApiResult:
        """POST /responses (JSON, no stream)."""
        model = str(payload.get("model", ""))
        return self._send(model, "/responses", redact_request(payload), media, False, json=payload)

    # the three witnesses ----------------------------------------------------------------

    def sam_segment(self, image: str | Path, concept: str = "pallet", model: str = SAM_MODEL) -> ApiResult:
        try:
            img, mime, media = prepare_image(image)
        except MediaError as e:
            return ApiResult("ERROR", model, detail=str(e))
        payload = build_sam_request(data_url(img, mime), concept, model)
        return self._send(model, "/responses", redact_request(payload), media, True, json=payload,
                          headers={"Accept": "text/event-stream"})

    def spark_extract_document(self, image: str | Path, model: str = SPARK_MODEL) -> ApiResult:
        try:
            img, mime, media = prepare_image(image)
        except MediaError as e:
            return ApiResult("ERROR", model, detail=str(e))
        return self.responses(build_spark_doc_request(data_url(img, mime), model), media)

    def voice_transcribe(self, audio: str | Path, model: str = VOICE_MODEL) -> ApiResult:
        try:
            wav, media = prepare_wav(audio)
        except MediaError as e:
            return ApiResult("ERROR", model, detail=str(e))
        req = build_voice_request(model)
        files = {
            "request": (None, json.dumps(req), "application/json"),
            "audio": ("audio.wav", wav, "audio/wav"),
        }
        shown = {"multipart": {"request": req, "audio": f"<wav {len(wav)} bytes sha256={media['sent_sha256']}>"}}
        return self._send(model, "/asr/transcribe", shown, media, False, files=files)


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
