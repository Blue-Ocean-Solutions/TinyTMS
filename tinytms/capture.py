"""Inputs for the vision/paper witnesses: an image file, or one frame from a camera (cv2.VideoCapture)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .witness import sha256_bytes


class CaptureError(RuntimeError):
    """The image or camera frame could not be obtained. Never replaced by a placeholder."""


@dataclass(frozen=True)
class Image:
    data: bytes
    sha256: str
    width: int
    height: int
    mime: str
    source: str


def _mime(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return None


def load_image(path: str | Path) -> Image:
    """JPEG/PNG file -> bytes + sha256 + size. Anything else -> CaptureError."""
    import cv2
    import numpy as np

    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise CaptureError(f"{p}: {e}") from e
    mime = _mime(data)
    if mime is None:
        raise CaptureError(f"{p.name}: not a JPEG or PNG")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise CaptureError(f"{p.name}: cv2 cannot decode the image")
    h, w = img.shape[:2]
    return Image(data=data, sha256=sha256_bytes(data), width=w, height=h, mime=mime, source=str(p))


def capture_frame(index: int = 0, path: str | Path | None = None) -> Image:
    """Grab one frame from camera `index` with cv2.VideoCapture, encode it as JPEG (and write it to `path`).

    No camera, no permission or no frame -> CaptureError. Nothing is faked.
    """
    import cv2

    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise CaptureError(f"camera {index} cannot be opened (no camera or no permission)")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise CaptureError(f"camera {index} opened but returned no frame")
    finally:
        cap.release()
    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise CaptureError("cv2 cannot encode the frame as JPEG")
    data = enc.tobytes()
    if path is not None:
        Path(path).write_bytes(data)
    h, w = frame.shape[:2]
    return Image(data=data, sha256=sha256_bytes(data), width=w, height=h, mime="image/jpeg",
                 source=str(path) if path is not None else f"camera:{index}")
