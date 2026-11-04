"""Draw a SAM receipt's boxes on its own photo, only when the receipt verifies.

    python scripts/receipt_boxes.py receipts/<live-sam-...>.json --out <frame>.png [--image <photo>]

Nothing is drawn, and the exit code is 1, unless all of this holds:
- the signature verifies and its signer is in keys/known_signers.txt (the same check as scripts/receipt_show.py);
- the receipt is a SAM witness (kind live-witness, witness sam) whose answer is OK;
- it holds exactly result.count boxes (x1, y1, x2, y2, score), each one inside the image_w x image_h it reported
  (SAM gives the size inside each box record, so a count-0 answer has no size and needs none);
- the photo is the receipt's input: input.path inside this repo (or --image), with the sha256 the receipt signed and
  the size the witness reported, and no EXIF rotation (which would leave the boxes' frame ambiguous). The photo is
  read once, and those same bytes are checked and drawn.

The frame is that photo under a caption band that names the model, the concept, the count and the receipt, with every
box outlined in one colour and its score. With count 0 not one pixel of the photo changes. Exit 3 when a file cannot
be read, 2 on a bad command line (--out must be a .png and never the photo or the receipt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from receipt_show import KNOWN_SIGNERS, check  # noqa: E402
from tinytms.receipts_index import load_known_signers  # noqa: E402

BOX_BGR = (255, 255, 0)  # cyan: it reads on brown cardboard and on a dark screen alike
INK_BGR = (255, 255, 255)
BAND_BGR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

Box = tuple[float, float, float, float, "float | None"]


class Refused(Exception):
    """The receipt cannot back this drawing, so nothing is drawn."""


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def boxes_from(receipt: dict[str, Any]) -> list[Box]:
    """The receipt's own boxes, checked against its own count and image size."""
    if receipt.get("kind") != "live-witness" or receipt.get("witness") != "sam":
        raise Refused("not a SAM witness receipt (kind live-witness, witness sam)")
    r = receipt.get("result")
    if not isinstance(r, dict) or r.get("status") != "OK":
        raise Refused(f"the witness status is {r.get('status') if isinstance(r, dict) else None}, not OK")
    w, h, raw, count = r.get("image_w"), r.get("image_h"), r.get("boxes"), r.get("count")
    if not isinstance(raw, list):
        raise Refused("the receipt holds no box list")
    if not (isinstance(count, int) and not isinstance(count, bool)) or count != len(raw):
        raise Refused(f"count {count!r} but {len(raw)} boxes")
    # SAM reports the image size inside each box record: a count-0 answer has none, and needs none.
    if raw and not (_number(w) and _number(h) and w > 0 and h > 0):
        raise Refused("the receipt has boxes but reports no image size to place them in")
    boxes: list[Box] = []
    for b in raw:
        if not isinstance(b, list) or len(b) not in (4, 5):
            raise Refused(f"a box is not four corners and a score: {b!r}")
        if not all(_number(x) for x in b):
            raise Refused(f"a box holds something that is not a number: {b!r}")
        x1, y1, x2, y2 = b[:4]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise Refused(f"a box is empty or outside the {w}x{h} image: {b!r}")
        boxes.append((float(x1), float(y1), float(x2), float(y2), float(b[4]) if len(b) == 5 else None))
    return boxes


def photo_path(receipt: dict[str, Any], image: str | None) -> Path:
    """--image when given; otherwise the receipt's input.path, which must stay inside this repo."""
    if image:
        return Path(image)
    inp = receipt.get("input")
    rel = inp.get("path") if isinstance(inp, dict) else None
    if not isinstance(rel, str) or not rel:
        raise Refused("the receipt names no input photo; pass --image")
    p = (REPO / rel).resolve()
    if Path(rel).is_absolute() or not p.is_relative_to(REPO):
        raise Refused(f"the receipt's input {rel!r} is outside the repo; pass --image")
    return p


def check_photo(receipt: dict[str, Any], data: bytes) -> np.ndarray:
    """The photo decoded from `data`, once its sha256 and its size are the ones the receipt holds."""
    inp = receipt.get("input")
    want = str(inp.get("sha256", "") if isinstance(inp, dict) else "").lower()
    got = hashlib.sha256(data).hexdigest()
    also = receipt["result"].get("source_sha256")
    if got != want or (also is not None and str(also).lower() != got):
        raise Refused(f"photo sha256 {got[:12]} is not the receipt's {want[:12] or '-'}")
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise Refused("the photo does not decode as an image")
    # An EXIF rotation turns the pixels at decode time, and a 180° turn keeps the size: nothing in the receipt says
    # which way up SAM saw the photo, so its boxes have no frame we can trust.
    raw = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if raw is None or raw.shape != img.shape or not np.array_equal(raw, img):
        raise Refused("the photo carries an EXIF rotation; strip its metadata and take a new receipt")
    h, w = img.shape[:2]
    rw, rh = receipt["result"].get("image_w"), receipt["result"].get("image_h")
    if (rw, rh) != (None, None) and (w, h) != (rw, rh):  # none only when there is no box to place
        raise Refused(f"photo size {w}x{h} is not the {rw}x{rh} the witness reported")
    return img


def frame(img: np.ndarray, boxes: list[Box], caption: str) -> np.ndarray:
    """The photo under a caption band, its boxes outlined. Scores go first and outlines over them, so no label ever
    hides an outline; outside the boxes and their scores, the photo's pixels are the photo's."""
    h, w = img.shape[:2]
    thick = max(2, round(min(w, h) / 240))
    scale = max(0.5, min(w, h) / 1200)
    cap = scale
    while cap > 0.3 and cv2.getTextSize(caption, FONT, cap, 2)[0][0] > w - 2 * (thick + 4):
        cap *= 0.9
    (_, th), base = cv2.getTextSize(caption, FONT, cap, 2)
    band = th + base + 2 * (thick + 4)
    canvas = np.zeros((h + band, w, 3), np.uint8)
    canvas[:band] = BAND_BGR
    canvas[band:] = img
    cv2.putText(canvas, caption, (thick + 4, thick + 4 + th), FONT, cap, INK_BGR, 2, cv2.LINE_AA)
    for x1, y1, _x2, _y2, score in boxes:
        if score is None:
            continue
        label = f"{score:.2f}"
        (tw, lh), lb = cv2.getTextSize(label, FONT, scale, 2)
        top = max(band, band + int(y1) - lh - lb - 6)  # above the box, or just inside it at the photo's top edge
        cv2.rectangle(canvas, (int(x1), top), (int(x1) + tw + 6, top + lh + lb + 6), BAND_BGR, -1)
        cv2.putText(canvas, label, (int(x1) + 3, top + lh + 3), FONT, scale, INK_BGR, 2, cv2.LINE_AA)
    for x1, y1, x2, y2, _score in boxes:
        cv2.rectangle(canvas, (int(x1), band + int(y1)), (int(x2), band + int(y2)), BOX_BGR, thick, cv2.LINE_8)
    return canvas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("receipt")
    ap.add_argument("--out", required=True, help="the frame to write, a .png")
    ap.add_argument("--image", help="the photo, when it is not at the receipt's input.path")
    ap.add_argument("--known-signers", default=str(KNOWN_SIGNERS),
                    help="allowlist of signer keys (default: %(default)s)")
    args = ap.parse_args(argv)
    out, src = Path(args.out), Path(args.receipt)
    if out.suffix.lower() != ".png":
        ap.error("--out must be a .png")
    try:
        raw = src.read_bytes()
        known = load_known_signers(args.known_signers)
    except (OSError, ValueError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    ok, line = check(raw, known)
    print(f"{src.name}: {line}")
    if not ok:
        return 1
    receipt = json.loads(raw)
    try:
        boxes = boxes_from(receipt)
        photo = photo_path(receipt, args.image)
        if out.resolve() in (photo.resolve(), src.resolve()):
            ap.error("--out would overwrite the photo or the receipt")
        try:
            data = photo.read_bytes()
        except OSError as e:
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        img = check_photo(receipt, data)
    except Refused as e:
        print(f"refused, nothing drawn: {e}")
        return 1
    model = receipt.get("model") or receipt["result"].get("model") or "-"
    concept = receipt.get("concept") or receipt["result"].get("concept") or "-"
    caption = f"{model} | {concept!r} | count {len(boxes)} | {src.name}".encode("ascii", "replace").decode()
    encoded, png = cv2.imencode(".png", frame(img, boxes, caption))
    if not encoded:
        print("ERROR: the frame did not encode as PNG", file=sys.stderr)
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".partial", dir=out.parent)  # a new file, never a link
    with os.fdopen(fd, "wb") as f:
        f.write(png.tobytes())
    os.replace(partial, out)  # a frame is whole or absent, never half-written
    shown = args.image or receipt["input"]["path"]
    print(f"image  {shown}  sha256 {hashlib.sha256(data).hexdigest()[:12]} = the receipt's")
    print(f"drew {len(boxes)} box{'' if len(boxes) == 1 else 'es'} ({model}, {concept!r}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
