# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Shared definitions for the room-classifier lesson.

Everything that must be identical between dataset building (training time)
and classification of new plans (inference time) lives here: the class list,
the mapping from CubiCasa5K room names to those classes, and the function that
turns "image + room polygon" into the square crop the network sees.

If the crop function differed between training and inference, the model would
be evaluated on inputs it was never trained on -- a classic silent ML bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- classes

# Alphabetical, so the order matches torchvision.datasets.ImageFolder.
CLASSES = ["bath", "bedroom", "entry", "garage", "kitchen", "living", "other", "outdoor", "storage"]

# CubiCasa5K SVG room names -> lesson classes. Grouping follows the dataset
# authors' own 12-class setup (floortrans/loaders/house.py, rooms_selected);
# every name not listed maps to "other".
CUBICASA_TO_CLASS = {
    "Bath": "bath", "Sauna": "bath",
    "Bedroom": "bedroom",
    "Entry": "entry", "HallWay": "entry", "DraughtLobby": "entry",
    "Garage": "garage", "CarPort": "garage",
    "Kitchen": "kitchen",
    "LivingRoom": "living", "Dining": "living", "EatingArea": "living", "Lounge": "living",
    "Outdoor": "outdoor",
    "Storage": "storage", "Closet": "storage", "DressingRoom": "storage",
}

# Output channels of the pretrained CubiCasa5K model (room head, 12 classes).
CUBICASA_MODEL_ROOMS = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                        "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
CUBICASA_MODEL_TO_CLASS = {1: "outdoor", 3: "kitchen", 4: "living", 5: "bedroom", 6: "bath",
                           7: "entry", 9: "storage", 10: "garage", 11: "other"}

# Danish names, for the annotation tool legend. ASCII spelling (ae/oe/aa) because
# OpenCV's built-in fonts cannot draw æ, ø, å.
DANISH_HINT = {
    "bath": "bad / toilet",
    "bedroom": "vaerelse / sovevaerelse",
    "entry": "entre / gang / hall",
    "garage": "garage / carport",
    "kitchen": "koekken",
    "living": "stue / spisestue / alrum",
    "other": "bryggers / kontor / teknik / andet",
    "outdoor": "altan / terrasse / have",
    "storage": "depot / garderobe / skab",
}

# BGR colours for drawing (one per class, same order as CLASSES).
CLASS_COLORS = {
    "bath": (200, 130, 0), "bedroom": (60, 160, 60), "entry": (0, 170, 230),
    "garage": (120, 120, 120), "kitchen": (40, 40, 220), "living": (180, 80, 180),
    "other": (80, 80, 140), "outdoor": (60, 200, 170), "storage": (30, 110, 170),
}

# ---------------------------------------------------------------- crops

CROP_SIZE = 256       # saved crop side (training re-crops to 224)
CONTEXT_FRAC = 0.15   # extra border around the room's bounding box, as a fraction of its longer side
DIM_ALPHA = 0.6       # how strongly pixels outside the room are faded toward white


def crop_room(image_bgr: np.ndarray, polygon_px: np.ndarray, context: str = "dim",
              size: int = CROP_SIZE, context_frac: float = CONTEXT_FRAC) -> np.ndarray | None:
    """Cut one room out of a plan image as a square `size` x `size` BGR crop.

    polygon_px: (N, 2) array of (x, y) pixel coordinates in `image_bgr`.
    context:
      "dim"  -- keep the surroundings but fade them toward white, so the network
                sees neighbouring rooms/doors yet knows which room is meant (default)
      "keep" -- plain bounding-box crop, surroundings untouched
      "none" -- surroundings replaced by white; only the room itself is visible
    The crop keeps the room's aspect ratio (a long thin hallway stays long and
    thin) and is padded with white to a square.
    """
    h, w = image_bgr.shape[:2]
    poly = np.asarray(polygon_px, dtype=np.float64)
    if poly.ndim != 2 or len(poly) < 3:
        return None
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return None
    pad = max(8.0, context_frac * max(bw, bh))
    cx0, cy0 = int(max(0, np.floor(x0 - pad))), int(max(0, np.floor(y0 - pad)))
    cx1, cy1 = int(min(w, np.ceil(x1 + pad))), int(min(h, np.ceil(y1 + pad)))
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None

    crop = image_bgr[cy0:cy1, cx0:cx1].copy()
    if context != "keep":
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        pts = np.round(poly - [cx0, cy0]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        outside = mask == 0
        if context == "dim":
            crop[outside] = (crop[outside] * (1 - DIM_ALPHA) + 255 * DIM_ALPHA).astype(np.uint8)
        else:  # "none"
            crop[outside] = 255

    ch, cw = crop.shape[:2]
    scale = size / max(ch, cw)
    nh, nw = max(1, round(ch * scale)), max(1, round(cw * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (nw, nh), interpolation=interp)
    out = np.full((size, size, 3), 255, dtype=np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out


# ---------------------------------------------------------------- room JSON files

def rooms_file_reviewed(path: Path) -> bool:
    """False only for drafts written by prelabel_rooms.py that nobody has accepted yet."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return not isinstance(data, dict) or bool(data.get("reviewed", True))


def load_rooms_json(path: Path, image_shape: tuple[int, int], require_reviewed: bool = False) -> list[dict]:
    """Read a rooms file and return [{"polygon_px": (N,2) array, "room_type": str|None, ...}, ...].

    Accepted layouts (polygons normalised to 0..1,
    or in pixels if any coordinate is > 1.5):
      {"rooms": [{"polygon": [[x, y], ...], "room_type": "kitchen"}, ...]}
      [{"polygon": [[x, y], ...]}, ...]
    room_type is optional; when it is one of CLASSES it is used as ground truth.

    Draft files (written by prelabel_rooms.py, "reviewed": false) contain the
    model's own guesses. With require_reviewed=True their room_type is dropped,
    so a guess can never be mistaken for ground truth.
    Extra per-room keys kept for review statistics: source ("auto"/"manual"),
    auto_type (the model's original guess), confidence.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rooms = data.get("rooms", []) if isinstance(data, dict) else data
    trusted = not require_reviewed or rooms_file_reviewed(path)
    h, w = image_shape[:2]
    out = []
    for r in rooms:
        poly = np.asarray(r.get("polygon", []), dtype=np.float64)
        if poly.ndim != 2 or len(poly) < 3:
            continue
        if poly.max() <= 1.5:
            poly = poly * [w, h]
        rt = r.get("room_type")
        out.append({"polygon_px": poly, "room_type": rt if (rt in CLASSES and trusted) else None,
                    "source": r.get("source", "manual"), "auto_type": r.get("auto_type"),
                    "confidence": r.get("confidence")})
    return out


def save_rooms_json(path: Path, image_name: str, image_shape: tuple[int, int], rooms: list[dict],
                    reviewed: bool = True, extra: dict | None = None) -> None:
    """Write rooms with polygons normalised to 0..1 of the image size."""
    h, w = image_shape[:2]
    out_rooms = []
    for r in rooms:
        d = {"polygon": [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in r["polygon_px"]],
             "room_type": r.get("room_type")}
        if r.get("source", "manual") != "manual":
            d.update({"source": r["source"], "auto_type": r.get("auto_type"), "confidence": r.get("confidence")})
        out_rooms.append(d)
    payload = {"image": image_name, "reviewed": reviewed, **(extra or {}), "rooms": out_rooms}
    Path(path).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


DEFAULT_ROOMS_DIR = Path(__file__).resolve().parent / "data" / "danish_rooms"


def rooms_json_path(image_path: Path, rooms_dir: Path = DEFAULT_ROOMS_DIR) -> Path:
    """Where the rooms file of an image lives: <rooms_dir>/<image stem>.rooms.json"""
    return Path(rooms_dir) / f"{Path(image_path).stem}.rooms.json"


# ---------------------------------------------------------------- drawing

def draw_label(img: np.ndarray, text: str, org: tuple[int, int], color=(0, 0, 0), scale: float = 0.6) -> None:
    """Text with a white box behind it, centred on `org`."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = max(1, round(scale * 2))
    (tw, tht), base = cv2.getTextSize(text, font, scale, th)
    x, y = int(org[0] - tw / 2), int(org[1] + tht / 2)
    cv2.rectangle(img, (x - 3, y - tht - 3), (x + tw + 3, y + base + 1), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)


def interior_point(polygon_px: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    """A point well inside the polygon (max of the distance transform), for placing labels."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon_px).astype(np.int32)], 255)
    if mask.max() == 0:
        c = polygon_px.mean(axis=0)
        return int(c[0]), int(c[1])
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)
