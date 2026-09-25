#!/usr/bin/env python3
# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Step 3a / 5b: draw and label rooms on your own (Danish) plans, or review model drafts.

CubiCasa5K test accuracy tells you how well the model does on *Finnish*
plans. To know how it does on Danish plans you need Danish rooms with the
correct answer attached. This tool lets you make that set by hand, or check
the drafts written by prelabel_rooms.py.

For each image it writes data/danish_rooms/<image>.rooms.json (polygons normalised
to 0..1), which classify_rooms.py and
finetune_danish.py read.

Controls (window must have focus):
  left click        add a corner to the current room
  right click       while drawing: remove the last corner
                    otherwise: select the room under the mouse (white outline)
  1-9               while drawing: close the room and give it that class
                    with a room selected: change its class
  d                 delete the selected room
  backspace         remove the last corner
  u                 undo the last finished room
  a                 ACCEPT this plan (drafts only): marks it reviewed, saves, goes to the next
  s                 save
  n / space         next image (saves changes)
  p                 previous image (saves changes)
  q / esc           quit (saves changes)

Reviewing a draft: every proposed room shows the model's guess and its
confidence ('?' marks confidence below 0.6). Check EVERY room, not only the
flagged ones -- confident mistakes are the ones that slip through. Delete
outlines that are wrong (d), fix wrong types (right-click + number), draw
missing rooms, then press 'a'. A draft that is not accepted is never used for
training or scoring.

Labelling guidance: label a room by its *function* as written on the plan
("Køkken" -> kitchen). For an open kitchen/living space, draw the kitchen
part (where the kitchen units are) as kitchen and the rest as living, or
skip it -- note which you chose, it affects the results.

Usage:
  python annotate_rooms.py data/plans
  python annotate_rooms.py data/plans/0003_*.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from roomlib import (CLASS_COLORS, CLASSES, DANISH_HINT, DEFAULT_ROOMS_DIR, draw_label, interior_point, load_rooms_json,
                     rooms_file_reviewed, rooms_json_path, save_rooms_json)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "annotate rooms"
LOW_CONFIDENCE = 0.6


class Annotator:
    def __init__(self, image_path: Path, max_w: int, max_h: int, rooms_dir: Path):
        self.path = image_path
        self.json_path = rooms_json_path(image_path, rooms_dir)
        self.img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError(f"unreadable image {image_path}")
        h, w = self.img.shape[:2]
        self.scale = min(1.0, max_w / w, max_h / h)
        self.rooms: list[dict] = []
        self.reviewed = True          # hand-drawn files are reviewed by definition
        self.extra: dict = {}         # keeps prelabel metadata (auto_proposed, ...) across saves
        if self.json_path.exists():
            self.rooms = load_rooms_json(self.json_path, self.img.shape)
            self.reviewed = rooms_file_reviewed(self.json_path)
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.extra = {k: v for k, v in data.items() if k not in ("image", "reviewed", "rooms")}
        self.current: list[tuple[float, float]] = []
        self.selected: int | None = None
        self.dirty = False

    # coordinates: stored in full-resolution pixels, displayed scaled
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected = None
            self.current.append((x / self.scale, y / self.scale))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current:
                self.current.pop()
            else:
                self.selected = self.room_at(x / self.scale, y / self.scale)

    def room_at(self, x: float, y: float) -> int | None:
        """Index of the smallest room containing the point (so a small room inside a big one can be picked)."""
        hits = [(cv2.contourArea(r["polygon_px"].astype(np.float32)), i) for i, r in enumerate(self.rooms)
                if cv2.pointPolygonTest(r["polygon_px"].astype(np.float32), (x, y), False) >= 0]
        return min(hits)[1] if hits else None

    def number_key(self, cls: str) -> None:
        if len(self.current) >= 3:
            self.rooms.append({"polygon_px": np.array(self.current), "room_type": cls, "source": "manual"})
            self.current = []
            self.dirty = True
        elif not self.current and self.selected is not None:
            self.rooms[self.selected]["room_type"] = cls
            self.dirty = True

    def delete_selected(self) -> None:
        if self.selected is not None:
            self.rooms.pop(self.selected)
            self.selected = None
            self.dirty = True

    def save(self):
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        save_rooms_json(self.json_path, str(self.path.resolve()), self.img.shape, self.rooms,
                        reviewed=self.reviewed, extra=self.extra)
        self.dirty = False

    def render(self, idx: int, total: int) -> np.ndarray:
        h, w = self.img.shape[:2]
        disp = cv2.resize(self.img, (round(w * self.scale), round(h * self.scale)), interpolation=cv2.INTER_AREA)
        overlay = disp.copy()
        for r in self.rooms:
            pts = np.round(r["polygon_px"] * self.scale).astype(np.int32)
            cv2.fillPoly(overlay, [pts], CLASS_COLORS.get(r["room_type"], (0, 0, 0)))
        disp = cv2.addWeighted(overlay, 0.3, disp, 0.7, 0)
        for i, r in enumerate(self.rooms):
            pts = np.round(r["polygon_px"] * self.scale).astype(np.int32)
            col = CLASS_COLORS.get(r["room_type"], (0, 0, 0))
            cv2.polylines(disp, [pts], True, col, 2)
            if i == self.selected:
                cv2.polylines(disp, [pts], True, (255, 255, 255), 5)
                cv2.polylines(disp, [pts], True, (0, 0, 0), 2)
            text = r["room_type"] or "?"
            if not self.reviewed and r.get("source") == "auto" and r["room_type"] == r.get("auto_type"):
                conf = r.get("confidence") or 0
                text += f" {conf:.2f}" + (" ?" if conf < LOW_CONFIDENCE else "")
                if conf < LOW_CONFIDENCE:
                    col = (0, 0, 255)
            draw_label(disp, text, interior_point(pts.astype(np.float64), disp.shape), col, 0.5)
        if self.current:
            pts = np.round(np.array(self.current) * self.scale).astype(np.int32)
            cv2.polylines(disp, [pts], False, (0, 0, 255), 2)
            for p in pts:
                cv2.circle(disp, tuple(int(v) for v in p), 4, (0, 0, 255), -1)
        # legend
        status = "" if self.reviewed else "  DRAFT - check every room, then press 'a' to accept"
        lines = [f"[{idx + 1}/{total}] {self.path.name}  rooms: {len(self.rooms)}"
                 f"{'  (unsaved)' if self.dirty else ''}{status}"]
        lines += [f"{i + 1} {c:<8} {DANISH_HINT[c]}" for i, c in enumerate(CLASSES)]
        lines += ["click=corner  right-click=select room/undo corner  d=delete  u=undo room",
                  "a=accept draft  n=next  p=prev  q=quit"]
        y = 18
        for i, t in enumerate(lines):
            col = CLASS_COLORS[CLASSES[i - 1]] if 1 <= i <= len(CLASSES) else ((0, 0, 200) if i == 0 and status else (0, 0, 0))
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(disp, (4, y - th - 3), (10 + tw, y + 4), (255, 255, 255), -1)
            cv2.putText(disp, t, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            y += th + 8
        return disp


def collect(inputs: list[str]) -> list[Path]:
    files = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Images and/or folders.")
    ap.add_argument("--max-width", type=int, default=1500, help="Display size limit (image is scaled down to fit).")
    ap.add_argument("--max-height", type=int, default=950)
    ap.add_argument("--rooms-dir", type=Path, default=DEFAULT_ROOMS_DIR, help="Where the .rooms.json files go.")
    ap.add_argument("--drafts-only", action="store_true", help="Only open images whose rooms file is an unaccepted draft.")
    args = ap.parse_args()

    files = collect(args.inputs)
    if args.drafts_only:
        files = [f for f in files if rooms_json_path(f, args.rooms_dir).exists()
                 and not rooms_file_reviewed(rooms_json_path(f, args.rooms_dir))]
    if not files:
        sys.exit("No images found.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    i = 0
    ann = Annotator(files[i], args.max_width, args.max_height, args.rooms_dir)
    cv2.setMouseCallback(WINDOW, ann.on_mouse)

    def go(step: int):
        nonlocal i, ann
        i = (i + step) % len(files)
        ann = Annotator(files[i], args.max_width, args.max_height, args.rooms_dir)
        cv2.setMouseCallback(WINDOW, ann.on_mouse)

    while True:
        cv2.imshow(WINDOW, ann.render(i, len(files)))
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # window closed with the mouse
                if ann.dirty:
                    ann.save()
                break
            continue
        if ord("1") <= key <= ord(str(len(CLASSES))):
            ann.number_key(CLASSES[key - ord("1")])
        elif key in (8, 127):
            if ann.current:
                ann.current.pop()
        elif key == ord("d"):
            ann.delete_selected()
        elif key == ord("u"):
            if ann.rooms:
                ann.rooms.pop(); ann.selected = None; ann.dirty = True
        elif key == ord("a"):
            if not ann.reviewed:
                ann.reviewed = True
                ann.save()
                go(1)
        elif key == ord("s"):
            ann.save()
        elif key in (ord("n"), ord(" "), ord("p")):
            if ann.dirty:
                ann.save()
            go(1 if key != ord("p") else -1)
        elif key in (ord("q"), 27):
            if ann.dirty:
                ann.save()
            break
    cv2.destroyAllWindows()
    have = [rooms_json_path(f, args.rooms_dir) for f in files if rooms_json_path(f, args.rooms_dir).exists()]
    drafts = sum(1 for p in have if not rooms_file_reviewed(p))
    print(f"{len(have)}/{len(files)} images have a rooms file ({drafts} still unaccepted drafts).")


if __name__ == "__main__":
    main()
