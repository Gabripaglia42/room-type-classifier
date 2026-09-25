#!/usr/bin/env python3
# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Step 2a: turn the CubiCasa5K dataset into one small image per room.

CubiCasa5K stores each plan as a folder with the raster image (F1_scaled.png)
and an SVG annotation (model.svg). Every room in the SVG is a group
<g class="Space Kitchen ..."> containing a <polygon>. Its coordinates are
pixel coordinates of F1_scaled.png.

For every room polygon this script:
  1. maps the CubiCasa room name to one of the lesson classes (roomlib.CLASSES),
  2. cuts the room out with some surrounding context (roomlib.crop_room),
  3. saves it as <out>/<split>/<class>/<plan-id>_<n>.png

The dataset's own train/val/test split files are used, so rooms of one plan
never end up in two different splits (that would leak information and make
test accuracy look better than it really is).

Output layout is exactly what torchvision's ImageFolder expects, plus an
index.csv with one row per crop (useful for error analysis).

Usage:
  python build_room_dataset.py --data data/cubicasa5k --out data/room_crops
  python build_room_dataset.py --data data/cubicasa5k --out data/room_crops_none --context none
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from roomlib import CLASSES, CROP_SIZE, CUBICASA_TO_CLASS, crop_room

HERE = Path(__file__).resolve().parent
MIN_ROOM_AREA_PX = 400      # skip tiny slivers (typical plan is ~1000 px across)


def find_dataset_root(path: Path) -> Path:
    """The folder containing train.txt; the zip nests it one or two levels deep."""
    for cand in [path, *sorted(path.glob("*")), *sorted(path.glob("*/*"))]:
        if (cand / "train.txt").is_file():
            return cand
    sys.exit(f"Could not find train.txt under {path}. Point --data at the extracted cubicasa5k folder.")


def parse_points(s: str) -> np.ndarray | None:
    pts = []
    for tok in s.replace("\n", " ").split():
        if "," not in tok:
            continue
        x, y = tok.split(",")[:2]
        try:
            pts.append((float(x), float(y)))
        except ValueError:
            return None
    return np.array(pts) if len(pts) >= 3 else None


def svg_rooms(svg_path: Path) -> list[tuple[str, np.ndarray]]:
    """[(cubicasa_room_name, polygon (N,2) in pixel coords), ...]"""
    root = ET.parse(svg_path).getroot()
    rooms = []
    for g in root.iter():
        if not g.tag.endswith("g"):
            continue
        cls = g.get("class", "")
        if not cls.startswith("Space "):
            continue
        name = cls.split(" ")[1]
        poly_el = next((c for c in g if c.tag.endswith("polygon")), None)
        if poly_el is None:
            continue
        pts = parse_points(poly_el.get("points", ""))
        if pts is not None:
            rooms.append((name, pts))
    return rooms


def process_plan(rel: str, root: Path, out: Path, split: str, context: str) -> list[dict]:
    folder = root / rel.strip("/")
    img = cv2.imread(str(folder / "F1_scaled.png"), cv2.IMREAD_COLOR)
    if img is None or not (folder / "model.svg").exists():
        return []
    try:
        rooms = svg_rooms(folder / "model.svg")
    except ET.ParseError:
        return []
    h, w = img.shape[:2]
    plan_id = rel.strip("/").replace("/", "_")
    rows = []
    for n, (name, poly) in enumerate(rooms):
        cls = CUBICASA_TO_CLASS.get(name, "other")
        poly = np.clip(poly, 0, [w - 1, h - 1])
        area = cv2.contourArea(poly.astype(np.float32))
        if area < MIN_ROOM_AREA_PX:
            continue
        crop = crop_room(img, poly, context=context)
        if crop is None:
            continue
        dst = out / split / cls / f"{plan_id}_{n:02d}.png"
        cv2.imwrite(str(dst), crop)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        rows.append({"split": split, "class": cls, "cubicasa_name": name, "file": str(dst.relative_to(out)),
                     "plan": rel.strip("/"), "area_px": int(area), "bbox_w": int(x1 - x0), "bbox_h": int(y1 - y0)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "cubicasa5k",
                    help="Extracted CubiCasa5K folder (the one containing train.txt, or a parent of it).")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "room_crops")
    ap.add_argument("--context", choices=["dim", "keep", "none"], default="dim",
                    help="How to treat pixels outside the room (see roomlib.crop_room). Default: dim.")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N plans per split (quick tests).")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = find_dataset_root(args.data)
    for split in ("train", "val", "test"):
        for c in CLASSES:
            (args.out / split / c).mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split in ("train", "val", "test"):
        plans = [l.strip() for l in (root / f"{split}.txt").read_text().splitlines() if l.strip()]
        if args.limit:
            plans = plans[:args.limit]
        fn = partial(process_plan, root=root, out=args.out, split=split, context=args.context)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, rows in enumerate(ex.map(fn, plans, chunksize=8), 1):
                all_rows += rows
                if i % 250 == 0 or i == len(plans):
                    print(f"  {split}: {i}/{len(plans)} plans", flush=True)

    with open(args.out / "index.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else ["split"])
        wr.writeheader()
        wr.writerows(all_rows)

    # Remember how the crops were made; training copies this into the model
    # checkpoint so classify_rooms.py crops new plans the same way.
    (args.out / "dataset.json").write_text(json.dumps(
        {"source": "CubiCasa5K", "context": args.context, "crop_size": CROP_SIZE, "classes": CLASSES,
         "counts": {s: sum(1 for r in all_rows if r["split"] == s) for s in ("train", "val", "test")}},
        indent=1), encoding="utf-8")

    print(f"\nRoom crops written to {args.out}  (context={args.context})")
    counts = Counter((r["split"], r["class"]) for r in all_rows)
    print(f"{'class':<10}" + "".join(f"{s:>8}" for s in ("train", "val", "test")))
    for c in CLASSES:
        print(f"{c:<10}" + "".join(f"{counts[(s, c)]:>8}" for s in ("train", "val", "test")))
    other_names = Counter(r["cubicasa_name"] for r in all_rows if r["class"] == "other")
    if other_names:
        print("\nMost common CubiCasa names grouped as 'other':",
              ", ".join(f"{n} ({k})" for n, k in other_names.most_common(8)))


if __name__ == "__main__":
    main()
