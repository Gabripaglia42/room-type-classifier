#!/usr/bin/env python3
# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Step 1 ("find"): run the public pretrained CubiCasa5K model on floor-plan images.

The CubiCasa5K model (Kalervo et al., 2019) was trained on 5,000 Finnish
floor plans. Among its outputs is a per-pixel room segmentation with 12
classes (background, outdoor, wall, kitchen, living room, bedroom, bath,
entry, railing, storage, garage, undefined). This script runs it on any
images -- e.g. Danish plans in data/plans/ -- and writes:

  <out>/<image>_rooms.png   the plan with every predicted room coloured and labelled
  <out>/<image>_rooms.json  one entry per predicted room region (class, area, position)

It answers the question "does an existing model already work on Danish plans?"
before any training is done.

Weights: download model_best_val_loss_var.pkl from the link in the CubiCasa5K
README (https://github.com/CubiCasa/CubiCasa5k) and place it in weights/.
License: CC BY-NC 4.0 -- personal/educational use only.

Usage:
  python predict_cubicasa.py data/plans
  python predict_cubicasa.py plan.jpg --max-side 1400 --tta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cubicasa_model import hg_furukawa_original
from roomlib import (CLASS_COLORS, CUBICASA_MODEL_TO_CLASS, draw_label, interior_point)

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights" / "model_best_val_loss_var.pkl"
N_CHANNELS = 44            # 21 junction heatmaps + 12 room classes + 11 icon classes
ROOM_SLICE = slice(21, 33)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_cubicasa(weights: Path, device: torch.device) -> torch.nn.Module:
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}\nDownload model_best_val_loss_var.pkl (link in the "
                 f"CubiCasa5K README) and put it in {DEFAULT_WEIGHTS.parent}/")
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        # The 2019 checkpoint also pickles non-tensor objects; it comes from the
        # dataset authors, so loading it with full unpickling is acceptable.
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = hg_furukawa_original(N_CHANNELS)
    model.load_state_dict(state)
    return model.eval().to(device)


@torch.no_grad()
def predict_room_map(model: torch.nn.Module, image_bgr: np.ndarray, device: torch.device,
                     max_side: int = 1024, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_map, confidence) at the original image size.

    class_map: HxW int array of CubiCasa room indices (0..11).
    confidence: HxW float array, softmax probability of the chosen class.
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1 else image_bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.from_numpy(2 * (rgb / 255.0) - 1).permute(2, 0, 1)[None].to(device)

    turns = [0, 1, 2, 3] if tta else [0]
    logits = 0
    for k in turns:
        out = model(torch.rot90(x, k, dims=(2, 3)))[:, ROOM_SLICE]
        out = torch.rot90(out, -k, dims=(2, 3))
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        logits = logits + F.softmax(out, dim=1)
    probs = logits / len(turns)
    probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)[0]
    conf, cls = probs.max(dim=0)
    return cls.cpu().numpy().astype(np.uint8), conf.cpu().numpy()


def room_regions(class_map: np.ndarray, conf: np.ndarray, min_area_frac: float = 0.003) -> list[dict]:
    """Split the per-pixel map into connected room regions of the lesson classes."""
    h, w = class_map.shape
    regions = []
    for idx, name in CUBICASA_MODEL_TO_CLASS.items():
        mask = (class_map == idx).astype(np.uint8)
        if not mask.any():
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_frac * h * w:
                continue
            comp = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            poly = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(poly) < 3:
                continue
            regions.append({
                "room_type": name,
                "area_fraction": round(area / (h * w), 4),
                "mean_confidence": round(float(conf[labels == i].mean()), 3),
                "label_point": list(interior_point(poly.astype(np.float64), (h, w))),
                "polygon": [[round(float(px) / w, 4), round(float(py) / h, 4)] for px, py in poly],
            })
    return regions


def propose_rooms(model: torch.nn.Module, image_bgr: np.ndarray, device: torch.device,
                  max_side: int = 1400, tta: bool = True, min_area_frac: float = 0.003) -> list[np.ndarray]:
    """Room outlines for pre-labelling: every connected area of 'some room' pixels, separated by walls.

    The room *type* predicted by CubiCasa is ignored here (it is poor on Danish
    plans); only the shape is used. Tested on 58 hand-drawn Danish rooms with
    these settings: about two thirds got an outline matching at IoU >= 0.8;
    open-plan spaces and some small rooms (WC, entré) are merged or missed.
    Returns polygons in pixel coordinates.
    """
    class_map, _ = predict_room_map(model, image_bgr, device, max_side, tta)
    room = np.isin(class_map, list(CUBICASA_MODEL_TO_CLASS)).astype(np.uint8)
    room = cv2.morphologyEx(room, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # cut thin bridges under doors
    n, labels, stats, _ = cv2.connectedComponentsWithStats(room, connectivity=4)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_frac * class_map.size:
            continue
        contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float64)
        if len(poly) >= 3:
            polys.append(poly)
    return polys


def render(image_bgr: np.ndarray, class_map: np.ndarray, regions: list[dict]) -> np.ndarray:
    colour = np.zeros_like(image_bgr)
    painted = np.zeros(class_map.shape, dtype=bool)
    for idx, name in CUBICASA_MODEL_TO_CLASS.items():
        m = class_map == idx
        colour[m] = CLASS_COLORS[name]
        painted |= m
    out = image_bgr.copy()
    out[painted] = (0.55 * out[painted] + 0.45 * colour[painted]).astype(np.uint8)
    scale = max(0.5, image_bgr.shape[1] / 1600)
    for r in regions:
        draw_label(out, f"{r['room_type']} {r['mean_confidence']:.2f}", tuple(r["label_point"]),
                   CLASS_COLORS[r["room_type"]], scale)
    return out


def collect_images(inputs: list[str]) -> list[Path]:
    files = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"skip (not found): {p}")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Image files and/or folders of images.")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "cubicasa")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="Longest image side fed to the network (default 1024). Larger = finer but more VRAM.")
    ap.add_argument("--tta", action="store_true", help="Average predictions over 4 rotations (slower, usually better).")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()

    files = collect_images(args.inputs)
    if not files:
        sys.exit("No images found.")
    device = pick_device(args.device)
    model = load_cubicasa(args.weights, device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  images={len(files)}  out={args.out}")

    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  unreadable: {f.name}")
            continue
        class_map, conf = predict_room_map(model, img, device, args.max_side, args.tta)
        regions = room_regions(class_map, conf)
        cv2.imwrite(str(args.out / f"{f.stem}_rooms.png"), render(img, class_map, regions))
        (args.out / f"{f.stem}_rooms.json").write_text(
            json.dumps({"image": str(f), "rooms": regions}, indent=1), encoding="utf-8")
        summary = ", ".join(sorted(r["room_type"] for r in regions)) or "(no rooms found)"
        print(f"  {f.name}: {summary}")


if __name__ == "__main__":
    main()
