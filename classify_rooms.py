#!/usr/bin/env python3
# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Step 3b: classify the rooms of new plans with the trained model -- and score it.

The room polygons of each image come from data/danish_rooms/<image>.rooms.json
(drawn with annotate_rooms.py, or drafted by prelabel_rooms.py and accepted).
Images without a rooms file are skipped: this model classifies rooms, it does
not find them.

Each room is cropped exactly as during training (roomlib.crop_room with the
context mode stored in the checkpoint) and classified. If the rooms file
contains the correct room_type, the prediction is scored against it, which
gives the real-world (Danish) accuracy to compare with the CubiCasa test score.

With --cubicasa-weights the pretrained CubiCasa5K model from step 1 is scored
on the same rooms (majority vote of its pixel predictions inside each polygon),
so both approaches are compared on identical data.

Outputs in outputs/classified/:
  <image>_classified.png   rooms outlined and labelled (X = wrong, with the correct class)
  results.json             every room: prediction, confidence, ground truth
  summary.txt              accuracy / macro-F1 per model when ground truth exists

Usage:
  python classify_rooms.py data/plans                       # latest model in runs/
  python classify_rooms.py data/plans --model runs/resnet18_x/best.pt
  python classify_rooms.py data/plans --cubicasa-weights weights/model_best_val_loss_var.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from roomlib import (CLASS_COLORS, CLASSES, CUBICASA_MODEL_TO_CLASS, DEFAULT_ROOMS_DIR, crop_room,
                     draw_label, interior_point, load_rooms_json, rooms_json_path)
from roomnet import eval_transform, load_checkpoint

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def latest_checkpoint() -> Path | None:
    cands = sorted((HERE / "runs").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


@torch.no_grad()
def classify_crops(model, crops_bgr: list[np.ndarray], device, tta: bool = True) -> np.ndarray:
    """Softmax probabilities, shape (n_rooms, n_classes). TTA = average over 4 rotations."""
    tf = eval_transform()
    x = torch.stack([tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr]).to(device)
    turns = range(4) if tta else [0]
    probs = sum(model(torch.rot90(x, k, dims=(2, 3))).softmax(1) for k in turns) / len(turns)
    return probs.cpu().numpy()


def cubicasa_votes(class_map: np.ndarray, polygon_px: np.ndarray) -> tuple[str | None, float]:
    """Majority CubiCasa room class inside a polygon, ignoring background/wall/railing pixels."""
    mask = np.zeros(class_map.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon_px).astype(np.int32)], 1)
    vals = class_map[mask > 0]
    vals = vals[np.isin(vals, list(CUBICASA_MODEL_TO_CLASS))]
    if vals.size == 0:
        return None, 0.0
    idx, n = Counter(vals.tolist()).most_common(1)[0]
    return CUBICASA_MODEL_TO_CLASS[idx], n / vals.size


def score(pairs: list[tuple[str, str | None]]) -> dict:
    """pairs = [(truth, prediction)]"""
    if not pairs:
        return {}
    acc = sum(t == p for t, p in pairs) / len(pairs)
    f1s = []
    for c in sorted({t for t, _ in pairs}):
        tp = sum(t == c and p == c for t, p in pairs)
        fp = sum(t != c and p == c for t, p in pairs)
        fn = sum(t == c and p != c for t, p in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"rooms": len(pairs), "accuracy": acc, "macro_f1": float(np.mean(f1s))}


def confusion_text(pairs: list[tuple[str, str | None]]) -> str:
    labels = sorted({t for t, _ in pairs} | {p or "none" for _, p in pairs})
    w = max(8, max(len(l) for l in labels) + 1)
    lines = ["true \\ pred".ljust(w) + "".join(l[:7].rjust(8) for l in labels)]
    for t in labels:
        row = [sum(1 for a, b in pairs if a == t and (b or "none") == p) for p in labels]
        if sum(row):
            lines.append(t.ljust(w) + "".join(str(v).rjust(8) if v else ".".rjust(8) for v in row))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Images and/or folders.")
    ap.add_argument("--model", type=Path, default=None, help="Checkpoint (default: newest runs/*/best.pt).")
    ap.add_argument("--rooms-dir", type=Path, default=DEFAULT_ROOMS_DIR)
    ap.add_argument("--cubicasa-weights", type=Path, default=None,
                    help="Also score the pretrained CubiCasa5K model (step 1) on the same rooms.")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "classified")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    ckpt_path = args.model or latest_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        sys.exit("No trained model found. Run train_room_classifier.py first, or pass --model.")
    model, ckpt = load_checkpoint(ckpt_path, device)
    classes, context = ckpt["classes"], ckpt.get("context", "dim")
    print(f"model: {ckpt_path}  (arch {ckpt['arch']}, context={context}, val macro-F1 {ckpt.get('val_macro_f1', 0):.3f})")

    cubi = None
    if args.cubicasa_weights:
        from predict_cubicasa import load_cubicasa, predict_room_map
        cubi = load_cubicasa(args.cubicasa_weights, device)

    files = []
    for s in args.inputs:
        p = Path(s)
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    args.out.mkdir(parents=True, exist_ok=True)

    results, ours, theirs = [], [], []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        jp = rooms_json_path(f, args.rooms_dir)
        if jp.exists():
            rooms, source = load_rooms_json(jp, img.shape, require_reviewed=True), "annotated"
        else:
            continue
        rooms = [r for r in rooms if crop_room(img, r["polygon_px"], context) is not None]
        if not rooms:
            print(f"  {f.name}: no usable rooms ({source})")
            continue

        probs = classify_crops(model, [crop_room(img, r["polygon_px"], context) for r in rooms], device,
                               tta=not args.no_tta)
        class_map = predict_room_map(cubi, img, device)[0] if cubi is not None else None

        vis = img.copy()
        lw, fs = max(2, w // 700), max(0.5, w / 2200)
        n_ok = n_gt = 0
        for r, p in zip(rooms, probs):
            pred, conf = classes[int(p.argmax())], float(p.max())
            truth = r["room_type"]
            entry = {"image": f.name, "source": source, "prediction": pred, "confidence": round(conf, 3),
                     "top3": [[classes[i], round(float(p[i]), 3)] for i in np.argsort(-p)[:3]], "truth": truth}
            if class_map is not None:
                entry["cubicasa_prediction"], entry["cubicasa_share"] = cubicasa_votes(class_map, r["polygon_px"])
            results.append(entry)
            if truth:
                n_gt += 1; n_ok += pred == truth
                ours.append((truth, pred))
                if class_map is not None:
                    theirs.append((truth, entry["cubicasa_prediction"]))
            pts = np.round(r["polygon_px"]).astype(np.int32)
            cv2.polylines(vis, [pts], True, CLASS_COLORS[pred], lw)
            text = f"{pred} {conf:.2f}" + ("" if not truth or truth == pred else f"  X ({truth})")
            draw_label(vis, text, interior_point(r["polygon_px"], img.shape),
                       (0, 0, 255) if truth and truth != pred else CLASS_COLORS[pred], fs)
        cv2.imwrite(str(args.out / f"{f.stem}_classified.png"), vis)
        tail = f"  {n_ok}/{n_gt} correct" if n_gt else ""
        print(f"  {f.name}: {len(rooms)} rooms ({source}){tail}")

    (args.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    report = [f"model: {ckpt_path}", f"rooms classified: {len(results)}"]
    if ours:
        s = score(ours)
        report += ["", f"OUR CLASSIFIER on annotated rooms: accuracy {s['accuracy']:.3f}  "
                       f"macro-F1 {s['macro_f1']:.3f}  (n={s['rooms']})", confusion_text(ours)]
    if theirs:
        s = score(theirs)
        report += ["", f"PRETRAINED CUBICASA5K on the same rooms: accuracy {s['accuracy']:.3f}  "
                       f"macro-F1 {s['macro_f1']:.3f}  (n={s['rooms']})", confusion_text(theirs)]
    if not ours:
        report += ["", "No ground truth found -- draw and label rooms with annotate_rooms.py to get a score."]
    elif ckpt.get("danish_finetune"):
        report += ["", "WARNING: this model was fine-tuned on the annotated Danish rooms, so the score above is "
                       "measured on its own training data and is meaningless. Use the cross-validation figure "
                       "from finetune_danish.py, or pass --model with the CubiCasa-only checkpoint."]
    (args.out / "summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"\nImages and results in {args.out}")


if __name__ == "__main__":
    main()
