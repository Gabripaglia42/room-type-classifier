#!/usr/bin/env python3
# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Step 5: let the models draft the labels, then review them with annotate_rooms.py.

For every image that has no rooms file yet:
  1. the pretrained CubiCasa5K model proposes room *outlines*
     (predict_cubicasa.propose_rooms -- roughly 2 in 3 rooms come out usable);
  2. your classifier (by default the newest Danish fine-tuned model from
     finetune_danish.py --save-final) proposes each room's *type*;
  3. the result is saved as a DRAFT: data/danish_rooms/<image>.rooms.json
     with "reviewed": false.

Drafts are ignored by finetune_danish.py and are never used as ground truth by
classify_rooms.py until you open them in annotate_rooms.py, fix them and press
'a' (accept). Otherwise the model would be trained on its own unchecked
guesses and its mistakes would be copied into the dataset.

Because every draft records the model's original guess, --report shows how much
you had to correct across all accepted drafts: an honest measure of how
well the models work on new plans.

Usage:
  python prelabel_rooms.py data/plans
  python prelabel_rooms.py data/plans --model runs/danish_finetune/best.pt
  python prelabel_rooms.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch

from classify_rooms import classify_crops
from predict_cubicasa import DEFAULT_WEIGHTS, load_cubicasa, propose_rooms
from roomlib import DEFAULT_ROOMS_DIR, crop_room, rooms_json_path, save_rooms_json
from roomnet import load_checkpoint

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_classifier() -> Path | None:
    """Newest Danish fine-tuned checkpoint; otherwise the newest checkpoint of any kind."""
    cands = sorted((HERE / "runs").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    danish = [p for p in cands if torch.load(p, map_location="cpu", weights_only=False).get("danish_finetune")]
    return (danish or cands or [None])[-1]


def report(rooms_dir: Path) -> None:
    files = proposed = kept = retyped = deleted = added = 0
    confusions = Counter()
    for jp in sorted(rooms_dir.glob("*.rooms.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "auto_proposed" not in d or not d.get("reviewed"):
            continue
        files += 1
        proposed += d["auto_proposed"]
        auto = [r for r in d["rooms"] if r.get("source") == "auto"]
        added += len(d["rooms"]) - len(auto)
        deleted += d["auto_proposed"] - len(auto)
        for r in auto:
            if r["room_type"] == r.get("auto_type"):
                kept += 1
            else:
                retyped += 1
                confusions[(r.get("auto_type"), r["room_type"])] += 1
    if not files:
        print("No accepted drafts yet (open them with annotate_rooms.py and press 'a').")
        return
    print(f"accepted drafts: {files}   proposed rooms: {proposed}")
    print(f"  kept as proposed        {kept:>5}  ({kept / max(1, proposed):.0%})")
    print(f"  type corrected          {retyped:>5}  ({retyped / max(1, proposed):.0%})")
    print(f"  deleted (bad outline)   {deleted:>5}  ({deleted / max(1, proposed):.0%})")
    print(f"  drawn by hand (missed)  {added:>5}")
    if kept + retyped:
        print(f"classifier accuracy on outlines you kept: {kept / (kept + retyped):.1%}")
    if confusions:
        print("most common corrections (model -> you): " +
              ", ".join(f"{a}->{b} ({n})" for (a, b), n in confusions.most_common(8)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="Images and/or folders to pre-label.")
    ap.add_argument("--model", type=Path, default=None, help="Room classifier (default: newest Danish fine-tune).")
    ap.add_argument("--cubicasa-weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--rooms-dir", type=Path, default=DEFAULT_ROOMS_DIR)
    ap.add_argument("--max-side", type=int, default=1400, help="CubiCasa input size; lower it if you run out of VRAM.")
    ap.add_argument("--report", action="store_true", help="Only print correction statistics of accepted drafts.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.report:
        report(args.rooms_dir)
        return
    if not args.inputs:
        sys.exit("Give image files or folders (or --report).")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    ckpt_path = args.model or pick_classifier()
    if ckpt_path is None:
        sys.exit("No classifier in runs/. Run finetune_danish.py --save-final first.")
    clf, ckpt = load_checkpoint(ckpt_path, device)
    if not ckpt.get("danish_finetune"):
        print("NOTE: using a CubiCasa-only classifier; run 'finetune_danish.py --save-final' for much better drafts.")
    classes, context = ckpt["classes"], ckpt.get("context", "dim")
    cubi = load_cubicasa(args.cubicasa_weights, device)
    print(f"classifier: {ckpt_path}\noutlines:   {args.cubicasa_weights}\n")

    files = []
    for s in args.inputs:
        p = Path(s)
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    args.rooms_dir.mkdir(parents=True, exist_ok=True)
    made = skipped = 0
    for f in files:
        jp = rooms_json_path(f, args.rooms_dir)
        if jp.exists():
            skipped += 1
            continue
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        polys = [p for p in propose_rooms(cubi, img, device, args.max_side) if crop_room(img, p, context) is not None]
        rooms = []
        if polys:
            probs = classify_crops(clf, [crop_room(img, p, context) for p in polys], device)
            for p, pr in zip(polys, probs):
                cls = classes[int(pr.argmax())]
                rooms.append({"polygon_px": p, "room_type": cls, "source": "auto", "auto_type": cls,
                              "confidence": round(float(pr.max()), 3)})
        save_rooms_json(jp, str(f.resolve()), img.shape, rooms, reviewed=False,
                        extra={"auto_proposed": len(rooms), "auto_classifier": str(ckpt_path)})
        made += 1
        print(f"  {f.name}: {len(rooms)} rooms drafted", flush=True)
    print(f"\n{made} drafts written, {skipped} images skipped (already had a rooms file).")
    print("Next: python annotate_rooms.py <same folder>  -> fix each plan, press 'a' to accept it.")


if __name__ == "__main__":
    main()
