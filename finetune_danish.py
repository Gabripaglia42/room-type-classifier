#!/usr/bin/env python3
# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Step 4: fine-tune the CubiCasa-trained classifier on your labelled Danish rooms.

The model trained in step 2 scores ~86% on Finnish (CubiCasa) plans but much
less on Danish plans -- a *domain shift*. The standard remedy is to continue
training on a small amount of data from the target domain.

The difficulty is evaluation: there are only ~230 Danish rooms, too few to set
a fixed test split aside. The script therefore uses **grouped k-fold
cross-validation**:

  - the houses (not the rooms!) are split into K folds; all floors of one
    listing stay in the same fold, because rooms of the same house look alike
    and would otherwise leak between training and testing;
  - for each fold, a copy of the CubiCasa model is fine-tuned on the other
    K-1 folds and then predicts the held-out fold;
  - every room thus gets exactly one prediction from a model that never saw
    its house. Those "out-of-fold" predictions are scored together.

The same held-out rooms are also scored by the un-tuned model, so the table at
the end shows directly what fine-tuning bought.

Optional comparisons (each one is a lesson on its own):
  --compare-imagenet   also fine-tune a network that starts from ImageNet only,
                       skipping CubiCasa: does the Finnish pre-training help at all?
  --freeze-backbone    only train the final layer instead of the whole network

--save-final trains one last model on *all* Danish rooms and saves it as
runs/<name>/best.pt (usable with classify_rooms.py on new, unlabelled plans).
Its expected accuracy is the cross-validation number, not a score measured on
the rooms it was trained on.

Usage:
  python finetune_danish.py
  python finetune_danish.py --compare-imagenet
  python finetune_danish.py --freeze-backbone --name head_only
  python finetune_danish.py --save-final
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from roomlib import DEFAULT_ROOMS_DIR, crop_room, load_rooms_json, rooms_file_reviewed
from roomnet import build_model, eval_transform, load_checkpoint, train_transform

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- data

def listing_id(image_stem: str) -> str:
    """'0001_Robinievej_892620_Albertslund_3' -> '0001_Robinievej_892620_Albertslund' (one house)."""
    head, _, tail = image_stem.rpartition("_")
    return head if tail.isdigit() and head else image_stem


def find_image(stored: str, stem: str, search_dirs: list[Path]) -> Path | None:
    p = Path(stored)
    if p.is_file():
        return p
    for d in search_dirs:
        for ext in IMAGE_EXTS:
            if (d / f"{stem}{ext}").is_file():
                return d / f"{stem}{ext}"
    return None


def load_danish_rooms(rooms_dir: Path, search_dirs: list[Path], classes: list[str], context: str) -> list[dict]:
    items, drafts = [], 0
    for jp in sorted(rooms_dir.glob("*.rooms.json")):
        if not rooms_file_reviewed(jp):      # unaccepted prelabel_rooms.py draft: model guesses, not labels
            drafts += 1
            continue
        stem = jp.name[: -len(".rooms.json")]
        stored = json.loads(jp.read_text(encoding="utf-8")).get("image", "")
        img_path = find_image(stored, stem, search_dirs)
        if img_path is None:
            print(f"  skip {jp.name}: image not found (pass its folder with --images)")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        for n, r in enumerate(load_rooms_json(jp, img.shape)):
            if r["room_type"] not in classes:
                continue
            crop = crop_room(img, r["polygon_px"], context)
            if crop is None:
                continue
            items.append({"image": img_path.name, "room": n, "group": listing_id(stem),
                          "label": classes.index(r["room_type"]),
                          "pil": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))})
    if drafts:
        print(f"  ignoring {drafts} unaccepted draft(s) -- review them in annotate_rooms.py and press 'a'")
    return items


def make_folds(items: list[dict], k: int, seed: int) -> list[set[str]]:
    """Assign whole houses to k folds, balancing the number of rooms per fold."""
    sizes = Counter(it["group"] for it in items)
    groups = list(sizes)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: -sizes[g])          # biggest houses first, then greedy fill
    folds, load = [set() for _ in range(k)], [0] * k
    for g in groups:
        i = int(np.argmin(load))
        folds[i].add(g)
        load[i] += sizes[g]
    return folds


class Crops(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(self.items[i]["pil"]), self.items[i]["label"]


# ---------------------------------------------------------------- model helpers

def final_layer(model: nn.Module) -> nn.Module:
    return model.fc if hasattr(model, "fc") else model.classifier


def fine_tune(model: nn.Module, train_items: list[dict], n_classes: int, device, epochs: int, lr: float,
              batch_size: int, freeze_backbone: bool, seed: int) -> nn.Module:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    counts = np.bincount([it["label"] for it in train_items], minlength=n_classes)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32, device=device),
                                    label_smoothing=0.05)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in final_layer(model).parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(Crops(train_items, train_transform()), batch_size=batch_size, shuffle=True,
                    drop_last=len(train_items) > batch_size)
    for _ in range(epochs):
        model.train()
        if freeze_backbone:
            model.eval()                 # keep BatchNorm statistics of the frozen backbone
            final_layer(model).train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict(model: nn.Module, items: list[dict], device) -> np.ndarray:
    """Softmax probabilities, averaged over the 4 rotations (same as classify_rooms.py)."""
    model.eval()
    tf = eval_transform()
    out = []
    for i in range(0, len(items), 64):
        x = torch.stack([tf(it["pil"]) for it in items[i:i + 64]]).to(device)
        out.append((sum(model(torch.rot90(x, k, dims=(2, 3))).softmax(1) for k in range(4)) / 4).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- scoring

def score(y: np.ndarray, p: np.ndarray, n: int) -> dict:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1[present].mean()), "cm": cm,
            "recall": rec, "support": cm.sum(1)}


def cm_text(cm: np.ndarray, classes: list[str]) -> str:
    lines = ["true \\ pred".ljust(12) + "".join(c[:7].rjust(8) for c in classes)]
    for i, c in enumerate(classes):
        if cm[i].sum():
            lines.append(c.ljust(12) + "".join((str(v) if v else ".").rjust(8) for v in cm[i]))
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None, help="CubiCasa-trained checkpoint (default: newest runs/*/best.pt "
                                                           "that is not a Danish fine-tune).")
    ap.add_argument("--rooms-dir", type=Path, default=DEFAULT_ROOMS_DIR)
    ap.add_argument("--images", type=Path, nargs="*", default=[HERE / "data" / "plans"],
                    help="Folders to look for the plan images in, if the stored path has moved.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4, help="Lower than in step 2: adjust the model, don't overwrite it.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--freeze-backbone", action="store_true", help="Train only the final layer.")
    ap.add_argument("--compare-imagenet", action="store_true",
                    help="Also fine-tune from ImageNet weights only (no CubiCasa pre-training).")
    ap.add_argument("--save-final", action="store_true", help="Afterwards train on all rooms and save the model.")
    ap.add_argument("--name", default="danish_finetune")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    if args.model is None:
        cands = [p for p in sorted((HERE / "runs").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
                 if not torch.load(p, map_location="cpu", weights_only=False).get("danish_finetune")]
        if not cands:
            sys.exit("No CubiCasa-trained model in runs/. Run train_room_classifier.py first, or pass --model.")
        args.model = cands[-1]
    base, ckpt = load_checkpoint(args.model, "cpu")
    classes, context = ckpt["classes"], ckpt.get("context", "dim")
    print(f"base model: {args.model}  (arch {ckpt['arch']}, context={context})  device={device}")

    items = load_danish_rooms(args.rooms_dir, args.images, classes, context)
    if len(items) < 20:
        sys.exit(f"Only {len(items)} labelled rooms found in {args.rooms_dir} -- label more with annotate_rooms.py.")
    groups = sorted({it["group"] for it in items})
    k = min(args.folds, len(groups))
    folds = make_folds(items, k, args.seed)
    print(f"{len(items)} rooms from {len(groups)} houses, {k} folds of "
          f"{[sum(it['group'] in f for it in items) for f in folds]} rooms")
    print("rooms per class: " + ", ".join(f"{c}={n}" for c, n in
                                          zip(classes, np.bincount([it['label'] for it in items], minlength=len(classes)))))

    variants = {"no fine-tuning (step 2 model)": None, "fine-tuned from CubiCasa model": "cubicasa"}
    if args.compare_imagenet:
        variants["fine-tuned from ImageNet only"] = "imagenet"
    y = np.array([it["label"] for it in items])
    oof = {v: np.zeros((len(items), len(classes))) for v in variants}

    for fi, held in enumerate(folds):
        tr_idx = [i for i, it in enumerate(items) if it["group"] not in held]
        te_idx = [i for i, it in enumerate(items) if it["group"] in held]
        tr, te = [items[i] for i in tr_idx], [items[i] for i in te_idx]
        line = f"fold {fi + 1}/{k}: train {len(tr)} rooms, test {len(te)} rooms |"
        for name, init in variants.items():
            if init is None:
                model = copy.deepcopy(base).to(device)
            else:
                model = (copy.deepcopy(base) if init == "cubicasa"
                         else build_model(ckpt["arch"], len(classes), pretrained=True)).to(device)
                model = fine_tune(model, tr, len(classes), device, args.epochs, args.lr, args.batch_size,
                                  args.freeze_backbone, args.seed + fi)
            oof[name][te_idx] = predict(model, te, device)
            acc = float((oof[name][te_idx].argmax(1) == y[te_idx]).mean())
            line += f"  {init or 'base'} {acc:.2f}"
            del model
        print(line, flush=True)

    out = HERE / "outputs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    rep = [f"base model: {args.model}", f"{len(items)} Danish rooms, {len(groups)} houses, {k}-fold grouped CV, "
           f"{args.epochs} epochs, lr {args.lr}, {'final layer only' if args.freeze_backbone else 'whole network'}", "",
           f"{'':<34}{'accuracy':>9}{'macro-F1':>10}   recall per class"]
    for name in variants:
        s = score(y, oof[name].argmax(1), len(classes))
        rec = "  ".join(f"{c[:4]} {r:.2f}" for c, r, n in zip(classes, s["recall"], s["support"]) if n)
        rep.append(f"{name:<34}{s['accuracy']:>9.3f}{s['macro_f1']:>10.3f}   {rec}")
    best = "fine-tuned from CubiCasa model"
    rep += ["", f"Confusion ({best}, out-of-fold):", cm_text(score(y, oof[best].argmax(1), len(classes))["cm"], classes)]
    report = "\n".join(rep)
    print("\n" + report)
    (out / "summary.txt").write_text(report + "\n", encoding="utf-8")
    (out / "oof_predictions.json").write_text(json.dumps([
        {"image": it["image"], "room": it["room"], "house": it["group"], "truth": classes[it["label"]],
         **{name: {"prediction": classes[int(oof[name][i].argmax())], "confidence": round(float(oof[name][i].max()), 3)}
            for name in variants}} for i, it in enumerate(items)], indent=1, ensure_ascii=False), encoding="utf-8")

    if args.save_final:
        model = fine_tune(copy.deepcopy(base).to(device), items, len(classes), device, args.epochs, args.lr,
                          args.batch_size, args.freeze_backbone, args.seed)
        run = HERE / "runs" / args.name
        run.mkdir(parents=True, exist_ok=True)
        cv = score(y, oof[best].argmax(1), len(classes))
        torch.save({**{k_: v for k_, v in ckpt.items() if k_ != "state_dict"},
                    "state_dict": model.state_dict(), "danish_finetune": True, "base_model": str(args.model),
                    "val_macro_f1": cv["macro_f1"], "cv_accuracy": cv["accuracy"], "n_danish_rooms": len(items)},
                   run / "best.pt")
        print(f"\nFinal model (all {len(items)} rooms) saved to {run / 'best.pt'}. Expected accuracy on new "
              f"plans ~ the cross-validation figure above; do not re-score it on these same rooms.")
    print(f"\nSummary saved to {out / 'summary.txt'}")


if __name__ == "__main__":
    main()
