#!/usr/bin/env python3
# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Step 2b: fine-tune an ImageNet CNN to classify room crops.

Reads the folders written by build_room_dataset.py, fine-tunes a pretrained
network (ResNet-18 by default, ~11M parameters, comfortable on a 6 GB GPU),
keeps the checkpoint with the best validation macro-F1, then evaluates it
once on the test split.

Why these choices:
  - Transfer learning: ImageNet features (edges, textures, shapes) transfer
    well to line drawings and need far less data than training from scratch.
  - Macro-F1 for model selection: rooms are imbalanced (many bedrooms, few
    garages). Plain accuracy rewards ignoring rare classes; macro-F1 averages
    the F1 score of every class equally.
  - Class-weighted loss: same reason, applied during training.
  - Test split used exactly once, at the end: choosing anything (epochs,
    architecture, learning rate) by looking at test results turns the test
    set into a second validation set and the reported number becomes optimistic.

Outputs in runs/<name>/:
  best.pt                 model checkpoint (+ classes and crop settings, used by classify_rooms.py)
  history.csv / curves.png   loss and scores per epoch
  test_metrics.json       accuracy, macro-F1, per-class precision/recall/F1
  confusion_matrix.png    which classes get confused with which
  test_errors/            the most confident mistakes, for error analysis

Usage:
  python train_room_classifier.py --data data/room_crops
  python train_room_classifier.py --data data/room_crops --arch efficientnet_b0 --epochs 20 --name effb0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from roomnet import ARCHS, IMG_SIZE, MEAN, STD, build_model, eval_transform, train_transform

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- metrics (numpy only, no sklearn needed)

def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def scores(cm: np.ndarray, classes: list[str]) -> dict:
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = cm.sum(1) > 0
    return {
        "accuracy": float(tp.sum() / max(1, cm.sum())),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {c: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                          "f1": round(float(f), 4), "support": int(s)}
                      for c, p, r, f, s in zip(classes, precision, recall, f1, cm.sum(1))},
    }


# ---------------------------------------------------------------- train / eval loops

def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n, preds, trues, probs = 0.0, 0, [], [], []
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            n += len(y)
            p = logits.float().softmax(1)
            probs.append(p.detach().cpu())
            preds.append(p.argmax(1).cpu())
            trues.append(y.cpu())
    return (total_loss / max(1, n), torch.cat(trues).numpy(), torch.cat(preds).numpy(),
            torch.cat(probs).numpy())


# ---------------------------------------------------------------- plots

def save_plots(out: Path, history: list[dict], cm: np.ndarray, classes: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots (pip install matplotlib)")
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train accuracy")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val accuracy")
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], label="val macro-F1")
    ax[1].set_title("Scores"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=120); plt.close(fig)

    norm = cm / np.maximum(1, cm.sum(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title("Test confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "room_crops")
    ap.add_argument("--arch", choices=list(ARCHS), default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--scratch", action="store_true",
                    help="Random initialisation instead of ImageNet weights (experiment: how much does pretraining help?).")
    ap.add_argument("--no-class-weights", action="store_true", help="Plain (unweighted) cross-entropy.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None, help="Run folder name under runs/ (default: arch + time).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    out = HERE / "runs" / (args.name or f"{args.arch}_{time.strftime('%Y%m%d-%H%M')}")
    out.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.ImageFolder(args.data / "train", transform=train_transform())
    val_ds = datasets.ImageFolder(args.data / "val", transform=eval_transform())
    test_ds = datasets.ImageFolder(args.data / "test", transform=eval_transform())
    classes = train_ds.classes
    assert val_ds.classes == classes == test_ds.classes, "class folders differ between splits"
    ds_info = json.loads((args.data / "dataset.json").read_text()) if (args.data / "dataset.json").exists() else {}

    counts = np.bincount(train_ds.targets, minlength=len(classes))
    print(f"device={device}  arch={args.arch}  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print("train counts: " + ", ".join(f"{c}={k}" for c, k in zip(classes, counts)))

    if args.no_class_weights:
        weights = None
    else:
        # inverse square-root frequency: rare classes count more, without over-correcting
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
        weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)

    model = build_model(args.arch, len(classes), pretrained=not args.scratch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Short warm-up, then cosine decay; stepped once per epoch.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs + 1,
                                                    pct_start=0.15)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p, _ = run_epoch(model, train_dl, device, criterion, optimizer, scaler)
        va_loss, va_y, va_p, _ = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        va = scores(confusion(va_y, va_p, len(classes)), classes)
        row = {"epoch": epoch, "train_loss": round(tr_loss, 4), "val_loss": round(va_loss, 4),
               "train_acc": round(float((tr_y == tr_p).mean()), 4), "val_acc": round(va["accuracy"], 4),
               "val_macro_f1": round(va["macro_f1"], 4), "seconds": round(time.time() - t0, 1)}
        history.append(row)
        mark = ""
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "classes": classes,
                        "img_size": IMG_SIZE, "mean": MEAN, "std": STD, "epoch": epoch,
                        "val_macro_f1": best_f1, "context": ds_info.get("context", "dim"),
                        "crop_size": ds_info.get("crop_size", 256)}, out / "best.pt")
            mark = "  * saved"
        print(f"epoch {epoch:>2}/{args.epochs}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} macro-F1 {row['val_macro_f1']:.3f}  "
              f"({row['seconds']}s){mark}", flush=True)

    with open(out / "history.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        wr.writeheader(); wr.writerows(history)

    # ---- final, one-time test evaluation with the best checkpoint
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, te_y, te_p, te_prob = run_epoch(model, test_dl, device, criterion)
    cm = confusion(te_y, te_p, len(classes))
    te = scores(cm, classes)
    te.update({"best_epoch": ckpt["epoch"], "val_macro_f1": ckpt["val_macro_f1"], "arch": args.arch,
               "confusion_matrix": cm.tolist(), "classes": classes})
    (out / "test_metrics.json").write_text(json.dumps(te, indent=1))
    save_plots(out, history, cm, classes)

    # ---- error analysis: the mistakes the model was most sure about
    err_dir = out / "test_errors"
    shutil.rmtree(err_dir, ignore_errors=True); err_dir.mkdir()
    wrong = np.where(te_y != te_p)[0]
    wrong = wrong[np.argsort(-te_prob[wrong, te_p[wrong]])][:60]
    for rank, i in enumerate(wrong):
        src = Path(test_ds.samples[i][0])
        shutil.copy(src, err_dir / f"{rank:02d}_true-{classes[te_y[i]]}_pred-{classes[te_p[i]]}"
                                   f"_{te_prob[i, te_p[i]]:.2f}_{src.name}")

    print(f"\nTEST (best epoch {ckpt['epoch']}): accuracy {te['accuracy']:.3f}   macro-F1 {te['macro_f1']:.3f}")
    print(f"{'class':<10}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>7}")
    for c, m in te["per_class"].items():
        print(f"{c:<10}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['support']:>7}")
    print(f"\nEverything saved in {out}")


if __name__ == "__main__":
    main()
