# Usage guide

Step-by-step instructions for every script. Run everything from the repository root with the virtual
environment active; every script also prints its options with `--help`.

Folder conventions (all ignored by git):

| Folder | Contents |
|---|---|
| `data/cubicasa5k/` | extracted CubiCasa5K dataset |
| `data/room_crops/` | room crops built from it |
| `data/plans/` | your own floor-plan images (e.g. Danish listings) |
| `data/danish_rooms/` | room labels drawn on those plans (`<image>.rooms.json`) |
| `weights/` | CubiCasa5K pretrained weights |
| `runs/` | trained models and training reports |
| `outputs/` | predictions, scores, overlays |

## 0. Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Training works on a CPU, but roughly 20–50× slower than on a GPU.

## 1. Baseline: the published CubiCasa5K model

```bash
mkdir -p weights
gdown 1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK -O weights/model_best_val_loss_var.pkl
python predict_cubicasa.py data/plans --tta
```

Writes a coloured overlay and a JSON file per plan to `outputs/cubicasa/`. If you run out of GPU memory,
use `--max-side 800`.

## 2. Train on CubiCasa5K

Download `cubicasa5k.zip` from <https://zenodo.org/records/2613548> and extract it into `data/`, so that
`data/cubicasa5k/train.txt` exists.

```bash
python build_room_dataset.py --data data/cubicasa5k --out data/room_crops
python train_room_classifier.py --data data/room_crops
```

`build_room_dataset.py` cuts one crop per room (dataset split kept) and prints the class counts.
`train_room_classifier.py` fine-tunes an ImageNet ResNet-18, keeps the epoch with the best validation
macro-F1, and scores the test split once. Outputs in `runs/<name>/`:

- `best.pt`: the checkpoint, with its classes and crop settings
- `history.csv`, `curves.png`: loss and scores per epoch
- `test_metrics.json`, `confusion_matrix.png`
- `test_errors/`: the 60 most confident test mistakes, named `true-X_pred-Y_<confidence>_...png`

Useful options: `--arch {resnet18,resnet50,efficientnet_b0,convnext_tiny}`, `--epochs`, `--batch-size`,
`--lr`, `--scratch` (no ImageNet weights), `--no-class-weights`, `--name`.

## 3. Label and score on your own plans

```bash
python annotate_rooms.py data/plans
python classify_rooms.py data/plans --cubicasa-weights weights/model_best_val_loss_var.pkl
```

In `annotate_rooms.py`, click the corners of a room and press its class number (1–9); `n` moves to the
next image and `q` quits. `classify_rooms.py` scores the newest model in `runs/` (and, with
`--cubicasa-weights`, the published model) on the labelled rooms. Results go to `outputs/classified/`:
`summary.txt`, `results.json`, and an overlay per plan where wrong rooms are marked `X (correct class)`.

## 4. Fine-tune on your plans

```bash
python finetune_danish.py                     # before vs after fine-tuning, same rooms
python finetune_danish.py --compare-imagenet  # + a model that skips CubiCasa5K entirely
python finetune_danish.py --freeze-backbone --name head_only
python finetune_danish.py --save-final        # also save a model trained on all labelled rooms
```

Evaluation uses k-fold cross-validation grouped by house: all images of one listing (files named
`<listing>_<n>.jpg`) stay in the same fold, and every room is predicted by a model that never saw its
house. Results in `outputs/<name>/summary.txt`. The model from `--save-final` is meant for new plans;
`classify_rooms.py` warns if it is scored on the rooms it was trained on.

## 5. Faster labelling with model drafts

```bash
python finetune_danish.py --save-final
python prelabel_rooms.py data/plans                   # drafts for images without a rooms file
python annotate_rooms.py data/plans --drafts-only     # review
python finetune_danish.py                             # only accepted plans are used
python prelabel_rooms.py --report                     # how much had to be corrected
```

The published CubiCasa5K model proposes the room outlines and the fine-tuned classifier proposes the types.
Review keys:

| Key | Action |
|---|---|
| right-click in a room | select it |
| 1–9 | change the selected room's type (or close a room being drawn) |
| d | delete the selected room |
| u | undo the last room |
| a | accept the plan (it becomes training data) and go to the next |

Check every room, not only those marked `?`: accepting a confident but wrong guess copies the model's
mistake into the dataset. Drafts that are not accepted are never used for training or scoring.

## Experiments

Change one thing at a time, give each run its own `--name`, and compare validation macro-F1.

1. **Pretraining:** `--scratch` shows how much ImageNet features help.
2. **Context:** rebuild the crops with `--context none` or `--context keep` and retrain.
3. **Architecture:** `--arch efficientnet_b0`, or `--arch convnext_tiny --batch-size 32`.
4. **Class imbalance:** `--no-class-weights`, then compare recall on rare classes.
5. **Class definition:** split or drop the catch-all class `other` in `roomlib.CUBICASA_TO_CLASS`.
