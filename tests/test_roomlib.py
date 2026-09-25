# Copyright 2026 Gabriele Paglia
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for roomlib: the crop function and the rooms-file format.

They need only numpy and OpenCV (no GPU, no dataset), so they run in CI.
"""

import json

import numpy as np
import pytest

from roomlib import (CLASSES, CROP_SIZE, CUBICASA_MODEL_TO_CLASS, CUBICASA_TO_CLASS, crop_room, load_rooms_json,
                     rooms_file_reviewed, rooms_json_path, save_rooms_json)


@pytest.fixture
def plan():
    """A 400 x 600 grey 'plan' with one dark 100 x 200 room at x 100..300, y 100..200."""
    img = np.full((400, 600, 3), 128, np.uint8)
    img[100:200, 100:300] = 20
    poly = np.array([[100, 100], [300, 100], [300, 200], [100, 200]], float)
    return img, poly


def test_classes_are_sorted_like_imagefolder():
    assert CLASSES == sorted(CLASSES)
    assert set(CUBICASA_TO_CLASS.values()) <= set(CLASSES)
    assert set(CUBICASA_MODEL_TO_CLASS.values()) <= set(CLASSES)


def test_crop_is_square_and_padded(plan):
    img, poly = plan
    crop = crop_room(img, poly)
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    # a 2:1 room keeps its proportions: white padding above and below
    assert (crop[0] == 255).all() and (crop[-1] == 255).all()


@pytest.mark.parametrize("context, outside_value", [("keep", 128), ("dim", 204), ("none", 255)])
def test_context_modes(plan, context, outside_value):
    """Pixels outside the room: unchanged (keep), faded 60% toward white (dim) or white (none)."""
    img, poly = plan
    crop = crop_room(img, poly, context=context)
    assert (crop[CROP_SIZE // 2, CROP_SIZE // 2] == 20).all()      # the room itself is untouched
    # 30 px margin around a 200 px wide room -> the left ~29 crop pixels are context
    assert abs(int(crop[CROP_SIZE // 2, 10, 0]) - outside_value) <= 2


@pytest.mark.parametrize("poly", [np.zeros((2, 2)), np.array([[10, 10], [11, 10], [11, 11]], float)])
def test_degenerate_polygons_are_rejected(plan, poly):
    img, _ = plan
    assert crop_room(img, poly) is None


def test_rooms_file_roundtrip(tmp_path, plan):
    img, poly = plan
    path = rooms_json_path("some/folder/plan_1.jpg", tmp_path)
    assert path.name == "plan_1.rooms.json"
    save_rooms_json(path, "plan_1.jpg", img.shape, [{"polygon_px": poly, "room_type": "kitchen"}])
    stored = json.loads(path.read_text())
    assert max(max(p) for p in stored["rooms"][0]["polygon"]) <= 1.0     # normalised coordinates
    (room,) = load_rooms_json(path, img.shape)
    assert room["room_type"] == "kitchen"
    assert np.allclose(room["polygon_px"], poly, atol=0.01)


def test_unreviewed_drafts_are_not_ground_truth(tmp_path, plan):
    img, poly = plan
    path = tmp_path / "draft.rooms.json"
    save_rooms_json(path, "x.jpg", img.shape,
                    [{"polygon_px": poly, "room_type": "bath", "source": "auto", "auto_type": "bath",
                      "confidence": 0.9}], reviewed=False)
    assert not rooms_file_reviewed(path)
    assert load_rooms_json(path, img.shape)[0]["room_type"] == "bath"           # visible for review
    assert load_rooms_json(path, img.shape, require_reviewed=True)[0]["room_type"] is None


def test_unknown_room_type_is_dropped(tmp_path, plan):
    img, poly = plan
    path = tmp_path / "p.rooms.json"
    path.write_text(json.dumps([{"polygon": (poly / [600, 400]).tolist(), "room_type": "ballroom"}]))
    assert load_rooms_json(path, img.shape)[0]["room_type"] is None
