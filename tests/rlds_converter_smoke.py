#!/usr/bin/env python3
"""CPU-only tests for LeRobot-to-RLDS value and split contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


project_root = Path(__file__).resolve().parents[1]
module_path = project_root / "scripts" / "convert_lerobot_to_rlds.py"
spec = importlib.util.spec_from_file_location("convert_lerobot_to_rlds", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

image = np.zeros((3, 480, 640), dtype=np.float32)
item = {
    "observation.images.third_person": image,
    "observation.images.wrist": image,
    "observation.state": np.array([0, 1, 2, 3, 4, 5, 0.04], dtype=np.float32),
    "action": np.array([1, 2, 3, 4, 5, 6, 0.08], dtype=np.float32),
    "task": "pick up the green blocks",
}

first = module.make_rlds_step(item, 0, 700, None)
last = module.make_rlds_step(item, 699, 700, None)
assert first["observation"]["third_person"].shape == (480, 640, 3)
assert first["observation"]["third_person"].dtype == np.uint8
assert first["action"].shape == (7,)
assert first["is_first"] and not first["is_last"] and not first["is_terminal"]
assert last["is_last"] and last["is_terminal"] and last["reward"] == np.float32(1.0)

train, val = module._select_splits(1000, None, 0.05, 7)
assert len(train) == 950 and len(val) == 50
assert not set(train).intersection(val)
assert sorted(train + val) == list(range(1000))

print("rlds_converter_contract=PASS frames_per_episode_preserved=700 train=950 val=50")
