"""CPU-only checks for the model-compatible client image transport."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".runtime/sources/openvla-oft"))

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("client resize smoke requires CUDA_VISIBLE_DEVICES='' exactly")
os.environ["PIPER_ACTION_CHUNK"] = "20"
os.environ["ROBOT_PLATFORM"] = "PIPER"

from openvla_pipeline.model_io import (
    ACTION_KEYS,
    MODEL_IMAGE_SIZE,
    observation_to_request,
    request_to_observation,
    resize_image_for_policy,
)
from experiments.robot import openvla_utils


def main() -> None:
    yy, xx = np.mgrid[:480, :640]
    image = np.stack(
        (xx % 256, yy % 256, (xx // 2 + yy // 2) % 256), axis=-1
    ).astype(np.uint8)
    expected = openvla_utils.resize_image_for_policy(image, MODEL_IMAGE_SIZE)
    actual = resize_image_for_policy(image)
    assert actual.shape == (224, 224, 3)
    assert actual.dtype == np.uint8
    assert np.array_equal(actual, expected)

    red = np.zeros((480, 640, 3), dtype=np.uint8)
    red[..., 0] = 255
    red_resized = resize_image_for_policy(red)
    assert float(red_resized[..., 0].mean()) > 250.0
    assert float(red_resized[..., 1:].mean()) < 2.0

    observation = {
        **{key: 0.0 for key in ACTION_KEYS},
        "third_person": image,
        "wrist": red,
    }
    timings: dict[str, float] = {}
    request = observation_to_request(
        observation,
        "two block pnp",
        "resize-smoke",
        resize_images=True,
        timings=timings,
    )
    decoded, task, request_id = request_to_observation(request)
    assert task == "two block pnp" and request_id == "resize-smoke"
    assert np.array_equal(decoded["full_image"], actual)
    assert np.array_equal(decoded["wrist_image"], red_resized)
    assert timings["client_resize_ms"] > 0.0

    # The pinned server helper must skip its JPEG/resize branch for client-224
    # inputs; center crop remains a separate downstream operation.
    original_resize = openvla_utils.resize_image_for_policy

    def unexpected_resize(*_args, **_kwargs):
        raise AssertionError("server applied a second model resize")

    openvla_utils.resize_image_for_policy = unexpected_resize
    try:
        prepared = openvla_utils.prepare_images_for_vla(
            [decoded["full_image"], decoded["wrist_image"]],
            SimpleNamespace(center_crop=False),
        )
    finally:
        openvla_utils.resize_image_for_policy = original_resize
    assert [item.size for item in prepared] == [(224, 224), (224, 224)]

    print("client_resize_shape=224x224")
    print("client_resize_color_order=RGB")
    print("client_resize_matches_pinned_oft=True")
    print("request_round_trip=True")
    print("server_double_resize=False")


if __name__ == "__main__":
    main()
