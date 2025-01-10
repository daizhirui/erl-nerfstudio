import json
import os.path
from dataclasses import dataclass
from typing import List

import fire
import numpy as np


@dataclass(frozen=True)
class Frame:
    file_path: str
    transform_matrix: List[List[float]]


@dataclass(frozen=True)
class TransformJson:
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    frames: List[Frame]

    # distortion parameters
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    aabb_scale: int = 16
    camera_model: str = "OPENCV"


@dataclass(frozen=True)
class Camera:
    # matrix: np.ndarray
    camera_to_world: np.ndarray
    fx: float
    fy: float


@dataclass(frozen=True)
class CameraPath:
    seconds: float
    camera_path: List[Camera]
    render_height: int
    render_width: int
    camera_type: str = "perspective"


def convert_dataset_cameras_to_camera_path(
    # model_transform_json: str,
    dataset_transforms_json,
    output_json,
    time_per_frame=1.0,
):
    with open(dataset_transforms_json) as f:
        dataset_transforms = json.load(f)

    out = dict()
    num_frames = len(dataset_transforms["frames"])
    out["seconds"] = time_per_frame * num_frames
    out["render_height"] = dataset_transforms["h"]
    out["render_width"] = dataset_transforms["w"]
    out["camera_type"] = "perspective"

    camera_path = []

    # by default, NeRF-Studio automatically generates a global transformation which
    # translates the mean translation of poses in the train dataset to the origin
    # scales the train dataset such that abs(max axis translation) = 1
    # if os.path.exists(model_transform_json):
    #     with open(model_transform_json) as f:
    #         dataparser_transforms = json.load(f)
    #         model_transform = np.array(dataparser_transforms["transform"])
    #         model_scale = np.array(dataparser_transforms["scale"])
    # else:  # in case the model_transform_json does not exist
    #     model_transform = np.eye(4)[:3]  # (3, 4)
    #     model_scale = 1.0
    for frame in dataset_transforms["frames"]:
        # mat = model_transform @ np.array(frame["transform_matrix"])
        # mat[:3, 3] *= model_scale
        # mat = mat.tolist()
        # mat.append([0, 0, 0, 1])
        camera_path.append(
            dict(
                camera_to_world=frame["transform_matrix"],
                fx=dataset_transforms["fl_x"],
                fy=dataset_transforms["fl_y"],
            )
        )
    out["camera_path"] = camera_path
    with open(output_json, "w") as f:
        json.dump(out, f, indent=4)


if __name__ == "__main__":
    fire.Fire(convert_dataset_cameras_to_camera_path)
