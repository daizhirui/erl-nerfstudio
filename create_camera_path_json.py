import numpy as np
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Dict
import yaml
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup


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
class KeyFrame:
    # matrix: np.ndarray
    camera_to_world: np.ndarray
    fov: float
    aspect: float
    override_transition_enabled: bool = False
    override_transition_sec: float = 0.0


@dataclass(frozen=True)
class CameraPath:
    default_fov: float
    keyframes: List[KeyFrame]
    camera_path: List[KeyFrame]
    render_height: int
    render_width: int
    is_cycle: bool = False
    smoothness_value: float = 0.0
    seconds: float = 1.0
    fps: int = 30
    camera_type: str = "perspective"
    default_transition_sec: float = 0.0


oTc = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

nTo = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
)
cTn = oTc.T @ nTo.T

test_pose = np.array(
    [
        [
            0.7645703724048531,
            0.17262940210214459,
            -0.6209921377687162,
            0.27380807756798403,
            -0.6445402591310369,
            0.20477747415065722,
            -0.7366369800924829,
            -0.6898194169735075,
            5.551115123125783e-17,
            0.9634652435922809,
            0.2678333892360454,
            0.013879851639431357,
            0,
            0,
            0,
            1,
        ]
    ]
)


# # Filenames
DATASET_FOLDER = os.getcwd()
dataset_directory = os.path.join(DATASET_FOLDER, "data/replica_room_0/test_dataset")
test_poses_pkl_filename = os.path.join(dataset_directory, "poses.pkl")
dataparser_transform_filename = os.path.join(
    DATASET_FOLDER, "outputs/replica_room_0/nerfacto/2024-11-26_180323/dataparser_transforms.json"
)  # update this path
param_file_dir = os.path.join(dataset_directory, "sensor_setting.yaml")


# Scale and Transform done internally in Nerfstudio
with open(dataparser_transform_filename) as f:
    dataparser_transform = json.load(f)
Tdata_parser = np.array(dataparser_transform["transform"])
Tdata_parser = np.vstack((Tdata_parser, np.array([0, 0, 0, 1])))
scale_dataparser = dataparser_transform["scale"]

test_pose = test_pose.reshape((4, 4))
test_pose[:3, 3] = test_pose[:3, 3] / scale_dataparser
test_pose = np.linalg.inv(Tdata_parser) @ test_pose
ctw = test_pose @ np.linalg.inv(cTn)
print(ctw)

# Generate test poses
# with open(test_poses_pkl_filename, 'rb') as f:
#     test_poses = pickle.load(f)
# test_poses_list = []
# for frame in test_poses:
#     rotmat = frame[0]
#     trans = frame[1]
#     tform = np.hstack((rotmat, trans.reshape(3, 1)))
#     tform = np.vstack((tform, np.array([0, 0, 0, 1])))
#     tform = tform @ cTn
#     tform = Tdata_parser @ tform
#     tform[:3, 3] = tform[:3, 3] * scale_dataparser
#     test_poses_list.append(tform)


# # Get camera parameters
# with open(param_file_dir) as f:
#     sensor_param = yaml.load(f, Loader=yaml.FullLoader)
# transform_json = TransformJson(
#     fl_x=sensor_param['camera_fx'],
#     fl_y=sensor_param['camera_fy'],
#     cx=sensor_param['camera_cx'],
#     cy=sensor_param['camera_cy'],
#     frames=[],
#     w=sensor_param['image_width'],
#     h=sensor_param['image_height'],
# )
# fov_y = np.rad2deg(2 * np.arctan(sensor_param['image_height'] / (2 * sensor_param['camera_fy'])))

# camera_path_list = []
# for pose in test_poses_list:
#     keyframe = KeyFrame(
#         # matrix=pose.flatten().tolist(),
#         camera_to_world=pose.flatten().tolist(),
#         fov=fov_y,
#         aspect=sensor_param['image_width'] / sensor_param['image_height'],
#     )
#     camera_path_list.append(keyframe.__dict__),

# camera_path = CameraPath(
#     default_fov=fov_y,
#     keyframes=camera_path_list,
#     camera_path=camera_path_list,
#     render_height=sensor_param['image_height'],
#     render_width=sensor_param['image_width'],
# )
# camera_path_dict = camera_path.__dict__
# with open('camera_path_dparser_test.json', 'w') as f:
#     json.dump(camera_path_dict, f)
