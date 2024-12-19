import os
import cv2
import numpy as np
import yaml
from dataclasses import dataclass
from typing import List
import json
import matplotlib.pyplot as plt
import pickle

DEPTH_SCALE = 6553.5  # Saved image is depth / DEPTH_SCALE

# Filenames
DATASET_FOLDER = os.getcwd()
DEPTH_IMAGE_FOLDER = os.path.join(DATASET_FOLDER, 'renders/mipnerf/replica_room_0')
DEPTH_PICKLE_FOLDER = os.path.join(DATASET_FOLDER, 'renders/mipnerf/replica_room_0_pickle')
GT_DEPTH_IMAGE_FOLDER = os.path.join(DATASET_FOLDER, 'data/replica_room_0/test_dataset/depth')
CAMERA_PATH_FILE = os.path.join(DATASET_FOLDER, 'data/replica_room_0/camera_paths/camera_path_dparser.json')
DEPTH_IMAGE_SAVE_DIR = os.path.join(DATASET_FOLDER, 'rendered_depth/mipnerf')

@dataclass(frozen=True)
class Frame:
    file_path: str
    transform_matrix: List[List[float]]


@dataclass(frozen=True)
class TransformJson:
    fl_x : float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    frames: List[Frame]
    # k1: float = -0.013472197525381842
    # k2: float = 0.007509466554079491
    # p1: float = -0.0011800209664517077
    # p2: float = 0.01116939407701522

    k1: float = 0.
    k2: float = 0.
    p1: float = 0.
    p2: float = 0.

    aabb_scale: int = 16
    camera_model: str = 'OPENCV'

def load_camera_path():
    with open(CAMERA_PATH_FILE) as f:
        camera_path_json = json.load(f)
    keyframe_list = camera_path_json['camera_path']
    camera_path_list = []
    for keyframe in keyframe_list:
        pose = keyframe['camera_to_world']
        camera_path_list.append(pose)
    return camera_path_list

def load_depth_images(path_to_folder: str) :
    depth_images = []
    filename_list = os.listdir(path_to_folder)
    filename_list.sort()
    for filename in filename_list:
        image_filename = os.path.join(path_to_folder, filename)
        depth_image = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
        depth_images.append(depth_image)
    return depth_images


def load_depth_pickles():
    depth_pickle = []
    filename_list = os.listdir(DEPTH_PICKLE_FOLDER)
    filename_list.sort()
    for filename in filename_list:
        pickle_filename = os.path.join(DEPTH_PICKLE_FOLDER, filename)
        with open(pickle_filename, 'rb') as f:
            depth_mat = pickle.load(f)
        depth_mat = np.squeeze(depth_mat)
        depth_mat *= DEPTH_SCALE
        depth_mat /= 0.32336484768491713
        depth_pickle.append(depth_mat)
    return depth_pickle


def compute_viewing_direction(u, v, camera_fx, camera_fy, camera_cx, camera_cy, camera_rotation):
    xu = -(u - camera_cx) / camera_fx
    yu = -(v - camera_cy) / camera_fy
    d = np.array([1.0, xu, yu])  # in camera frame
    d = d / np.linalg.norm(d)
    d = np.dot(camera_rotation, d)
    return d


def get_distance_images(depth_image, camera_params, camera_rotation):
    distance_image = np.zeros_like(depth_image)
    for u in range(camera_params.w):
        for v in range(camera_params.h):
            viewing_direction = compute_viewing_direction(u, v, camera_params.fl_x, camera_params.fl_y, camera_params.cx, camera_params.cy, camera_rotation)
            distance_image[v, u] = depth_image[v, u] * np.linalg.norm(viewing_direction)
    return distance_image


def get_camera_params():
    dataset_directory = os.path.join(DATASET_FOLDER, 'data/replica_room_0/test_dataset')
    param_file_dir = os.path.join(dataset_directory, 'sensor_setting.yaml')
    with open(param_file_dir) as f:
        sensor_param = yaml.load(f, Loader=yaml.FullLoader)
        transform_json = TransformJson(
            fl_x=sensor_param['camera_fx'],
            fl_y=sensor_param['camera_fy'],
            cx=sensor_param['camera_cx'],
            cy=sensor_param['camera_cy'],
            frames=[],
            w=sensor_param['image_width'],
            h=sensor_param['image_height'],
        )   
    return transform_json


def compute_mae(img1, img2):
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    return sum(abs(img1_flat - img2_flat))/len(img1_flat)

if __name__ == '__main__':
    depth_images = load_depth_images(DEPTH_IMAGE_FOLDER)
    gt_depth_images = load_depth_images(GT_DEPTH_IMAGE_FOLDER)
    camera_path = load_camera_path()
    depth_pickle_mats = load_depth_pickles()
    transform_json = get_camera_params()
    mae_list = []
    for i, frame in enumerate(camera_path):
        depth_image = depth_images[i]
        depth_mat = depth_pickle_mats[i]
        gt_depth_image = gt_depth_images[i]
        pose = np.array(camera_path[i]).reshape((4, 4))
        gt_distance_image =  get_distance_images(
            depth_image=gt_depth_image,
            camera_params=transform_json,
            camera_rotation=pose[:3,:3]
        )
        distance_image = get_distance_images(
            depth_image=depth_mat,
            camera_params=transform_json,
            camera_rotation=pose[:3,:3]
        )

        # fig = plt.figure()
        # ax0 = plt.subplot(1, 4, 1)
        # ax1 = plt.subplot(1, 4, 2)
        # ax2 = plt.subplot(1, 4, 3)
        # ax3 = plt.subplot(1, 4, 4)
        # im0 = ax0.imshow(distance_image)
        # im1 =ax1.imshow(gt_distance_image)
        # im2 = ax2.imshow(distance_image - gt_distance_image)
        # im3 = ax3.imshow(distance_image/gt_distance_image)

        # fig.colorbar(im0, ax=ax0, orientation='vertical')
        # fig.colorbar(im1, ax=ax1, orientation='vertical')
        # fig.colorbar(im2, ax=ax2, orientation='vertical')
        # fig.colorbar(im3, ax=ax3, orientation='vertical')

        # plt.waitforbuttonpress()
        # plt.show()
        
        img_save_dir = os.path.join(DEPTH_IMAGE_SAVE_DIR, str(i) + '.png')
        plt.imsave(img_save_dir, distance_image)

        mae = compute_mae(distance_image, gt_distance_image)
        mae_list.append(mae)

    mae_arr = np.array(mae_list)
    print(f"mean mae: {np.mean(mae_arr)}")
                
