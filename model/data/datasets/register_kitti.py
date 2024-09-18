# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/cityscapes_panoptic.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


def get_kitti_sequence_files(data_root, files_list, img_ext=".jpg"):
    with open(files_list, 'r') as f:
        lines = f.read().splitlines()
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    files = []

    for line in lines:
        info = line.split()
        folder = info[0]
        if len(info) == 3:
            frame_index = int(info[1])
        else:
            frame_index = 0

        if len(info) == 3:
            side = info[2]
        else:
            side = None

        f_str = "{:010d}{}".format(frame_index, img_ext)
        left_image_file = os.path.join(
            data_root, folder, "image_0{}/data".format(side_map[side]), f_str)
        left_prev_image_file = os.path.join(
            data_root, folder, "image_0{}/data".format(side_map[side]), "{:010d}{}".format(frame_index - 1, img_ext))
        left_nxt_image_file = os.path.join(
            data_root, folder, "image_0{}/data".format(side_map[side]), "{:010d}{}".format(frame_index + 1, img_ext))

        calib_path = os.path.join(data_root, folder.split("/")[0])
        velo_filename = os.path.join(
            data_root,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        if os.path.isfile(left_image_file):
            if os.path.isfile(left_prev_image_file) and \
                    os.path.isfile(left_nxt_image_file) and \
                    os.path.isdir(calib_path) and \
                    os.path.isfile(velo_filename):
                files.append((left_image_file, left_prev_image_file,
                             left_nxt_image_file, calib_path, velo_filename, side))
            elif os.path.isdir(calib_path) and \
                    os.path.isfile(velo_filename):
                files.append((left_image_file, None, None, calib_path, velo_filename, side))
            else:
                raise NotImplementedError

    assert len(files), "No images found in {}".format(data_root)
    return files


def load_kitti_sequence(data_root, files_list, img_ext=".jpg"):

    files = get_kitti_sequence_files(data_root, files_list, img_ext)
    ret = []
    for left_image_file, left_prev_image_file, left_nxt_image_file, calib_path, velo_file, side in files:
        ret.append(
            {
                "type": "sequence",
                "file_name": left_image_file,
                "image_id": os.path.splitext(os.path.basename(left_image_file))[0],
                "left_prev_image_file": left_prev_image_file,
                "left_nxt_image_file": left_nxt_image_file,
                "calib_path": calib_path,
                "velo_file": velo_file,
                "side": side,
            }
        )
    assert len(ret), f"No images found in {data_root}!"
    return ret


def register_all_cityscapes_sequence(root):
    for key, (data_root, files_list_path, ext) in _RAW_CITYSCAPES_SEQUENCE_SPLITS.items():
        data_root = os.path.join(root, data_root)
        files_list_path = os.path.join(root, files_list_path)

        if key in DatasetCatalog.list():
            DatasetCatalog.remove(key)

        DatasetCatalog.register(
            key, lambda x=data_root, y=files_list_path, z=ext: load_kitti_sequence(
                x, y, z)
        )
        MetadataCatalog.get(key).set(
            left_image_root=data_root,
            evaluator_type="kitti_depth",
        )


_RAW_CITYSCAPES_SEQUENCE_SPLITS = {
    "KITTI_eigen_zhou_train_split": (
        "kitti_data",
        "kitti_data/eigen_zhou_train_files_kitti.txt",
        ".jpg",
    ),
    "KITTI_standard_eigen_test_split": (
        "kitti_data",
        "kitti_data/standard_eigen_test_files.txt",
        ".jpg",
    ),
}


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes_sequence(_root)
