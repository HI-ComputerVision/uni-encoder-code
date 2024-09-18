# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/cityscapes_panoptic.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


def get_cityscapes_sequence_files(files_list_path, left_img_dir, left_sequence_img_dir, cam_info_dir, depth_dir):
    files = []
    with open(files_list_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        city, file_id = line.strip().split(' ')
        city_left_img_dir = os.path.join(left_img_dir, city)
        city_left_sequence_img_dir = os.path.join(left_sequence_img_dir, city)
        city_cam_info_dir = None if cam_info_dir is None else os.path.join(cam_info_dir, city)
        city_depth_dir = None if depth_dir is None else os.path.join(depth_dir, city)

        basename = file_id + '_leftImg8bit.png'
        left_image_file = os.path.join(city_left_img_dir, basename)
        left_image_basename = basename.split("_")
        left_prev_image_basename = left_image_basename.copy()
        left_prev_image_basename[2] = str(
            int(left_prev_image_basename[2]) - 2).zfill(6)
        left_prev_image_basename = "_".join(left_prev_image_basename)

        left_prev_image_file = os.path.join(
            city_left_sequence_img_dir, left_prev_image_basename)
        left_nxt_image_basename = left_image_basename.copy()
        left_nxt_image_basename[2] = str(
            int(left_nxt_image_basename[2]) + 2).zfill(6)
        left_nxt_image_basename = "_".join(left_nxt_image_basename)
        left_nxt_image_file = os.path.join(
            city_left_sequence_img_dir, left_nxt_image_basename)

        left_cam_info_file = None if city_cam_info_dir is None else os.path.join(
            city_cam_info_dir, basename.replace("_leftImg8bit.png", "_camera.json")
        )

        left_disp_file = None if city_depth_dir is None else os.path.join(
            city_depth_dir, basename
        )

        suffix = "_leftImg8bit.png"
        assert basename.endswith(suffix), basename
        basename = os.path.basename(basename)[: -len(suffix)]
        if PathManager.isfile(left_prev_image_file) and PathManager.isfile(left_nxt_image_file):
            files.append((left_image_file, left_prev_image_file,
                            left_nxt_image_file, left_cam_info_file, left_disp_file))

    assert len(files), "No images found in {}".format(left_img_dir)
    return files


def load_cityscapes_sequence(files_list_path, left_image_dir, left_sequence_img_dir, cam_info_dir, depth_dir):

    files = get_cityscapes_sequence_files(files_list_path, left_image_dir, left_sequence_img_dir, cam_info_dir, depth_dir)
    ret = []
    for left_image_file, left_prev_image_file, left_nxt_image_file, left_cam_info_file, left_disp_file in files:
        ret.append(
            {
                "type": "sequence",
                "file_name": left_image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(left_image_file))[
                        0].split("_")[:3]
                ),
                "left_prev_image_file": left_prev_image_file,
                "left_nxt_image_file": left_nxt_image_file,
                "cam_info_file": left_cam_info_file,
                "left_disp_file": left_disp_file,
            }
        )
    assert len(ret), f"No images found in {left_image_dir}!"
    return ret


def register_all_cityscapes_sequence(root):
    for key, (files_list_path, left_image_dir, left_sequence_img_dir, cam_info_dir, depth_dir) in _RAW_CITYSCAPES_SEQUENCE_SPLITS.items():
        files_list_path = os.path.join(root, files_list_path)
        left_image_dir = os.path.join(root, left_image_dir)
        left_sequence_img_dir = os.path.join(root, left_sequence_img_dir)
        cam_info_dir = None if cam_info_dir is None else os.path.join(root, cam_info_dir)
        depth_dir = None if depth_dir is None else os.path.join(root, depth_dir)

        if key in DatasetCatalog.list():
            DatasetCatalog.remove(key)

        DatasetCatalog.register(
            key, lambda x=files_list_path, y=left_image_dir, z=left_sequence_img_dir, t=cam_info_dir, u=depth_dir: load_cityscapes_sequence(
                x, y, z, t, u)
        )
        MetadataCatalog.get(key).set(
            left_image_root=left_image_dir,
            evaluator_type="cityscapes_depth",
        )


_RAW_CITYSCAPES_SEQUENCE_SPLITS = {
    "cityscapes_sequence_crop_full_sequence_train": (
        "cityscapes_full_crop/train_files.txt",
        "cityscapes_full_crop/leftImg8bit_sequence/train",
        "cityscapes_full_crop/leftImg8bit_sequence/train",
        "cityscapes_full_crop/camera/train",
        None,
    ),
    "cityscapes_crop_test": (
        "cityscapes_crop/test_files.txt",
        "cityscapes_crop/leftImg8bit/test",
        "cityscapes_crop/leftImg8bit_sequence/test",
        "cityscapes_crop/camera/test",
        "cityscapes_crop/gt_depths",
    ),
}


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes_sequence(_root)
