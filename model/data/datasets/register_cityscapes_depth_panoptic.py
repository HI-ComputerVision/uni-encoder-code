# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/cityscapes_panoptic.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


def get_cityscapes_panoptic_files(left_img_dir, left_gt_dir, right_img_dir, left_disp_dir, cam_info_dir, json_info, sequence_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(left_img_dir)
    logger.info(f"{len(cities)} cities found in '{left_img_dir}'.")
    left_image_dict, right_image_dict, left_disp_dict, cam_info_dict, left_prev_image, left_nxt_image = {}, {}, {}, {}, {}, {}
    for city in cities:
        city_left_img_dir = os.path.join(left_img_dir, city)
        city_right_img_dir = os.path.join(right_img_dir, city)
        city_left_disp_dir = os.path.join(left_disp_dir, city)
        city_cam_info_dir = os.path.join(cam_info_dir, city)
        city_sequence_dir = os.path.join(sequence_dir, city)
        for basename in PathManager.ls(city_left_img_dir):
            left_image_file = os.path.join(city_left_img_dir, basename)
            left_image_basename = basename.split("_")
            left_prev_image_basename = left_image_basename.copy()
            left_prev_image_basename[2] = str(
                int(left_prev_image_basename[2]) - 1).zfill(6)
            left_prev_image_basename = "_".join(left_prev_image_basename)
            left_prev_image_file = os.path.join(
                city_sequence_dir, left_prev_image_basename)
            left_nxt_image_basename = left_image_basename.copy()
            left_nxt_image_basename[2] = str(
                int(left_nxt_image_basename[2]) + 1).zfill(6)
            left_nxt_image_basename = "_".join(left_nxt_image_basename)
            left_nxt_image_file = os.path.join(
                city_sequence_dir, left_nxt_image_basename)
            right_image_file = os.path.join(
                city_right_img_dir, basename.replace('left', 'right'))
            left_disp_file = os.path.join(
                city_left_disp_dir, basename.replace('leftImg8bit', 'disparity'))
            cam_info_file = os.path.join(
                city_cam_info_dir, basename.replace('leftImg8bit.png', 'camera.json'))

            suffix = "_leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = os.path.basename(basename)[: -len(suffix)]

            left_image_dict[basename] = left_image_file
            right_image_dict[basename] = right_image_file
            left_disp_dict[basename] = left_disp_file
            cam_info_dict[basename] = cam_info_file
            left_prev_image[basename] = left_prev_image_file
            left_nxt_image[basename] = left_nxt_image_file

    for ann in json_info["annotations"]:
        left_image_file = left_image_dict.get(ann["image_id"], None)
        right_image_file = right_image_dict.get(ann["image_id"], None)
        left_disp_file = left_disp_dict.get(ann["image_id"], None)
        cam_info_file = cam_info_dict.get(ann["image_id"], None)
        left_prev_image_file = left_prev_image.get(ann["image_id"], None)
        left_nxt_image_file = left_nxt_image.get(ann["image_id"], None)
        assert left_image_file is not None, "No left image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        assert right_image_file is not None, "No right image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        assert left_disp_file is not None, "No left disparity {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        assert cam_info_file is not None, "No camera info {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        assert left_prev_image_file is not None, "No left prev image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        assert left_nxt_image_file is not None, "No left nxt image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        left_label_file = os.path.join(left_gt_dir, ann["file_name"])
        left_segments_info = ann["segments_info"]
        files.append((left_image_file, left_label_file, right_image_file,
                     left_disp_file, cam_info_file, left_segments_info, left_prev_image_file, left_nxt_image_file))

    assert len(files), "No images found in {}".format(left_img_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files


def load_cityscapes_panoptic(left_image_dir, left_gt_dir, right_image_dir, left_disp_dir, cam_info_dir, gt_json, sequence_dir, meta):
    """
    Args:
        left_image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        right_image_dir (str): path to the raw dataset. e.g., "~/cityscapes/rightImg8bit/train".
        left_gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        gt_json
    ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa

    with open(gt_json) as f:
        json_info = json.load(f)

    files = get_cityscapes_panoptic_files(
        left_image_dir, left_gt_dir, right_image_dir, left_disp_dir, cam_info_dir, json_info, sequence_dir)
    ret = []
    for left_image_file, left_label_file, right_image_file, left_disp_file, cam_info_file, left_segments_info, left_prev_image_file, left_nxt_image_file in files:
        left_sem_label_file = (
            left_image_file.replace("leftImg8bit", "gtFine").split(".")[
                0] + "_labelTrainIds.png"
        )
        left_segments_info = [_convert_category_id(
            x, meta) for x in left_segments_info]
        ret.append(
            {
                "file_name": left_image_file,
                "right_file_name": right_image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(left_image_file))[
                        0].split("_")[:3]
                ),
                "left_disp_file": left_disp_file,
                "cam_info_file": cam_info_file,
                "left_sem_seg_file_name": left_sem_label_file,
                "left_pan_seg_file_name": left_label_file,
                "left_segments_info": left_segments_info,
                "left_prev_image_file": left_prev_image_file,
                "left_nxt_image_file": left_nxt_image_file,
            }
        )
    assert len(ret), f"No images found in {left_image_dir}!"
    assert PathManager.isfile(
        ret[0]["left_sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    assert PathManager.isfile(
        ret[0]["left_pan_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa
    return ret


_RAW_CITYSCAPES_PANOPTIC_SPLITS = {
    "cityscapes_dummy_fine_panoptic_train": (
        "cityscapes_dummy/leftImg8bit/train",
        "cityscapes_dummy/gtFine/cityscapes_panoptic_train",
        "cityscapes_dummy/rightImg8bit/train",
        "cityscapes_dummy/disparity/train",
        "cityscapes_dummy/camera/train",
        "cityscapes_dummy/gtFine/cityscapes_panoptic_train.json",
        "cityscapes_dummy/leftImg8bit_sequence/train",
    ),
    "cityscapes_dummy_fine_panoptic_val": (
        "cityscapes_dummy/leftImg8bit/val",
        "cityscapes_dummy/gtFine/cityscapes_panoptic_val",
        "cityscapes_dummy/rightImg8bit/val",
        "cityscapes_dummy/disparity/val",
        "cityscapes_dummy/camera/val",
        "cityscapes_dummy/gtFine/cityscapes_panoptic_val.json",
        "cityscapes_dummy/leftImg8bit_sequence/val",
    ),
    "cityscapes_fine_panoptic_train": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/gtFine/cityscapes_panoptic_train",
        "cityscapes/rightImg8bit/train",
        "cityscapes/disparity/train",
        "cityscapes/camera/train",
        "cityscapes/gtFine/cityscapes_panoptic_train.json",
        "cityscapes/leftImg8bit_sequence/train",
    ),
    "cityscapes_fine_panoptic_val": (
        "cityscapes/leftImg8bit/val",
        "cityscapes/gtFine/cityscapes_panoptic_val",
        "cityscapes/rightImg8bit/val",
        "cityscapes/disparity/val",
        "cityscapes/camera/val",
        "cityscapes/gtFine/cityscapes_panoptic_val.json",
        "cityscapes/leftImg8bit_sequence/val",
    ),
    "cityscapes_crop_fine_panoptic_train": (
        "cityscapes_crop/leftImg8bit/train",
        "cityscapes_crop/gtFine/cityscapes_panoptic_train",
        "cityscapes_crop/rightImg8bit/train",
        "cityscapes_crop/disparity/train",
        "cityscapes_crop/camera/train",
        "cityscapes_crop/gtFine/cityscapes_panoptic_train.json",
        "cityscapes_crop/leftImg8bit_sequence/train",
    ),
    "cityscapes_crop_fine_panoptic_val": (
        "cityscapes_crop/leftImg8bit/val",
        "cityscapes_crop/gtFine/cityscapes_panoptic_val",
        "cityscapes_crop/rightImg8bit/val",
        "cityscapes_crop/disparity/val",
        "cityscapes_crop/camera/val",
        "cityscapes_crop/gtFine/cityscapes_panoptic_val.json",
        "cityscapes_crop/leftImg8bit_sequence/val",
    ),
    # "cityscapes_fine_panoptic_test": not supported yet
}


def register_all_cityscapes_panoptic(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (left_image_dir, left_gt_dir, right_image_dir, left_disp_dir, cam_info_dir, gt_json, sequence_dir) in _RAW_CITYSCAPES_PANOPTIC_SPLITS.items():
        left_image_dir = os.path.join(root, left_image_dir)
        left_gt_dir = os.path.join(root, left_gt_dir)
        right_image_dir = os.path.join(root, right_image_dir)
        left_disp_dir = os.path.join(root, left_disp_dir)
        cam_info_dir = os.path.join(root, cam_info_dir)
        gt_json = os.path.join(root, gt_json)
        sequence_dir = os.path.join(root, sequence_dir)

        if key in DatasetCatalog.list():
            DatasetCatalog.remove(key)

        DatasetCatalog.register(
            key, lambda x=left_image_dir, y=left_gt_dir, z=right_image_dir, u=left_disp_dir, v=cam_info_dir, t=gt_json, s=sequence_dir: load_cityscapes_panoptic(
                x, y, z, u, v, t, s, meta)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=left_gt_dir,
            left_image_root=left_image_dir,
            right_image_root=right_image_dir,
            left_disp_dir=left_disp_dir,
            cam_info_dir=cam_info_dir,
            panoptic_json=gt_json,
            sequence_dir=sequence_dir,
            gt_dir=left_gt_dir.replace("cityscapes_panoptic_", ""),
            evaluator_type="cityscapes_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes_panoptic(_root)
