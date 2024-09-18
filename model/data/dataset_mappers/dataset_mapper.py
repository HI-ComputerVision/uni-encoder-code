# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/dataset_mapper.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import copy
import json
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from PIL import Image
from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.utils.file_io import PathManager
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from model.data.tokenizer import SimpleTokenizer, Tokenize

__all__ = ["DatasetMapper"]


def build_augmentation(cfg, is_train, for_segmentation=True):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    prefix = "SEG_" if for_segmentation else "DEPTH_"
    if is_train:
        min_size = getattr(cfg.INPUT, f"{prefix}MIN_SIZE_TRAIN")
        max_size = getattr(cfg.INPUT, f"{prefix}MAX_SIZE_TRAIN")
        sample_style = getattr(cfg.INPUT, f"{prefix}MIN_SIZE_TRAIN_SAMPLING")
    else:
        min_size = getattr(cfg.INPUT, f"{prefix}MIN_SIZE_TEST")
        max_size = getattr(cfg.INPUT, f"{prefix}MAX_SIZE_TEST")
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation


def read_image(file_name, format=None, dataset='cs'):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with PathManager.open(file_name, "rb") as f:
        if dataset == "cs":
            h, w = 192, 512
        elif dataset == "kitti":
            h, w = 192, 640
        else:
            raise NotImplementedError
        image = Image.open(f).resize((w, h), Image.LANCZOS)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        seg_augmentations: List[Union[T.Augmentation, T.Transform]],
        dep_augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        task_seq_len: int,
        task: str = "panoptic",
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.seg_augmentations      = T.AugmentationList(seg_augmentations)
        self.dep_augmentations      = T.AugmentationList(dep_augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.task_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=task_seq_len)
        self.task = task
        assert self.task in ["panoptic", "semantic", "instance"]

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[DatasetMapper] Augmentations used in {mode} for segmentations: {seg_augmentations}")
        logger.info(
            f"[DatasetMapper] Augmentations used in {mode} for depth: {dep_augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        seg_augs = build_augmentation(cfg, is_train, for_segmentation=True)
        dep_augs = build_augmentation(cfg, is_train, for_segmentation=False)
        if cfg.INPUT.SEG_CROP.ENABLED and is_train:
            seg_augs.insert(0, T.RandomCrop(
                cfg.INPUT.SEG_CROP.TYPE, cfg.INPUT.SEG_CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        if cfg.INPUT.DEPTH_CROP.ENABLED and is_train:
            dep_augs.insert(0, T.RandomCrop(
                cfg.INPUT.DEPTH_CROP.TYPE, cfg.INPUT.DEPTH_CROP.SIZE))

        ret = {
            "is_train": is_train,
            "seg_augmentations": seg_augs,
            "dep_augmentations": dep_augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "recompute_boxes": recompute_boxes,
            "task": cfg.MODEL.TEST.TASK,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        # Check the type of dataset and process accordingly
        if dataset_dict.get("type") == "segmentation":
            return self.process_segmentation_data(dataset_dict)
        elif dataset_dict.get("type") == "sequence":
            return self.process_sequence_data(dataset_dict)
        else:
            raise ValueError("Unknown dataset type: {}".format(dataset_dict.get("type")))

    def process_segmentation_data(self, dataset_dict):
        left_image = utils.read_image(
            dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, left_image)

        task = f"The task is {self.task}"
        dataset_dict["task"] = task

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "left_sem_seg_file_name" in dataset_dict:
            left_sem_seg_gt = utils.read_image(dataset_dict.pop(
                "left_sem_seg_file_name"), "L").squeeze(2)
        else:
            left_sem_seg_gt = None

        seg_aug_input = T.AugInput(left_image, sem_seg=left_sem_seg_gt)
        transforms = self.seg_augmentations(seg_aug_input)
        left_image, left_sem_seg_gt = seg_aug_input.image, seg_aug_input.sem_seg

        image_shape = left_image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["left_image"] = torch.as_tensor(
            np.ascontiguousarray(left_image.transpose(2, 0, 1)))
        if left_sem_seg_gt is not None:
            dataset_dict["left_sem_seg"] = torch.as_tensor(
                left_sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("left_sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict

    def process_sequence_data(self, dataset_dict):
        # Hardcoded for KITTI dataset
        left_image = read_image(
            dataset_dict["file_name"], format=self.image_format, dataset="kitti")
        utils.check_image_size(dataset_dict, left_image)
        load_prev_next = dataset_dict["left_prev_image_file"] is not None
        if load_prev_next:
            left_prev_image = read_image(
                dataset_dict["left_prev_image_file"], format=self.image_format, dataset="kitti")
            left_next_image = read_image(
                dataset_dict["left_nxt_image_file"], format=self.image_format, dataset="kitti")
            utils.check_image_size(dataset_dict, left_prev_image)
            utils.check_image_size(dataset_dict, left_next_image)

        dep_aug_input = T.AugInput(left_image)
        transforms = self.dep_augmentations(dep_aug_input)
        left_image = dep_aug_input.image
        if load_prev_next:
            left_prev_image = transforms.apply_image(left_prev_image)
            left_next_image = transforms.apply_image(left_next_image)

        image_shape = left_image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["left_image"] = torch.as_tensor(
            np.ascontiguousarray(left_image.transpose(2, 0, 1)))
        if load_prev_next:
            dataset_dict["left_prev_image"] = torch.as_tensor(
                np.ascontiguousarray(left_prev_image.transpose(2, 0, 1)))
            dataset_dict["left_next_image"] = torch.as_tensor(
                np.ascontiguousarray(left_next_image.transpose(2, 0, 1)))

        if getattr(dataset_dict, "cam_info_file", None) is not None:
            with open(dataset_dict["cam_info_file"], 'r') as file:
                camera_data = json.load(file)

            # Extract intrinsic values
            baseline = camera_data.get('extrinsic', {}).get('baseline', 0.0)
            dataset_dict["baseline"] = baseline

        return dataset_dict
