# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/mask_former_panoptic_dataset_mapper.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import copy
import json
import logging

from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F

from panopticapi.utils import rgb2id
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from detectron2.data import MetadataCatalog
from model.data.dataset_mappers.dataset_mapper import read_image
from model.utils.box_ops import masks_to_boxes
from model.data.tokenizer import SimpleTokenizer, Tokenize
from model.data.dataset_mappers.custom_augs import CustomColorAugSSDAugment, CustomColorJitterAugment
from fvcore.transforms.transform import HFlipTransform, CropTransform
from detectron2.data.transforms import ResizeTransform

__all__ = ["DepthCityscapesMapper"]


class DepthCityscapesMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by OneFormer for universal segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        name,
        meta,
        depth_resize_augmentations,
        depth_color_jitter_augmentations,
        image_format,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.meta = meta
        self.name = name
        self.depth_resize_tfm_gens = depth_resize_augmentations
        self.depth_color_jitter_tfm_gens = depth_color_jitter_augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode} for depth: {depth_resize_augmentations} and {depth_color_jitter_augmentations}")


    @classmethod
    def from_config(cls, cfg, is_train=True):

        depth_resize_augs = []
        # depth_resize_augs = [
        #     T.ResizeShortestEdge(
        #         cfg.INPUT.DEPTH_MIN_SIZE_TRAIN,
        #         cfg.INPUT.DEPTH_MAX_SIZE_TRAIN,
        #         cfg.INPUT.DEPTH_MIN_SIZE_TRAIN_SAMPLING,
        #     )
        # ]
        # if cfg.INPUT.DEPTH_CROP.ENABLED:
        #     depth_resize_augs.append(
        #         T.RandomCrop_CategoryAreaConstraint(
        #             cfg.INPUT.DEPTH_CROP.TYPE,
        #             cfg.INPUT.DEPTH_CROP.SIZE,
        #         )
        #     )
        depth_resize_augs.append(T.RandomFlip())

        if cfg.INPUT.DEPTH_COLOR_JITTER:
            depth_color_jitter_augs = [CustomColorJitterAugment(img_format=cfg.INPUT.FORMAT)]

        # Assume always applies to the training set.
        dataset_names = (*cfg.DATASETS.TRAIN, *cfg.DATASETS.TRAIN)
        meta = MetadataCatalog.get(dataset_names[0])

        ret = {
            "is_train": is_train,
            "meta": meta,
            "name": dataset_names[0],
            "depth_resize_augmentations": depth_resize_augs,
            "depth_color_jitter_augmentations": depth_color_jitter_augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "OneFormerDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)

        # Check the type of dataset and process accordingly
        if dataset_dict.get("type") == "sequence":
            return self.process_sequence_data(dataset_dict)
        else:
            raise ValueError("Unknown dataset type: {}".format(dataset_dict.get("type")))

    def process_sequence_data(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        left_image = read_image(
            dataset_dict["file_name"], format=self.img_format, dataset="cs")
        left_prev_image = read_image(
            dataset_dict["left_prev_image_file"], format=self.img_format, dataset="cs")
        left_next_image = read_image(
            dataset_dict["left_nxt_image_file"], format=self.img_format, dataset="cs")
        utils.check_image_size(dataset_dict, left_image)
        utils.check_image_size(dataset_dict, left_prev_image)
        utils.check_image_size(dataset_dict, left_next_image)

        left_aug_input = T.AugInput(left_image)
        left_aug_input, resize_transforms = T.apply_transform_gens(
            self.depth_resize_tfm_gens, left_aug_input)
        left_image = left_aug_input.image
        left_prev_image = resize_transforms.apply_image(left_prev_image)
        left_next_image = resize_transforms.apply_image(left_next_image)

        orig_left_image = copy.deepcopy(left_image)
        orig_left_prev_image = copy.deepcopy(left_prev_image)
        orig_left_next_image = copy.deepcopy(left_next_image)

        left_aug_input = T.AugInput(left_image)
        left_aug_input, color_transforms = T.apply_transform_gens(
            self.depth_color_jitter_tfm_gens, left_aug_input)
        left_image = left_aug_input.image
        left_prev_image = color_transforms.apply_image(left_prev_image)
        left_next_image = color_transforms.apply_image(left_next_image)

        # if isinstance(resize_transforms[0], ResizeTransform):
        #     w = resize_transforms[0].w
        #     resize_ratio = resize_transforms[0].new_w / w
        # else:
        #     raise NotImplementedError

        # if len(resize_transforms) > 1:
        #     if isinstance(resize_transforms[1], CropTransform):
        #         new_w = resize_transforms[1].w
        #         new_h = resize_transforms[1].h
        #         x0, y0 = resize_transforms[1].x0, resize_transforms[1].y0
        #     else:
        #         raise NotImplementedError
        # else:
        #     new_w = resize_transforms[0].new_w
        #     new_h = resize_transforms[0].new_h
        #     x0, y0 = 0, 0

        left_image = torch.as_tensor(
            np.ascontiguousarray(left_image.transpose(2, 0, 1)))
        left_prev_image = torch.as_tensor(
            np.ascontiguousarray(left_prev_image.transpose(2, 0, 1)))
        left_next_image = torch.as_tensor(
            np.ascontiguousarray(left_next_image.transpose(2, 0, 1)))
        orig_left_image = torch.as_tensor(
            np.ascontiguousarray(orig_left_image.transpose(2, 0, 1)))
        orig_left_prev_image = torch.as_tensor(
            np.ascontiguousarray(orig_left_prev_image.transpose(2, 0, 1)))
        orig_left_next_image = torch.as_tensor(
            np.ascontiguousarray(orig_left_next_image.transpose(2, 0, 1)))

        if self.size_divisibility > 0:
            left_image_size = (left_image.shape[-2], left_image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - left_image_size[1],
                0,
                self.size_divisibility - left_image_size[0],
            ]
            left_image = F.pad(left_image, padding_size, value=128).contiguous()
            left_prev_image = F.pad(
                left_prev_image, padding_size, value=128).contiguous()
            left_next_image = F.pad(
                left_next_image, padding_size, value=128).contiguous()
            orig_left_image = F.pad(
                orig_left_image, padding_size, value=128).contiguous()
            orig_left_prev_image = F.pad(
                orig_left_prev_image, padding_size, value=128).contiguous()
            orig_left_next_image = F.pad(
                orig_left_next_image, padding_size, value=128).contiguous()

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["left_image"] = left_image
        dataset_dict["left_prev_image"] = left_prev_image
        dataset_dict["left_next_image"] = left_next_image
        dataset_dict["orig_left_image"] = orig_left_image
        dataset_dict["orig_left_prev_image"] = orig_left_prev_image
        dataset_dict["orig_left_next_image"] = orig_left_next_image

        with open(dataset_dict['cam_info_file'], 'r') as f:
            data = json.load(f)
            fx = data['intrinsic']['fx'] / 2048.0 * 512.0
            fy = data['intrinsic']['fy'] / 768.0 * 192.0
            u0 = data['intrinsic']['u0'] / 2048.0  * 512.0
            v0 = data['intrinsic']['v0'] / 768.0 * 192.0
        
        if isinstance(resize_transforms[-1], HFlipTransform):
            u0 = 512. - u0

        # Create the 4x4 intrinsic K matrix
        K = np.array([[fx, 0, u0, 0],
                      [0, fy, v0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)

        inv_K = np.linalg.pinv(K)
        dataset_dict["K"] = torch.from_numpy(K).float()
        dataset_dict["inv_K"] = torch.from_numpy(inv_K).float()

        # stereo_T = np.eye(4, dtype=np.float32)
        # stereo_T[0, 3] = 0.1 if isinstance(transforms[-1], HFlipTransform) else -0.1

        # dataset_dict["stereo_T"] = torch.from_numpy(stereo_T).float()

        return dataset_dict
