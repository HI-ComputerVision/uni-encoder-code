# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/mask_former_panoptic_dataset_mapper.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import copy
import json
import logging
import os
import cv2

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from detectron2.data import MetadataCatalog
from model.utils.box_ops import masks_to_boxes
from model.data.tokenizer import SimpleTokenizer, Tokenize
from model.data.dataset_mappers.custom_augs import CustomColorAugSSDAugment
from fvcore.transforms.transform import HFlipTransform

__all__ = ["OneFormerUnifiedDatasetMapper"]


class OneFormerUnifiedDatasetMapper:
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
        num_queries,
        meta,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        task_seq_len,
        max_seq_len,
        semantic_prob,
        instance_prob,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.meta = meta
        self.name = name
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.num_queries = num_queries

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

        self.things = []
        for k, v in self.meta.thing_dataset_id_to_contiguous_id.items():
            self.things.append(v)
        self.class_names = self.meta.stuff_classes
        self.text_tokenizer = Tokenize(
            SimpleTokenizer(), max_seq_len=max_seq_len)
        self.task_tokenizer = Tokenize(
            SimpleTokenizer(), max_seq_len=task_seq_len)
        self.semantic_prob = semantic_prob
        self.instance_prob = instance_prob

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.SEG_MIN_SIZE_TRAIN,
                cfg.INPUT.SEG_MAX_SIZE_TRAIN,
                cfg.INPUT.SEG_MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )

        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(CustomColorAugSSDAugment(img_format=cfg.INPUT.FORMAT))
        # augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "meta": meta,
            "name": dataset_names[0],
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES - cfg.MODEL.TEXT_ENCODER.N_CTX,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "semantic_prob": cfg.INPUT.TASK_PROB.SEMANTIC,
            "instance_prob": cfg.INPUT.TASK_PROB.INSTANCE,
        }
        return ret

    def _get_semantic_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)

        classes = []
        texts = ["a semantic photo"] * self.num_queries
        masks = []
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if not np.all(mask == False):
                    if class_id not in classes:
                        cls_name = self.class_names[class_id]
                        classes.append(class_id)
                        masks.append(mask)
                        num_class_obj[cls_name] += 1
                    else:
                        idx = classes.index(class_id)
                        masks[idx] += mask
                        masks[idx] = np.clip(masks[idx], 0, 1).astype(np.bool)
                    label[mask] = class_id

        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            masks = BitMasks(
                torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            # Placeholder bounding boxes for stuff regions. Note that these are not used during training.
            instances.gt_bboxes = torch.stack(
                [torch.tensor([0., 0., 1., 1.])] * instances.gt_masks.shape[0])
        return instances, texts, label

    def _get_instance_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)

        classes = []
        texts = ["an instance photo"] * self.num_queries
        masks = []
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if class_id in self.things:
                if not segment_info["iscrowd"]:
                    mask = pan_seg_gt == segment_info["id"]
                    if not np.all(mask == False):
                        cls_name = self.class_names[class_id]
                        classes.append(class_id)
                        masks.append(mask)
                        num_class_obj[cls_name] += 1
                        label[mask] = class_id

        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            masks = BitMasks(
                torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_bboxes = masks_to_boxes(instances.gt_masks)
        return instances, texts, label

    def _get_panoptic_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)

        classes = []
        texts = ["a panoptic photo"] * self.num_queries
        masks = []
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if not np.all(mask == False):
                    cls_name = self.class_names[class_id]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1
                    label[mask] = class_id

        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            masks = BitMasks(
                torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_bboxes = masks_to_boxes(instances.gt_masks)
            for i in range(instances.gt_classes.shape[0]):
                # Placeholder bounding boxes for stuff regions. Note that these are not used during training.
                if instances.gt_classes[i].item() not in self.things:
                    instances.gt_bboxes[i] = torch.tensor([0., 0., 1., 1.])
        return instances, texts, label

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "OneFormerDatasetMapper should only be used for training!"

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        left_image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format)
        right_image = utils.read_image(
            dataset_dict["right_file_name"], format=self.img_format)
        left_prev_image = utils.read_image(
            dataset_dict["left_prev_image_file"], format=self.img_format)
        left_next_image = utils.read_image(
            dataset_dict["left_nxt_image_file"], format=self.img_format)
        utils.check_image_size(dataset_dict, left_image)
        utils.check_image_size(dataset_dict, right_image)
        utils.check_image_size(dataset_dict, left_prev_image)
        utils.check_image_size(dataset_dict, left_next_image)

        # semantic segmentation
        if "left_sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            left_sem_seg_gt = utils.read_image(dataset_dict.pop(
                "left_sem_seg_file_name")).astype("double")
        else:
            left_sem_seg_gt = None

        # panoptic segmentation
        if "left_pan_seg_file_name" in dataset_dict:
            left_pan_seg_gt = utils.read_image(
                dataset_dict.pop("left_pan_seg_file_name"), "RGB")
            left_segments_info = dataset_dict["left_segments_info"]
        else:
            left_pan_seg_gt = None
            left_segments_info = None

        if left_pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["left_file_name"]
                )
            )

        left_aug_input = T.AugInput(left_image, sem_seg=left_sem_seg_gt)
        left_aug_input, transforms = T.apply_transform_gens(
            self.tfm_gens, left_aug_input)
        left_image = left_aug_input.image
        right_image = transforms.apply_image(right_image)
        left_prev_image = transforms.apply_image(left_prev_image)
        left_next_image = transforms.apply_image(left_next_image)

        if left_sem_seg_gt is not None:
            left_sem_seg_gt = left_aug_input.sem_seg

        # apply the same transformation to panoptic segmentation
        left_pan_seg_gt = transforms.apply_segmentation(left_pan_seg_gt)

        from panopticapi.utils import rgb2id

        left_pan_seg_gt = rgb2id(left_pan_seg_gt)

        # Pad image and segmentation label here!
        left_image = torch.as_tensor(
            np.ascontiguousarray(left_image.transpose(2, 0, 1)))
        right_image = torch.as_tensor(
            np.ascontiguousarray(right_image.transpose(2, 0, 1)))
        left_prev_image = torch.as_tensor(
            np.ascontiguousarray(left_prev_image.transpose(2, 0, 1)))
        left_next_image = torch.as_tensor(
            np.ascontiguousarray(left_next_image.transpose(2, 0, 1)))
        if left_sem_seg_gt is not None:
            left_sem_seg_gt = torch.as_tensor(left_sem_seg_gt.astype("long"))
        left_pan_seg_gt = torch.as_tensor(left_pan_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            left_image_size = (left_image.shape[-2], left_image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - left_image_size[1],
                0,
                self.size_divisibility - left_image_size[0],
            ]
            left_image = F.pad(left_image, padding_size, value=128).contiguous()
            right_image = F.pad(right_image, padding_size,
                                value=128).contiguous()
            left_prev_image = F.pad(
                left_prev_image, padding_size, value=128).contiguous()
            left_next_image = F.pad(
                left_next_image, padding_size, value=128).contiguous()
            if left_sem_seg_gt is not None:
                left_sem_seg_gt = F.pad(
                    left_sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            left_pan_seg_gt = F.pad(
                left_pan_seg_gt, padding_size, value=0
            ).contiguous()  # 0 is the VOID panoptic label

        left_image_shape = (left_image.shape[-2], left_image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["left_image"] = left_image
        dataset_dict["right_image"] = right_image
        dataset_dict["left_prev_image"] = left_prev_image
        dataset_dict["left_next_image"] = left_next_image

        if "annotations" in dataset_dict:
            raise ValueError(
                "Pemantic segmentation dataset should not have 'annotations'.")

        prob_task = np.random.uniform(0, 1.)

        num_class_obj = {}

        for name in self.class_names:
            num_class_obj[name] = 0

        if prob_task < self.semantic_prob:
            task = "The task is semantic"
            instances, text, sem_seg = self._get_semantic_dict(
                left_pan_seg_gt, left_image_shape, left_segments_info, num_class_obj)
        elif prob_task < self.instance_prob:
            task = "The task is instance"
            instances, text, sem_seg = self._get_instance_dict(
                left_pan_seg_gt, left_image_shape, left_segments_info, num_class_obj)
        else:
            task = "The task is panoptic"
            instances, text, sem_seg = self._get_panoptic_dict(
                left_pan_seg_gt, left_image_shape, left_segments_info, num_class_obj)

        dataset_dict["sem_seg"] = torch.from_numpy(sem_seg).long()
        dataset_dict["instances"] = instances
        dataset_dict["orig_shape"] = left_image_shape
        dataset_dict["task"] = task
        dataset_dict["text"] = text
        dataset_dict["thing_ids"] = self.things

        if dataset_dict["cam_info_file"] is not None:
            with open(dataset_dict["cam_info_file"], 'r') as file:
                camera_data = json.load(file)

            baseline = camera_data.get('extrinsic', {}).get('baseline', 0.0)
            dataset_dict["baseline"] = baseline

        # # Extract intrinsic values
        # intrinsic_data = camera_data.get('intrinsic', {})
        # fx = intrinsic_data.get('fx', 0.0) / 2048. * 512.  # TODO: hardcoding
        # fy = intrinsic_data.get('fy', 0.0) / 1024. * 256.   # TODO: hardcoding
        # u0 = intrinsic_data.get('u0', 0.0) / 2048. * 512.  # TODO: hardcoding
        # v0 = intrinsic_data.get('v0', 0.0) / 1024. * 256.   # TODO: hardcoding
        # assert fx > 0 and fy > 0 and u0 > 0 and v0 > 0, "Invalid intrinsic values"
        # if isinstance(transforms[-1], HFlipTransform):
        #     u0 = 512 - u0

        # Create the 4x4 intrinsic K matrix
        K = np.array([[2262.52 / 2048 * 512, 0, 0.5 * 512, 0],
                      [0, 2265.30 / 1024 * 256, 0.5 * 256, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)

        inv_K = np.linalg.pinv(K)
        dataset_dict["K"] = torch.from_numpy(K).float()
        dataset_dict["inv_K"] = torch.from_numpy(inv_K).float()

        stereo_T = np.eye(4, dtype=np.float32)
        stereo_T[0, 3] = 0.1 if isinstance(transforms[-1], HFlipTransform) else -0.1

        dataset_dict["stereo_T"] = torch.from_numpy(stereo_T).float()

        return dataset_dict
