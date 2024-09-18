# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from model.modeling.motion_decoder.dynamo_motion_decoder_mod import MotionDecoderV2
from model.modeling.pose_decoder.resnet_like_pose_decoder import ResNetLike

from .modeling.monodepth_loss import transformation_from_parameters

from einops import rearrange
from .modeling.transformer_decoder.oneformer_transformer_decoder import MLP
from model.data.tokenizer import SimpleTokenizer, Tokenize


@META_ARCH_REGISTRY.register()
class OneFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        pose_decoder: nn.Module,
        motion_decoder: nn.Module,
        motion_mask: nn.Module,
        task_mlp: nn.Module,
        text_encoder: nn.Module,
        text_projector: nn.Module,
        prompt_ctx: nn.Embedding,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        depth_on: bool,
        detection_on: bool,
        test_topk_per_image: int,
        task_seq_len: int,
        max_seq_len: int,
        is_demo: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.pose_decoder = pose_decoder
        self.motion_decoder = motion_decoder
        self.motion_mask = motion_mask
        self.task_mlp = task_mlp
        self.text_encoder = text_encoder
        self.text_projector = text_projector
        self.prompt_ctx = prompt_ctx
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(
            pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(
            pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.depth_on = depth_on
        self.panoptic_on = panoptic_on
        self.detection_on = detection_on
        self.test_topk_per_image = test_topk_per_image

        self.text_tokenizer = Tokenize(
            SimpleTokenizer(), max_seq_len=max_seq_len)
        self.task_tokenizer = Tokenize(
            SimpleTokenizer(), max_seq_len=task_seq_len)
        self.is_demo = is_demo

        try:
            self.thing_indices = [
                k for k in self.metadata.thing_dataset_id_to_contiguous_id.keys()]
        except:
            self.thing_indices = []

        # if not self.semantic_on:
        #     assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        pose_decoder = ResNetLike()
        motion_decoder = MotionDecoderV2(num_input_images=2, out_dim=3)
        motion_mask = MotionDecoderV2(num_input_images=2, out_dim=1)

        text_encoder = None
        text_projector = None
        prompt_ctx = None

        task_mlp = MLP(cfg.INPUT.TASK_SEQ_LEN, cfg.MODEL.ONE_FORMER.HIDDEN_DIM,
                       cfg.MODEL.ONE_FORMER.HIDDEN_DIM, 2)


        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pose_decoder": pose_decoder,
            "motion_decoder": motion_decoder,
            "motion_mask": motion_mask,
            "task_mlp": task_mlp,
            "prompt_ctx": prompt_ctx,
            "text_encoder": text_encoder,
            "text_projector": text_projector,
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.ONE_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.TEST.PANOPTIC_ON
                or cfg.MODEL.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.TEST.PANOPTIC_ON,
            "depth_on": cfg.MODEL.TEST.DEPTH_ON,
            "detection_on": cfg.MODEL.TEST.DETECTION_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
            "is_demo": cfg.MODEL.IS_DEMO,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def encode_text(self, text):
        assert text.ndim in [2, 3], text.ndim
        b = text.shape[0]
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)
            squeeze_dim = True

        # [B, C]
        x = self.text_encoder(text)

        text_x = self.text_projector(x)

        if squeeze_dim:
            text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
            if self.prompt_ctx is not None:
                text_ctx = self.prompt_ctx.weight.unsqueeze(
                    0).repeat(text_x.shape[0], 1, 1)
                text_x = torch.cat([text_x, text_ctx], dim=1)

        return {"texts": text_x}

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        processed_results = []
        if len([1 for e in batched_inputs if e['type'] == 'segmentation']) > 0:
            images4seg = [x["left_image"].to(self.device) for x in batched_inputs if x["type"] == "segmentation"]
            images4seg = [(x - self.pixel_mean) / self.pixel_std for x in images4seg]
            images4seg = ImageList.from_tensors(images4seg, self.size_divisibility)

            tasks = torch.cat([self.task_tokenizer(x["task"]).to(
                self.device).unsqueeze(0) for x in batched_inputs if x["type"] == "segmentation"], dim=0)
            tasks = self.task_mlp(tasks.float())
            features4seg = self.backbone(images4seg.tensor)
            outputs4seg, _ = self.sem_seg_head(features4seg, None, tasks)

            mask_cls_results = outputs4seg["pred_logits"]
            mask_pred_results = outputs4seg["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images4seg.tensor.shape[-2], images4seg.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            del outputs4seg

            for i, data in enumerate(zip(
                mask_cls_results, mask_pred_results, batched_inputs, images4seg.image_sizes
            )):
                mask_cls_result, mask_pred_result, input_per_image, image_size = data
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result, input_per_image["task"])
                    processed_results[-1]["instances"] = instance_r

                if self.detection_on:
                    bbox_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result, input_per_image["task"])
                    processed_results[-1]["box_instances"] = bbox_r

        if len([1 for e in batched_inputs if e['type'] == 'sequence']) > 0:
            processed_results.append({})
            images4depth = [x["left_image"].to(self.device) for x in batched_inputs if x["type"] == "sequence"]
            images4depth = [(x - self.pixel_mean) / self.pixel_std for x in images4depth]
            images4depth = ImageList.from_tensors(images4depth, self.size_divisibility)

            p_images = [x["left_prev_image"].to(self.device) for x in batched_inputs if x["type"] == "sequence"]
            p_images = [(x - self.pixel_mean) / self.pixel_std for x in p_images]
            p_images = ImageList.from_tensors(p_images, self.size_divisibility)
            features4depth = self.backbone(images4depth.tensor)
            p_features4depth = self.backbone(p_images.tensor)

            features4motion_0 = {}
            for k, v in features4depth.items():
                f_0 = v
                f_p = p_features4depth[k]
                features4motion_0[k] = torch.cat([f_p, f_0], dim=1)

            pose_images_0 = [
                torch.cat([p_images[i], images4depth[i]], dim=0) for i in range(len(images4depth))]
            pose_images_0 = ImageList.from_tensors(pose_images_0, self.size_divisibility)
            axisangle_0, translation_0 = self.pose_decoder(features4motion_0)
            axisangle_0, translation_0 = axisangle_0[:, 0], translation_0[:, 0]

            pose_outputs = {
                ("axisangle", 0, -1): axisangle_0,
                ("translation", 0, -1): translation_0,
                ("cam_T_cam", 0, -1): transformation_from_parameters(
                    axisangle_0, translation_0, invert=True),
            }

            motion_outputs_0 = {
                "motion_input": {
                    "full_res_input": pose_images_0.tensor, **features4motion_0},
            }

            ego_translation_0 = pose_outputs[('translation', 0, -1)].detach()
            ego_axisangle_0 = pose_outputs[('axisangle', 0, -1)].detach()
            ego_motion_0 = torch.cat((ego_translation_0, ego_axisangle_0), -1).permute(0, 2, 1).unsqueeze(3)

            motion_out_0 = self.motion_decoder(motion_outputs_0, ego_motion_0)
            motion_outputs_0.update({(k[0], -1, k[1]): v for k, v in motion_out_0.items()})
            motion_prob_0 = self.motion_mask(motion_outputs_0, ego_motion_0)
            motion_outputs_0.update({(k[0], -1, k[1]): v for k, v in motion_prob_0.items()})

            dummy_tasks = torch.zeros(
                (images4depth.tensor.shape[0], self.task_mlp.layers[1].out_features), device=self.device)
            _, outputs4depth = self.sem_seg_head(None, features4depth, dummy_tasks)

            disp_results = outputs4depth[('disp', 0)]
            del outputs4depth

            if self.depth_on:
                disp_r = retry_if_cuda_oom(
                    self.depth_inference)(disp_results)
                processed_results[-1]["disp_results"] = disp_r
                processed_results[-1]["motion_mask"] = motion_outputs_0[('motion_mask', -1, 0)]
                processed_results[-1]["complete_flow"] = motion_outputs_0[('complete_flow', -1, 0)]
                processed_results[-1]["cam_T_cam"] = pose_outputs[("cam_T_cam", 0, -1)]
        return processed_results

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros(
            (h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(
                                pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(
                                pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, task_type):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(
            0).repeat(self.num_queries, 1).flatten(0, 1)

        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(
            0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # Only consider scores with confidence over [self.object_mask_threshold] for demo
        if self.is_demo:
            keep = scores_per_image > self.object_mask_threshold
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        if 'ade20k' in self.metadata.name and not self.is_demo and "instance" in task_type:
            for i in range(labels_per_image.shape[0]):
                labels_per_image[i] = self.thing_indices.index(
                    labels_per_image[i].item())

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        if self.detection_on:
            # Uncomment the following to get boxes from masks (this is slow)
            result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(
            1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def depth_inference(self, disp_results):
        return disp_results