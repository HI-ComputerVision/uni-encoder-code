# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import json
import numpy as np
import torch
import detectron2.data.transforms as T
from PIL import Image
from matplotlib import cm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.modeling import build_model

from model.modeling.monodepth_loss import BackprojectDepth, Project3D, disp_to_depth


__all__ = [
    "DefaultPredictor",
]


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.SEG_MIN_SIZE_TEST, cfg.INPUT.SEG_MIN_SIZE_TEST], cfg.INPUT.SEG_MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self.backproject_depth = BackprojectDepth(1, 192, 512).cuda()
        self.project_3d = Project3D(1, 192, 512).cuda()

    def __call__(self, original_image, previous_frame, task):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
                previous_frame = previous_frame[:, :, ::-1]
            height, width = original_image.shape[:2]
            aug4depth = T.ResizeTransform(
                height, width, 192, 512
            )
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image4depth = aug4depth.apply_image(original_image)
            prev_image4depth = aug4depth.apply_image(previous_frame)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image4depth = torch.as_tensor(image4depth.astype("float32").transpose(2, 0, 1))
            prev_image4depth = torch.as_tensor(prev_image4depth.astype("float32").transpose(2, 0, 1))

            task = f"The task is {task}"

            inputs = {"left_image": image4depth, "left_prev_image": prev_image4depth, "height": height, "width": width, "task": task, 'type': 'sequence'}
            predictions = self.model([inputs])[0]
            disp = predictions["disp_results"]
            disp, depth = disp_to_depth(disp)
            disp = disp[0, 0].cpu().numpy()
            vmax = np.percentile(disp, 95)
            disp = disp / vmax
            disp = np.clip(disp, 0, 1)
            disp = cm.magma(disp)[..., :3] * 255
            im = Image.fromarray(np.uint8(disp))
            depth_prediction = {'depth_inference': im}

            camera_info_path = '/data/home_hi/shared_projects/mit/data/DETECTRON_DATASETS/cityscapes_full_crop/camera/val/frankfurt/frankfurt_000000_000294_camera.json'
            with open(camera_info_path, 'r') as f:
                data = json.load(f)
                fx = data['intrinsic']['fx'] / 2048.0 * 512.0
                fy = data['intrinsic']['fy'] / 768.0 * 192.0
                u0 = data['intrinsic']['u0'] / 2048.0  * 512.0
                v0 = data['intrinsic']['v0'] / 768.0 * 192.0
            
            # Create the 4x4 intrinsic K matrix
            K = np.array([[fx, 0, u0, 0],
                        [0, fy, v0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)

            inv_K = np.linalg.pinv(K)
            K = torch.tensor(K[np.newaxis, ...]).cuda()
            inv_K = torch.tensor(inv_K[np.newaxis, ...]).cuda()

            complete_flow = predictions["complete_flow"].cpu().numpy()
            motion_mask = predictions["motion_mask"].cpu().numpy()
            motion_mask = motion_mask * 255.
            motion_mask = motion_mask[..., np.newaxis].repeat(3, -1)
            im = Image.fromarray(np.uint8(motion_mask[0, 0]))
            mask_prediction = {"mask_inference": im}

            complete_flow = predictions['complete_flow'].view(1, 3, -1)
            cam_points = self.backproject_depth(depth, inv_K)
            t = predictions['cam_T_cam']
            sample_ego, ego_flow = self.project_3d(cam_points, K, t)
            residual_flow = complete_flow - ego_flow
            independ_flow = residual_flow * predictions["motion_mask"].view(1, 1, -1)
            independ_flow = independ_flow.reshape(1, 3, 192, 512)
            _, ego_hsv, ego_mag = self.vis_motion(depth, K, inv_K, motion_map=None, camTcam=t, scale=0)
            _, ind_hsv, ind_mag = self.vis_motion(depth, K, inv_K, motion_map=independ_flow, camTcam=None, scale=0)
            _, tot_hsv, tot_mag = self.vis_motion(depth, K, inv_K, motion_map=independ_flow, camTcam=t, scale=0)
            max_mag = max(ego_mag, ind_mag, tot_mag)
            ind_hsv[:, 2] = torch.clamp(ind_hsv[:, 2] * ind_mag / max_mag, 0, 1)
            tot_hsv[:, 2] = torch.clamp(tot_hsv[:, 2] * tot_mag / max_mag, 0, 1)
            ind_flow = 1 - hsv_to_rgb(ind_hsv)
            tot_flow = 1 - hsv_to_rgb(tot_hsv)
            ind_flow = ind_flow[0].detach().cpu().permute(1, 2, 0).numpy()
            tot_flow = tot_flow[0].detach().cpu().permute(1, 2, 0).numpy()
            ind_flow = np.clip(ind_flow * 255, 0, 255)
            tot_flow = np.clip(tot_flow * 255, 0, 255)
            im = Image.fromarray(np.uint8(ind_flow))
            ind_flow_prediction = {'ind_flow_inference': im}
            im = Image.fromarray(np.uint8(tot_flow))
            tot_flow_prediction = {'tot_flow_inference': im}

            inputs = {"left_image": image, "height": height, "width": width, "task": task, 'type': 'segmentation'}
            predictions = self.model([inputs])[0]
            predictions = {**predictions, **depth_prediction, **ind_flow_prediction, **mask_prediction, **tot_flow_prediction}
            return predictions


    def vis_motion(self, depth, K, inv_K, motion_map=None, camTcam=None, scale=0):
            """ Compute optical flow map based on the input motion map and/or egomotion (camTcam)
                Projection via K and inv_K is used along with the depth predictions
            """

            assert motion_map != None or camTcam != None, 'At least one form of motion is supplied'
            b, _, h, w = depth.shape
            pix_ind_map = make_ind_map(h, w).cuda()

            # === obtain pix_motion_err from K and K_inv === #
            cam_points = self.backproject_depth(depth, inv_K)        # (B, 4, H*W)
            pix_coords, _ = self.project_3d(cam_points, K, T=None)
            pix_motion_err = pix_coords - pix_ind_map                       # this should be zero - used for error correction

            # === compute raw optical flow === #
            cam_points = self.backproject_depth(depth, inv_K)        # (B, 4, H*W)
            if motion_map != None:
                b, _, h, w = motion_map.shape
                cam_points[:, :3, :] += motion_map.reshape(b, 3, h * w)
            pix_coords, _ = self.project_3d(cam_points, K, camTcam)     # (B, H, W, 2)
            pix_motion_raw = pix_coords - pix_ind_map - pix_motion_err

            # === visualize optical flow === #
            mag, theta = cart2polar(pix_motion_raw)
            max_mag = (mag.max().item() + 1e-8)
            hsv = torch.ones(b, 3, h, w).cuda()
            hsv[:, 0] = (theta - torch.pi / 4) % (2 * torch.pi) / (2 * torch.pi)
            hsv[:, 1] = 1.0
            hsv[:, 2] = mag / max_mag
            motion_visual = 1 - hsv_to_rgb(hsv)  # (B, 3, H, W)

            return motion_visual, hsv, max_mag

def interp(x, shape, mode='bilinear', align_corners=False):
    """ Image tensor interpolation of x with shape (B, C, H, W) -> (B, C, *shape)
    """
    return torch.nn.functional.interpolate(x, shape, mode=mode, align_corners=align_corners)


def make_ind_map(height, width):
    """ Create identity indices map of shape (1,H,W,2) where top left corner is [-1,-1] and bottom right corner is [1,1]
    """
    v_strip = torch.arange(0, height) / height * 2 - 1
    h_strip = torch.arange(0, width) / width * 2 - 1

    return torch.stack([h_strip.unsqueeze(0).repeat(height, 1), v_strip.unsqueeze(1).repeat(1, width),]).permute(1, 2, 0).unsqueeze(0)


def cart2polar(cart):
    """ Convert cartian points into polar coordinates, the last dimension of the input must contain y and x component
    """
    assert cart.shape[-1] == 2, 'Last dimension must contain y and x vector component'

    r = torch.sqrt(torch.sum(cart**2, -1))
    theta = torch.atan(cart[..., 0] / cart[..., 1])
    theta[torch.where(torch.isnan(theta))] = 0  # torch.atan(0/0) gives nan

    theta[cart[..., 1] < 0] += torch.pi
    theta = (5 * torch.pi / 2 - theta) % (2 * torch.pi)

    return r, theta

def hsv_to_rgb(image):
    """ Convert image from hsv to rgb color space, input must be torch.Tensor of shape (*, 3, H, W)
    """
    assert isinstance(image, torch.Tensor), f"Input type is not a torch.Tensor. Got {type(image)}"
    assert len(
        image.shape) >= 3 and image.shape[-3] == 3, f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"

    h = image[..., 0, :, :]
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()  # turns very negative for nan
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out