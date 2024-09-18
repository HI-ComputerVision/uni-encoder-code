import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from matplotlib import cm
from PIL import Image

from model.utils.misc import is_dist_avail_and_initialized


class GroundPlane(nn.Module):
    def __init__(self, num_points_per_it=5, max_it=25, tol=0.1, g_prior=0.5, vertical_axis=1):
        super(GroundPlane, self).__init__()
        self.num_points_per_it = num_points_per_it
        self.max_it = max_it
        self.tol = tol
        self.g_prior = g_prior
        self.vertical_axis = vertical_axis

    def forward(self, points):
        """ estiamtes plane parameters and return each points distance to it
        :param points     (B, 3, H, W)
        :ret distance     (B, 1, H, W)
        :ret plane_param  (B, 3)
        """

        B, _, H, W = points.shape
        ground_points = points[:, :, -int(self.g_prior * H):, :]
        ground_points_inp = ground_points.reshape(B, 3, -1).permute(0, 2, 1)  # (B, N, 3)

        plane_param = self.estimate_ground_plane(ground_points_inp)

        all_points = points.reshape(B, 3, H * W).permute(0, 2, 1)  # (B, H*W, 3)
        dist2plane = self.dist_from_plane(all_points, plane_param).permute(0, 2, 1).reshape(B, 1, H, W)

        return dist2plane.detach(), plane_param.detach()

    def dist_from_plane(self, points, param):
        """ get vertical distance of each point from plane specified by param
        :param points   (B, 3) or (SB, B, 3)
        :param param    (3, 1) or (SB, 3, 1)
        :ret distance   (B, 1) or (SB, B, 1)
        """

        A, B = self.get_AB(points)
        return A @ param - B

    def estimate_ground_plane(self, points):
        """
        :param points           (B, N, 3)            
        :ret plane parameter    (B, 3) (B)
        """

        B, N, _ = points.shape
        T = self.num_points_per_it * self.max_it

        rand_points = []

        for b in range(B):
            rand_ind = np.random.choice(np.arange(N), T, replace=True)
            rand_points.append(points[b][rand_ind])
        rand_points = torch.stack(rand_points)  # (B, T, 3)

        ws = self.calc_param(rand_points).reshape(-1, 3, 1)    # (B*self.max_it, 3, 1)
        ps = points.repeat(self.max_it, 1, 1)                    # (B*self.max_it, N, 3)

        abs_dist = torch.abs(self.dist_from_plane(ps, ws)).reshape(B, self.max_it, N)

        param_fit = (abs_dist < self.tol).float().mean(2)
        best_fit = param_fit.argmax(1)
        best_w = ws.reshape(B, self.max_it, 3, 1)[np.arange(B), best_fit]

        return best_w

    def calc_param(self, points):
        """
        :param points           (B, self.max_it, self.num_points_per_it, 3)            
        :ret plane parameter    (B, self.max_it, 3)
        """

        batched_points = points.reshape(-1, self.num_points_per_it, 3)

        A, B = self.get_AB(batched_points)
        At = A.transpose(2, 1)  # batched transpose

        w = (torch.inverse(At @ A + 1e-6) @ At @ B).reshape(points.size(0), self.max_it, 3, 1)

        return w

    def get_AB(self, points):
        """ get mat A and B associated with points
        :param points   (B, 3) or (SB, B, 3)
        :ret A    (B, 3) or (SB, B, 3)
        :ret B    (B, 1) or (SB, B, 1)
        """
        B = points[..., self.vertical_axis:self.vertical_axis + 1]
        A = torch.cat([points[..., i:i + 1] for i in range(3) if i != self.vertical_axis] + [torch.ones_like(B)], -1)
        return A, B


def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth=0.1, max_depth=100.0):
    """Inverse of the previous function
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)
    return disp


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


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(
        device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(
            range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # P = torch.matmul(K, T)[:, :3, :]
        cam_points_3D = torch.matmul(T, points) if T is not None else points
        cam_points = torch.matmul(K[:, :3, :], cam_points_3D)

        # pinhole model - normalize by plane in front, not a spherical ball
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(-1, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        ego_motion = cam_points_3D[:, :3] - points[:, :3]

        return pix_coords, ego_motion


def compute_smooth_loss(inp, img=None):
    """ Computes the smoothness loss for an arbitrary tensor of size [B, C, H, W]
        The color image is used for edge-aware smoothness
    """

    grad_inp_x = torch.abs(inp[:, :, :, :-1] - inp[:, :, :, 1:])
    grad_inp_y = torch.abs(inp[:, :, :-1, :] - inp[:, :, 1:, :])

    if img is not None:
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_inp_x *= torch.exp(-grad_img_x)
        grad_inp_y *= torch.exp(-grad_img_y)

    return grad_inp_x.mean() + grad_inp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


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


class MonodepthLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(MonodepthLoss, self).__init__()
        # TODO: remove hard-coded values
        self.backproject_depth = []
        self.project_3d = []
        self.prob_target = {}

        bs = cfg.SOLVER.IMS_PER_BATCH // len(cfg.DATASETS.TRAIN)
        for scale in range(4):
            h, w = cfg.INPUT.DEPTH_CROP.SIZE
            h, w = h // (2 ** scale), w // (2 ** scale)
            self.backproject_depth.append(BackprojectDepth(bs, h, w).cuda())
            self.project_3d.append(Project3D(bs, h, w).cuda())
            self.prob_target[scale] = torch.zeros(bs, 1, h, w).cuda()
        self.ssim = SSIM().to(cfg.MODEL.DEVICE)
        self.gp_prior = 0.4
        self.gp_tol = 0.005
        self.gp_max_it = 100
        self.gp_num_points_per_it = 5
        self.gplane = GroundPlane(num_points_per_it=self.gp_num_points_per_it,
                                  max_it=self.gp_max_it, tol=self.gp_tol, g_prior=self.gp_prior)
        self.mask_disp_thrd = 0.03
        self.bce = nn.BCEWithLogitsLoss()
        self.pixel_mean = np.array([0., 0., 0.])
        self.pixel_std = np.array([255., 255., 255.])
        self.device = cfg.MODEL.DEVICE
        self.frame_ids = [-1, 1]
        self.count = 0

    def generate_images_pred(self, outputs, targets):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in range(4):
            source_scale = 0

            _, H, W = targets[0][('color', 0, 0)].shape
            B, _, h, w = outputs[('disp', 0, scale)].shape
            disp = outputs[('disp', 0, scale)]
            assert h * 2**(scale) == H and w * 2**(scale) == W
            disp = interp(disp, (H, W))
            disp_scaled, depth = disp_to_depth(disp)
            if (not is_dist_avail_and_initialized() or dist.get_rank() == 0) and scale == 0:
                print(f"Scale {scale} Min disp: {disp.detach().cpu().min():.2f}, max disp: {disp.detach().cpu().max():.2f}")
                print(f"Scale {scale} Min depth: {depth.min():.2f}, max depth: {depth.max():.2f}\n")
            outputs[("depth", 0, scale)] = depth
            outputs[("disp_scaled", 0, scale)] = disp_scaled

            to_plot = {}
            if scale == 0:
                self.count += 1
            for i, frame_id in enumerate(self.frame_ids):

                if frame_id == "s":
                    T = torch.stack([target['stereo_T'] for target in targets])
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                K = torch.stack([target['K'] for target in targets])
                inv_K = torch.stack([target['inv_K'] for target in targets])

                cam_points = self.backproject_depth[source_scale](depth, inv_K)
                outputs[('cam_points', 0, scale)] = cam_points

                # === Compute motion mask === #
                if self.bool_MotMask:
                    outputs[('motion_mask_r', frame_id, scale)] = interp(
                        outputs[('motion_mask', frame_id, scale)], (H, W))   # resize to original shape
                else:
                    outputs[('motion_mask', frame_id, scale)] = torch.ones(B, 1, h, w).to(self.device)
                    outputs[('motion_mask_r', frame_id, scale)] = torch.ones(B, 1, H, W).to(self.device)

                # === Compute sample for reconstruction === #
                if self.bool_CmpFlow:  # complete 3D flow is predicted
                    # compute 3D flows
                    sample_ego, ego_flow = self.project_3d[source_scale](cam_points, K, T)  # (B, H, W, 2), (B, 3, H*W)
                    complete_flow = interp(outputs[('complete_flow', frame_id, scale)], (H, W)).view(B, 3, -1)
                    residual_flow = complete_flow - ego_flow
                    independ_flow = residual_flow * outputs[('motion_mask_r', frame_id, scale)].view(B, 1, -1)

                    # compute 2D samples - detached since they are only used for motion mask supervision
                    outputs[('sample_ego', frame_id, scale)] = sample_ego.detach()
                    cam_points_tmp = cam_points.detach().clone()
                    cam_points_tmp[:, :3] += complete_flow
                    sample_complete, _ = self.project_3d[source_scale](cam_points_tmp, K, T=None)
                    outputs[('sample_complete', frame_id, scale)] = sample_complete.detach()

                    if self.bool_MotMask:
                        # Project into 3D via inv_K again - second pass for flow calculation
                        # only add the contribution of independent flow, transformation is applied afterwards
                        cam_points = self.backproject_depth[source_scale](depth, inv_K)
                        cam_points[:, :3] += independ_flow
                        sample, _ = self.project_3d[source_scale](cam_points, K, T)
                    else:
                        # since only learning the complete flow, it is added without using any transformation
                        cam_points[:, :3] += complete_flow
                        # (B, H, W, 2), (B, 1, H, W), (B, 3, H*W)
                        sample, _ = self.project_3d[source_scale](cam_points, K, T=None)

                else:                           # complete 3D flow is not predicted
                    # compute 3D flows
                    sample, ego_flow = self.project_3d[source_scale](
                        cam_points, K, T)  # (B, H, W, 2), (B, 1, H, W), (B, 3, H*W)
                    residual_flow = torch.zeros_like(ego_flow)
                    independ_flow = torch.zeros_like(ego_flow)
                    # Since complete 3D flow is not predicted, 2D samples sample_ego and sample_complete are the same, hence not recorded

                outputs[('sample', frame_id, scale)] = sample
                outputs[('color', frame_id, scale)] = F.grid_sample(torch.stack([
                    target[('color', frame_id, source_scale)] for target in targets
                ]), sample, padding_mode='border', align_corners=True)
                outputs[('ego_flow', frame_id, scale)] = ego_flow
                outputs[('independ_flow', frame_id, scale)] = independ_flow.reshape(B, 3, H, W)
                outputs[('residual_flow', frame_id, scale)] = interp(
                    residual_flow.reshape(B, 3, H, W), (h, w))

                if self.bool_automask:
                    outputs[('color_identity', frame_id, scale)] = torch.stack([
                        target[('color', frame_id, source_scale)] for target in targets
                    ])

                if (not is_dist_avail_and_initialized() or dist.get_rank() == 0) and scale == 0 and self.count % 20 == 0:
                    if to_plot.get('input', None) is None:
                        to_plot['input'] = targets[0][('color', 0, 0)].detach().cpu().permute(1, 2, 0).numpy()
                        to_plot['input'] = to_plot['input'] * self.pixel_std + self.pixel_mean
                        to_plot['input'] = np.clip(to_plot['input'], 0, 255).astype(np.uint8)
                        h, w, _ = to_plot['input'].shape
                        to_plot['input_crop'] = to_plot['input'][int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                        to_plot['input_crop'] = cv2.resize(to_plot['input_crop'], to_plot['input'].shape[1::-1])

                    estimated_input = outputs[("color", frame_id, scale)].detach().cpu()[0].permute(1, 2, 0).numpy()
                    estimated_input = estimated_input * self.pixel_std + self.pixel_mean
                    estimated_input = np.clip(estimated_input, 0, 255).astype(np.uint8)
                    estimated_input_crop = estimated_input[int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                    estimated_input_crop = cv2.resize(estimated_input_crop, estimated_input.shape[1::-1])

                    to_plot[f'estimated_input_{frame_id}'] = estimated_input
                    to_plot[f'estimated_input_crop_{frame_id}'] = estimated_input_crop

                    to_plot['disp'] = disp.detach().cpu()[0, 0].numpy()
                    _, to_plot['depth'] = disp_to_depth(to_plot['disp'])
                    vmax = np.percentile(to_plot['disp'], 95)
                    to_plot['disp'] = to_plot['disp'] / vmax
                    to_plot['disp'] = np.clip(to_plot['disp'], 0, 1)
                    to_plot['disp'] = cm.magma(to_plot['disp'])[..., :3] * 255
                    to_plot['disp_crop'] = to_plot['disp'][int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                    to_plot['disp_crop'] = cv2.resize(to_plot['disp_crop'], to_plot['disp'].shape[1::-1])

                    to_plot['motion_mask'] = outputs[('motion_mask', -1, 0)].detach().cpu()[0, 0].numpy()
                    to_plot['motion_mask'] = np.clip(to_plot['motion_mask'], 0, 1)
                    to_plot['motion_mask'] = (to_plot['motion_mask'] * 255.).astype(np.uint8)
                    to_plot['motion_mask'] = to_plot['motion_mask'][..., np.newaxis].repeat(3, -1)
                    to_plot['motion_mask_crop'] = to_plot['motion_mask'][int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                    to_plot['motion_mask_crop'] = cv2.resize(
                        to_plot['motion_mask_crop'], to_plot['motion_mask'].shape[1::-1])

                    motion = outputs[('independ_flow', -1, scale)]
                    K, inv_K, camTcam = K, inv_K, outputs[('cam_T_cam', 0, -1)]
                    _, ego_hsv, ego_mag = self.vis_motion(depth, K, inv_K, motion_map=None, camTcam=camTcam, scale=0)
                    _, ind_hsv, ind_mag = self.vis_motion(depth, K, inv_K, motion_map=motion, camTcam=None, scale=0)
                    _, tot_hsv, tot_mag = self.vis_motion(depth, K, inv_K, motion_map=motion, camTcam=camTcam, scale=0)
                    # _, tot_teacher_hsv, tot_teacher_mag = self.vis_motion(depth, K, inv_K, motion_map=outputs[('complete_flow_teacher', -1, 0)], camTcam=None, scale=0)
                    max_mag = max(ind_mag, ego_mag, tot_mag)

                    ego_hsv[:, 2] = torch.clamp(ego_hsv[:, 2] * ego_mag / max_mag, 0, 1)
                    ind_hsv[:, 2] = torch.clamp(ind_hsv[:, 2] * ind_mag / max_mag, 0, 1)
                    tot_hsv[:, 2] = torch.clamp(tot_hsv[:, 2] * tot_mag / max_mag, 0, 1)
                    # tot_teacher_hsv[:, 2] = torch.clamp(tot_teacher_hsv[:, 2] * tot_teacher_mag / max_mag, 0, 1)

                    ego_flow = 1 - hsv_to_rgb(ego_hsv)
                    ind_flow = 1 - hsv_to_rgb(ind_hsv)
                    tot_flow = 1 - hsv_to_rgb(tot_hsv)
                    # tot_teacher_flow = 1 - hsv_to_rgb(tot_teacher_hsv)

                    to_plot['ego_flow'] = ego_flow[0].detach().cpu().permute(1, 2, 0).numpy()
                    to_plot['ind_flow'] = ind_flow[0].detach().cpu().permute(1, 2, 0).numpy()
                    to_plot['tot_flow'] = tot_flow[0].detach().cpu().permute(1, 2, 0).numpy()
                    # to_plot['tot_teacher_flow'] = tot_teacher_flow[0].detach().cpu().permute(1, 2, 0).numpy()
                    to_plot['ego_flow'] = np.clip(to_plot['ego_flow'] * 255, 0, 255).astype(np.uint8)
                    to_plot['ind_flow'] = np.clip(to_plot['ind_flow'] * 255, 0, 255).astype(np.uint8)
                    to_plot['tot_flow'] = np.clip(to_plot['tot_flow'] * 255, 0, 255).astype(np.uint8)
                    # to_plot['tot_teacher_flow'] = np.clip(to_plot['tot_teacher_flow'] * 255, 0, 255).astype(np.uint8)
                    # to_plot['mask_teacher'] = outputs[('motion_mask_teacher', 1, 0)][0].detach().repeat(3, 1, 1).cpu().permute(1, 2, 0).numpy()
                    # to_plot['mask_teacher'] = np.clip(to_plot['mask_teacher'] * 255, 0, 255).astype(np.uint8)
                    to_plot['ego_flow_crop'] = to_plot['ego_flow'][int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                    to_plot['ind_flow_crop'] = to_plot['ind_flow'][int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                    to_plot['tot_flow_crop'] = to_plot['tot_flow'][int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                    # to_plot['tot_teacher_flow_crop'] = to_plot['tot_teacher_flow'][int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                    # to_plot['mask_teacher_crop'] = to_plot['mask_teacher'][int(h * 0.25):int(h * 0.75), int(w * 0.5):]
                    # to_plot['mask_teacher_crop'] = cv2.resize(
                    #     to_plot['mask_teacher_crop'], to_plot['mask_teacher'].shape[1::-1])
                    to_plot['ego_flow_crop'] = cv2.resize(to_plot['ego_flow_crop'], to_plot['ego_flow'].shape[1::-1])
                    to_plot['ind_flow_crop'] = cv2.resize(to_plot['ind_flow_crop'], to_plot['ind_flow'].shape[1::-1])
                    to_plot['tot_flow_crop'] = cv2.resize(to_plot['tot_flow_crop'], to_plot['tot_flow'].shape[1::-1])
                    # to_plot['tot_teacher_flow_crop'] = cv2.resize(to_plot['tot_teacher_flow_crop'], to_plot['tot_flow'].shape[1::-1])

            if len(to_plot) == 0:
                continue
            all_imgs_orig = np.concatenate((
                to_plot['input'],
                *[to_plot[f'estimated_input_{frame_id}'] for frame_id in self.frame_ids],
                to_plot['disp'],
                to_plot['motion_mask'],
                to_plot['ego_flow'],
                to_plot['ind_flow'],
                to_plot['tot_flow'],
                # to_plot['mask_teacher'],
                # to_plot['tot_teacher_flow'],
            ), axis=0)
            all_imgs_crop = np.concatenate((
                to_plot['input_crop'],
                *[to_plot[f'estimated_input_crop_{frame_id}'] for frame_id in self.frame_ids],
                to_plot['disp_crop'],
                to_plot['motion_mask_crop'],
                to_plot['ego_flow_crop'],
                to_plot['ind_flow_crop'],
                to_plot['tot_flow_crop'],
                # to_plot['mask_teacher_crop'],
                # to_plot['tot_teacher_flow_crop'],
            ), axis=0)
            all_imgs = np.concatenate((all_imgs_orig, all_imgs_crop), axis=1)
            im = Image.fromarray(np.uint8(all_imgs))
            im.save('reprojected.png')

    def vis_motion(self, depth, K, inv_K, motion_map=None, camTcam=None, scale=0):
        """ Compute optical flow map based on the input motion map and/or egomotion (camTcam)
            Projection via K and inv_K is used along with the depth predictions
        """

        assert motion_map != None or camTcam != None, 'At least one form of motion is supplied'
        b, _, h, w = depth.shape
        pix_ind_map = make_ind_map(h, w).to(self.device)

        # === obtain pix_motion_err from K and K_inv === #
        cam_points = self.backproject_depth[scale](depth, inv_K)        # (B, 4, H*W)
        pix_coords, _ = self.project_3d[scale](cam_points, K, T=None)
        pix_motion_err = pix_coords - pix_ind_map                       # this should be zero - used for error correction

        # === compute raw optical flow === #
        cam_points = self.backproject_depth[scale](depth, inv_K)        # (B, 4, H*W)
        if motion_map != None:
            b, _, h, w = motion_map.shape
            cam_points[:, :3, :] += motion_map.reshape(b, 3, h * w)
        pix_coords, _ = self.project_3d[scale](cam_points, K, camTcam)     # (B, H, W, 2)
        pix_motion_raw = pix_coords - pix_ind_map - pix_motion_err

        # === visualize optical flow === #
        mag, theta = cart2polar(pix_motion_raw)
        max_mag = (mag.max().item() + 1e-8)
        hsv = torch.ones(b, 3, h, w).to(self.device)
        hsv[:, 0] = (theta - torch.pi / 4) % (2 * torch.pi) / (2 * torch.pi)
        hsv[:, 1] = 1.0
        hsv[:, 2] = mag / max_mag
        motion_visual = 1 - hsv_to_rgb(hsv)  # (B, 3, H, W)

        return motion_visual, hsv, max_mag

    def get_ground_depth(self, plane_param, inv_K, scale=0, outputs=None):
        """ Create a new disparity map that fills the holes indicated by the mask with the pixel below it
            :param plane_param  (B, 3, 1)
        """
        B, _, h, w = outputs[('disp', 0, scale)].shape
        cam_points_init = torch.matmul(inv_K[:, :3, :3], self.backproject_depth[scale].pix_coords[:B])  # (B, 3, H*W)

        w1, w2, w3 = plane_param[:, 0:1], plane_param[:, 1:2], plane_param[:, 2:3]
        vx, vy, vz = cam_points_init[:, 0:1], cam_points_init[:, 1:2], cam_points_init[:, 2:3]

        ground_depth = (w3 / (vy - vx * w1 - vz * w2)).reshape(B, 1, h, w)
        ground_depth[torch.logical_or(ground_depth < 0, ground_depth > 100)] = 100
        ground_disp = depth_to_disp(ground_depth)

        return ground_disp, ground_depth

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def process_ground(self, inputs, outputs, scale=0):
        """ Estimate and predict the ground plane given the disparity predictions
        """
        disp = outputs[('disp', 0, scale)]
        _, depth = disp_to_depth(disp)
        inv_K = torch.stack([input['inv_K'] for input in inputs])
        h, w = disp.shape[-2:]

        cam_points = self.backproject_depth[scale](depth, inv_K)
        plane_dist, plane_param = self.gplane(cam_points[:, :3].reshape(-1, 3, h, w))

        g_mask = (torch.abs(plane_dist) < self.gp_tol).float()
        plane_param4diff = plane_param.clone()
        plane_param4diff[:, 2] += self.gp_tol
        ground_disp, ground_depth = self.get_ground_depth(plane_param4diff, inv_K, scale, outputs)

        disp_diff = disp - ground_disp
        disp_diff[ground_depth == 100.] = 0

        return plane_dist, disp_diff, g_mask

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        source_scale = 0
        losses = {'loss': 0}
        loss_terms = [
            'p_photo', 'd_smooth', 'd_ground', 'c_smooth', 'c_consistency', 'm_sparsity', 'm_smooth'
        ]
        coefs = {
            'g_p_photo': 1.0,
            'g_d_smooth': 1e-3,
            'g_d_ground': 0.1,
            'g_c_smooth': 1e-3,
            'g_c_consistency': 5.0,
            'g_m_sparsity': 0.04,
            'g_m_smooth': 0.1,
        }
        weight_ramp = ['g_c_smooth', 'g_c_consistency', 'g_m_sparsity', 'g_m_smooth']

        for term in loss_terms + [0, 1, 2, 3]:
            losses[f'loss_term/{term}'] = 0

        for loss_term in loss_terms:
            coef_name = 'g_' + loss_term
            loss_val = coefs[coef_name]

            if coef_name in weight_ramp:
                loss_val *= np.clip(3 * self.step / 8_000, 0.0, 1.0) if self.phrage in ['mask init', 'finetune'] else np.clip(3 * self.step / 35_000, 0.0, 1.0)

            losses[f'loss_coef/{loss_term}'] = loss_val

        for scale in range(4):
            losses_ps = {k: 0 for k in loss_terms}
            # Photometric Loss
            reprojection_losses = []
            h, w = outputs[("disp", 0, scale)].shape[-2:]
            color = torch.stack([input[('color', 0, 0)] for input in inputs])
            color = F.interpolate(color, [h, w], mode="bilinear", align_corners=False)
            target = torch.stack([input[('color', 0, 0)] for input in inputs])

            for frame_id in self.frame_ids:
                pred = outputs[('color', frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if self.bool_automask:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids:
                    pred = outputs[('color_identity', frame_id, scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                # save both images, and do min all at once below
                identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)

            reprojection_loss = reprojection_losses

            if self.bool_automask:
                # add random numbers to remove ties
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape,
                                                          device=self.device) * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if self.bool_automask:
                outputs['identity_selection/{}'.format(scale)] = (idxs >
                                                                  identity_reprojection_loss.shape[1] - 1).float()

            losses_ps['p_photo'] = to_optimise.mean()

            # Disparity Regularization
            if self.move_Depth:
                if losses['loss_coef/d_smooth'] > 0:
                    disp = outputs[('disp', 0, scale)]
                    norm_disp = disp / (disp.mean(2, True).mean(3, True) + 1e-7)
                    losses_ps['d_smooth'] = compute_smooth_loss(norm_disp, color) / (2 ** scale)

                if losses['loss_coef/d_ground'] > 0 and self.bool_MotMask:
                    plane_dist, disp_diff, g_mask = self.process_ground(inputs, outputs, scale=scale)
                    disp_diff[disp_diff > 0] = 0
                    losses_ps['d_ground'] = -1 * torch.mean(disp_diff) / (2 ** scale)  # below ground is negative

            # Motion Regularization

            num_frames = len(self.frame_ids)
            for frame_id in self.frame_ids:

                # (B, 1, h, w)                     # (B, 1, h, w)
                disp = outputs[('disp', 0, scale)]
                motion_mask = outputs[('motion_mask', frame_id, scale)]     # (B, 1, h, w)
                h, w = motion_mask.shape[-2:]

                if self.move_CmpFlow and self.bool_CmpFlow:
                    complete_flow = outputs[('complete_flow', frame_id, scale)]     # (B, 3, h, w)
                    residual_flow = outputs[('residual_flow', frame_id, scale)]     # (B, 3, h, w)

                    if losses['loss_coef/c_smooth'] > 0:
                        losses_ps['c_smooth'] += compute_smooth_loss(complete_flow, color) / (2 ** scale) / num_frames

                    # consistency can only be computed when the motion mask is predicted as well
                    if self.bool_MotMask and losses['loss_coef/c_consistency'] > 0:
                        valid_disp = (disp > self.mask_disp_thrd).detach()  # avoid rotational edge cases
                        losses_ps['c_consistency'] += torch.mean(valid_disp * (1 - motion_mask.detach())
                                                                 * torch.abs(residual_flow)) / (2 ** scale) / num_frames

                if self.move_MotMask and self.bool_MotMask:
                    sample_ego = outputs[('sample_ego', frame_id, scale)]               # (B, H, W, 2)
                    sample_complete = outputs[('sample_complete', frame_id, scale)]     # (B, H, W, 2)
                    motion_prob = outputs[('motion_prob', frame_id, scale)]             # (B, 1, h, w)

                    if losses['loss_coef/m_sparsity'] > 0:
                        sample_ego = interp(sample_ego.permute(0, 3, 1, 2), (h, w))             # (B, 2, h, w)
                        sample_complete = interp(sample_complete.permute(0, 3, 1, 2), (h, w))   # (B, 2, h, w)
                        disp_mag = torch.sum((sample_ego - sample_complete) ** 2, 1)            # (B, h, w)
                        static = (disp_mag < disp_mag.mean()).unsqueeze(1)                      # (B, 1, h, w)
                        if torch.all(torch.sum(static, (1, 2, 3)) > 0):
                            losses_ps['m_sparsity'] += 3 * self.bce(motion_prob[static],
                                                                self.prob_target[scale][static]) / (2 ** scale) / num_frames

                    if losses['loss_coef/m_smooth'] > 0:
                        losses_ps['m_smooth'] += compute_smooth_loss(
                            motion_mask, color) / (2 ** scale) / num_frames

            # Compile Losses
            for loss_term in loss_terms:
                losses[f'loss_term/{scale}'] += losses_ps[loss_term] * losses[f'loss_coef/{loss_term}']
                losses[f'loss_term/{loss_term}'] += losses_ps[loss_term]

            losses[f'loss'] += losses[f'loss_term/{scale}'] / 4

        return losses

    def forward(self, outputs, targets, *args):
        self.generate_images_pred(outputs, targets)
        losses = self.compute_losses(targets, outputs)
        return {'loss_monodepth': losses['loss']}