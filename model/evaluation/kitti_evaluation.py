# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/cityscapes_evaluation.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import glob
import json
import logging
import cv2
from matplotlib import cm
import numpy as np
import os
import tempfile
import skimage
import torch.nn.functional as F
from collections import OrderedDict
import torch
from PIL import Image
from collections import Counter

from detectron2.utils.events import get_event_storage
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator
from ..modeling.monodepth_loss import disp_to_depth


class KITTIEvaluator(DatasetEvaluator):
    """
    Base class for evaluation using KITTI API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(
            prefix="KITTI_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        assert (
            comm.get_local_size() == comm.get_world_size()
        ), "KittiEvaluator currently do not work with multiple machines."
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing KITTI results to temporary directory {} ...".format(
                self._temp_dir)
        )


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


class KITTIDepthEvaluator(KITTIEvaluator):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def load_velodyne_points(self, filename):
        """Load 3D point cloud from KITTI file format
        (adapted from https://github.com/hunse/kitti)
        """
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # homogeneous
        return points

    def read_calib_file(self, path):
        """Read KITTI calibration file
        (from https://github.com/hunse/kitti)
        """
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass

        return data

    def sub2ind(self, matrixSize, rowSub, colSub):
        """Convert row, col matrix subscripts to linear indices
        """
        m, n = matrixSize
        return rowSub * (n - 1) + colSub - 1

    def generate_depth_map(self, calib_dir, velo_filename, cam=2, vel_depth=False):
        """Generate a depth map from velodyne data
        """
        # load calibration files
        cam2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # get image shape
        im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        # load velodyne points and remove all behind image plane (approximation)
        # each row of the velodyne data is forward, left, up, reflectance
        velo = self.load_velodyne_points(velo_filename)
        velo = velo[velo[:, 0] >= 0, :]

        # project the points to the camera
        velo_pts_im = np.dot(P_velo2im, velo.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

        if vel_depth:
            velo_pts_im[:, 2] = velo[:, 0]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros((im_shape[:2]))
        depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = self.sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        return depth

    def process(self, inputs, outputs):
        try:
            storage = get_event_storage()
        except:
            storage = None
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            left_image = input['left_image']
            sequence_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_name))))
            basename = sequence_name + os.path.splitext(os.path.basename(file_name))[0]
            velo_file = input['velo_file']
            calib_path = input['calib_path']
            depth_gt = self.generate_depth_map(calib_path, velo_file, 2, True)

            np.save(os.path.join(self._temp_dir,
                    f"{basename}_depth_gt.npy"), depth_gt)

            disp_resized, _ = disp_to_depth(output['disp_results'])
            disp_resized = disp_resized.squeeze().cpu().detach().numpy()
            disp_resized = cv2.resize(disp_resized, depth_gt.shape[::-1])
            depth_pred = 1 / disp_resized
            np.save(os.path.join(self._temp_dir,
                    f"{basename}_depth_pred.npy"), depth_pred)

            if storage is not None:
                if len(storage._vis_data) == 0:
                    artifact_dir = './artifacts'
                    os.makedirs(artifact_dir, exist_ok=True)
                    path_left_depth = os.path.join(
                        artifact_dir, basename + "_depth.png")

                    left_image = left_image.cpu().detach().permute(
                        1, 2, 0).numpy().astype(np.uint8)
                    left_image = cv2.resize(left_image, disp_resized.shape[::-1])
                    vmax = np.percentile(disp_resized, 95)
                    disp_resized = disp_resized / vmax
                    disp_resized = np.clip(disp_resized, 0, 1)
                    disp_resized = cm.magma(disp_resized)[..., :3] * 255
                    im2save = np.vstack([
                        left_image,
                        disp_resized
                    ])
                    im = Image.fromarray(np.uint8(im2save))
                    im.save(path_left_depth)
                    storage._vis_data.append(
                        ("left_depth", path_left_depth, storage.iter))

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.

        self._logger.info(
            "Evaluating results under {} ...".format(self._temp_dir))

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        depth_gt_list = glob.glob(os.path.join(
            self._temp_dir, "*_depth_gt.npy"))

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = [], [], [], [], [], [], []
        for img in depth_gt_list:
            depth_gt = np.load(img)
            depth_pred = np.load(img.replace("_depth_gt.npy", "_depth_pred.npy"))

            gt_height, gt_width = depth_gt.shape[:2]
            mask = np.logical_and(depth_gt > self.MIN_DEPTH, depth_gt < self.MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            depth_pred = depth_pred[mask]
            depth_gt = depth_gt[mask]

            ratio = np.median(depth_gt) / np.median(depth_pred)
            depth_pred *= ratio
            print(
                f"Min depth: {np.min(depth_pred):.2f}, Max depth: {np.max(depth_pred):.2f}")
            depth_pred[depth_pred < self.MIN_DEPTH] = self.MIN_DEPTH
            depth_pred[depth_pred > self.MAX_DEPTH] = self.MAX_DEPTH

            _abs_rel, _sq_rel, _rmse, _rmse_log, _a1, _a2, _a3 = compute_errors(
                depth_gt, depth_pred)
            abs_rel.append(_abs_rel)
            sq_rel.append(_sq_rel)
            rmse.append(_rmse)
            rmse_log.append(_rmse_log)
            a1.append(_a1)
            a2.append(_a2)
            a3.append(_a3)

        abs_rel = np.mean(abs_rel)
        sq_rel = np.mean(sq_rel)
        rmse = np.mean(rmse)
        rmse_log = np.mean(rmse_log)
        a1 = np.mean(a1)
        a2 = np.mean(a2)
        a3 = np.mean(a3)

        ret = OrderedDict()
        ret["depth_error"] = {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "a1": a1,
            "a2": a2,
            "a3": a3,
        }
        self._working_dir.cleanup()
        return ret


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
