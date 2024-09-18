# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/cityscapes_evaluation.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import copy
import glob
import json
import pickle
import logging
import cv2
from matplotlib import cm
import numpy as np
import os
import tempfile
import torch.nn.functional as F
from collections import OrderedDict
import torch
from PIL import Image

from detectron2.utils.events import get_event_storage
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator
from ..modeling.monodepth_loss import disp_to_depth


class CityscapesEvaluator(DatasetEvaluator):
    """
    Base class for evaluation using cityscapes API.
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
            prefix="cityscapes_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        assert (
            comm.get_local_size() == comm.get_world_size()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(
                self._temp_dir)
        )


class CityscapesInstanceEvaluator(CityscapesEvaluator):
    """
    Evaluate instance segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")

            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.stuff_classes[pred_class]
                        class_id = name2label[classes].id
                        score = output.scores[i]
                        mask = output.pred_masks[i].numpy().astype("uint8")
                        png_filename = os.path.join(
                            self._temp_dir, basename +
                            "_{}_{}.png".format(i, classes)
                        )

                        Image.fromarray(mask * 255).save(png_filename)
                        fout.write(
                            "{} {} {}\n".format(os.path.basename(
                                png_filename), class_id, score)
                        )
            else:
                # Cityscapes requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info(
            "Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.pickleOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(
            self._temp_dir, "gtInstances.pickle")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(
            gt_dir, "*", "*_gtFine_instanceIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] *
                       100, "AP50": results["allAp50%"] * 100}
        self._working_dir.cleanup()
        return ret


class CityscapesSemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output = output["sem_seg"].argmax(
                dim=0).to(self._cpu_device).numpy()
            pred = 255 * np.ones(output.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info(
            "Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.pickleOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(
            gt_dir, "*", "*_gtFine_labelIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "iIoU": 100.0 * results["averageScoreInstClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
            "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        }
        self._working_dir.cleanup()
        return ret


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


class CityscapesDepthEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    def process(self, inputs, outputs):
        original_w, original_h = get_image_dimensions(inputs[0]["file_name"])
        # storage = get_event_storage()
        pred_disps, depth_gts = [], []
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            gt_path = file_name.replace('/leftImg8bit/test/', '/gt_depths/').replace('.png', '.npy')
            basename = os.path.splitext(os.path.basename(file_name))[0]
            left_image = input['left_image']

            pred_disp, _ = disp_to_depth(output['disp_results'])
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            np.save(os.path.join(self._temp_dir,
                    f"{basename}_pred_disp.npy"), pred_disp)
            pred_disps.append(pred_disp)
            depth_gt = np.load(gt_path)
            np.save(os.path.join(self._temp_dir,
                    f"{basename}_depth_gt.npy"), depth_gt)
            depth_gts.append(depth_gt)
            
        pred_disps = np.concatenate(pred_disps)
        depth_gts = np.concatenate(depth_gts)

        # for idx in range(len(pred_disps)):
        #     disp_resized = pred_disps[idx]

            # if len(storage._vis_data) == 0:
            #     artifact_dir = './artifacts'
            #     os.makedirs(artifact_dir, exist_ok=True)
            #     path_left_depth = os.path.join(
            #         artifact_dir, basename.replace("_leftImg8bit", "_left_depth.png"))

            #     left_image = inputs[idx]['left_image'].cpu().detach().permute(
            #         1, 2, 0).numpy().astype(np.uint8)
            #     left_image = cv2.resize(left_image, disp_resized.shape[::-1])
            #     vmax = np.percentile(disp_resized, 95)
            #     disp_resized = disp_resized / vmax
            #     disp_resized = np.clip(disp_resized, 0, 1)
            #     disp_resized = cm.magma(disp_resized)[..., :3] * 255
            #     im2save = np.vstack([
            #         left_image,
            #         disp_resized
            #     ])
            #     im = Image.fromarray(np.uint8(im2save))
            #     im.save(path_left_depth)
            #     storage._vis_data.append(
            #         ("left_depth", path_left_depth, storage.iter))

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
            disp_pred = np.load(img.replace("_depth_gt.npy", "_pred_disp.npy"))
            gt_height, gt_width = depth_gt.shape[:2]
            gt_height = int(round(gt_height * 0.75))
            depth_gt = depth_gt[:gt_height]

            disp_pred = cv2.resize(np.squeeze(disp_pred), (gt_width, gt_height))
            depth_pred = 1 / disp_pred

            depth_gt = depth_gt[256:, 192:1856]
            depth_pred = depth_pred[256:, 192:1856]

            mask = np.logical_and(depth_gt > self.MIN_DEPTH, depth_gt < self.MAX_DEPTH)

            depth_pred = depth_pred[mask]
            depth_gt = depth_gt[mask]

            ratio = np.median(depth_gt) / np.median(depth_pred)
            depth_pred *= ratio
            print(f"Min depth: {np.min(depth_pred):.2f}, Max depth: {np.max(depth_pred):.2f}")

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
