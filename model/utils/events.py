import os
import shutil
import zipfile
import yaml
import mlflow
import wandb
from pathlib import Path
from dotenv import dotenv_values
from detectron2.utils import comm
from detectron2.utils.events import EventWriter, get_event_storage
from mlflow import log_params, log_metrics, log_artifact


def set_environment_variables(path_env):
    """Set environment variables for AWS and MLFLOW.

    Args:
        path_env (str): Path to the .env file
    """
    config = dotenv_values(path_env)
    for key, value in config.items():
        os.environ[key] = value


def zip_folder(folder_path, output_path):
    """
    Zip the contents of an entire folder (with subfolders) into a zip file.
    Parameters:
    - folder_path: The path of the folder to zip.
    - output_path: The path of the output zip file.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # os.walk() generates the file names in a directory tree by walking either top-down or bottom-up.
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create the full path to the file
                file_path = os.path.join(root, file)
                # Add the file to the zip file
                # Arcname defines the name of the file in the archive. Path relative to the folder being archived is used.
                zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(folder_path)))


def setup_wandb(cfg, args):
    if comm.is_main_process():
        init_args = {
            k.lower(): v
            for k, v in cfg.WANDB.items()
            if isinstance(k, str) and k not in ["config"]
        }
        # only include most related part to avoid too big table
        # TODO: add configurable params to select which part of `cfg` should be saved in config
        if "config_exclude_keys" in init_args:
            init_args["config"] = cfg
            init_args["config"]["cfg_file"] = args.config_file
        else:
            init_args["config"] = {
                "model": cfg.MODEL,
                "solver": cfg.SOLVER,
                "cfg_file": args.config_file,
            }
        if ("name" not in init_args) or (init_args["name"] is None):
            init_args["name"] = os.path.basename(args.config_file)
        else:
            init_args["name"] = init_args["name"] + '_' + os.path.basename(args.config_file)
        wandb.init(**init_args)


def setup_mlflow(cfg, args):
    if comm.is_main_process():
        # Set MLflow server URI
        mlflow.set_tracking_uri(cfg.MLFLOW.get("TRACKING_URI", "http://localhost:5000"))  # Change the URI as needed

        # Set MLflow experiment/project name
        mlflow.set_experiment(cfg.MLFLOW.get("PROJECT", "default_project"))  # Change the project name as needed

        mlflow.start_run(run_name=cfg.MLFLOW.get("NAME", None))
        _d2_out_dir = cfg.OUTPUT_DIR

        config_path = os.path.join(_d2_out_dir, 'config.yaml')
        log_artifact(config_path, artifact_path=f"")

        _oneformer_dir = os.path.join(os.path.dirname(os.path.dirname(_d2_out_dir)), 'oneformer')
        _oneformer_zip_path = os.path.join(_d2_out_dir, 'oneformer.zip')
        zip_folder(_oneformer_dir, _oneformer_zip_path)
        log_artifact(_oneformer_zip_path, artifact_path='')


class BaseRule(object):
    def __call__(self, target):
        return target


class IsIn(BaseRule):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(self, target):
        return self.keyword in target


class Prefix(BaseRule):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(self, target):
        return "/".join([self.keyword, target])


class WandbWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self):
        """
        Args:
            log_dir (str): the directory to save the output events
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._last_write = -1
        self._group_rules = [
            (IsIn("/"), BaseRule()),
            (IsIn("loss"), Prefix("train")),
        ]

    def write(self):

        storage = get_event_storage()

        def _group_name(scalar_name):
            for (rule, op) in self._group_rules:
                if rule(scalar_name):
                    return op(scalar_name)
            return scalar_name

        stats = {
            _group_name(name): scalars[0]
            for name, scalars in storage.latest().items()
            if scalars[1] > self._last_write
        }
        if len(stats) > 0:
            self._last_write = max([v[1] for k, v in storage.latest().items()])

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            stats["image"] = [
                wandb.Image(img, caption=img_name)
                for img_name, img, step_num in storage._vis_data
            ]
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:

            def create_bar(tag, bucket_limits, bucket_counts, **kwargs):
                data = [
                    [label, val] for (label, val) in zip(bucket_limits, bucket_counts)
                ]
                table = wandb.Table(data=data, columns=["label", "value"])
                return wandb.plot.bar(table, "label", "value", title=tag)

            stats["hist"] = [create_bar(**params) for params in storage._histograms]

            storage.clear_histograms()

        if len(stats) == 0:
            return
        wandb.log(stats, step=storage.iter)

    def close(self):
        wandb.finish()


class MLflowWriter(EventWriter):
    """
    Write scalars and visualizations to MLflow.
    """

    def __init__(self, cfg, experiment_name="default"):
        """
        Args:
            experiment_name (str): the name of the MLflow experiment
        """
        self._d2_out_dir = cfg.OUTPUT_DIR
        self._checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self._last_write = -1
        self._experiment_name = experiment_name
        self._step_logged = False  # Track if "step" parameter has been logged

    def write(self):
        storage = get_event_storage()

        iter = storage.iter

        if (iter + 1) % self._checkpoint_period == 0:
            with open(os.path.join(self._d2_out_dir, 'last_checkpoint'), 'r') as f:
                model_name = f.read()

            old_model_path = os.path.join(self._d2_out_dir, f'{model_name}')
            new_model_path = os.path.join(self._d2_out_dir, 'model.pth')
            shutil.copy(old_model_path, new_model_path)
            log_artifact(new_model_path, artifact_path=f"models")

            old_checkpoint_info_path = os.path.join(self._d2_out_dir, 'last_checkpoint')
            new_checkpoint_info_path = os.path.join(self._d2_out_dir, 'last_checkpoint.txt')
            shutil.copy(old_checkpoint_info_path, new_checkpoint_info_path)
            log_artifact(new_checkpoint_info_path, artifact_path=f"models")

        metrics = {
            name: scalars[0]
            for name, scalars in storage.latest().items()
            if scalars[1] > self._last_write
        }

        if len(metrics) > 0:
            self._last_write = max([v[1] for k, v in storage.latest().items()])

            # Log experiment parameters
            if not self._step_logged:
                log_params({
                    "cfg_file": storage.iter,
                })
                self._step_logged = True

            # Log metrics
            log_metrics(metrics, step=storage.iter)

            # Log visualizations (if any)
            if len(storage._vis_data) >= 1:
                for img_name, img, step_num in storage._vis_data:
                    # Assuming img is a PIL Image or similar
                    log_artifact(img, artifact_path=f"images/{step_num:06}_{img_name}".replace("_depth", "_est"))

                # Clear visualization data
                storage.clear_images()

        if len(storage._histograms) >= 1:
            # Log histograms (if any)
            for params in storage._histograms:
                # Assuming you have histogram data in params, adjust as needed
                # log_histogram(params, step=storage.iter)
                pass

            # Clear histogram data
            storage.clear_histograms()

    def close(self):
        # Finish MLflow run
        mlflow.end_run()
