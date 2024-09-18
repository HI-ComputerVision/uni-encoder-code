# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

__all__ = ["add_common_config", "add_uni_encoder_config", "add_swin_config",
           "add_dinat_config", "add_convnext_config", "add_resnet_posenet_config"]


def add_common_config(cfg):
    """
    Add config for common configuration
    """

    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "oneformer_unified"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.INPUT.TASK_SEQ_LEN = 77
    cfg.INPUT.MAX_SEQ_LEN = 77

    cfg.INPUT.TASK_PROB = CN()
    cfg.INPUT.TASK_PROB.SEMANTIC = 0.33
    cfg.INPUT.TASK_PROB.INSTANCE = 0.66

    # test dataset
    cfg.DATASETS.SEG_TEST_PANOPTIC = ("",)
    cfg.DATASETS.SEG_TEST_INSTANCE = ("",)
    cfg.DATASETS.SEG_TEST_SEMANTIC = ("",)
    cfg.DATASETS.TRAIN = ("",)
    cfg.DATASETS.TRAIN = ("",)
    cfg.DATASETS.DEPTH_TEST = ("",)

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.DISP_INIT_ITER = 0
    cfg.SOLVER.MOTION_INIT_ITER = 10000
    cfg.SOLVER.MASK_INIT_ITER = 20000
    cfg.SOLVER.FINE_TUNE_ITER = 30000

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "OneFormer"
    cfg.WANDB.NAME = None

    # MLFLOW
    cfg.MLFLOW = CN()
    cfg.MLFLOW.PROJECT = "MonoDepthTinyOneFormer"
    cfg.MLFLOW.NAME = None
    cfg.MLFLOW.TRACKING_URI = "http://localhost:5000"

    cfg.MODEL.IS_TRAIN = True
    cfg.MODEL.IS_DEMO = False

    # text encoder config
    cfg.MODEL.TEXT_ENCODER = CN()

    cfg.MODEL.TEXT_ENCODER.WIDTH = 256
    cfg.MODEL.TEXT_ENCODER.CONTEXT_LENGTH = 77
    cfg.MODEL.TEXT_ENCODER.NUM_LAYERS = 12
    cfg.MODEL.TEXT_ENCODER.VOCAB_SIZE = 49408
    cfg.MODEL.TEXT_ENCODER.PROJ_NUM_LAYERS = 2
    cfg.MODEL.TEXT_ENCODER.N_CTX = 16

    # oneformer inference config
    cfg.MODEL.TEST = CN()
    cfg.MODEL.TEST.SEMANTIC_ON = True
    cfg.MODEL.TEST.INSTANCE_ON = False
    cfg.MODEL.TEST.PANOPTIC_ON = False
    cfg.MODEL.TEST.DEPTH_ON = False
    cfg.MODEL.TEST.DETECTION_ON = False
    cfg.MODEL.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.TEST.TASK = "panoptic"

    # TEST AUG Slide
    cfg.TEST.AUG.IS_SLIDE = False
    cfg.TEST.AUG.CROP_SIZE = (640, 640)
    cfg.TEST.AUG.STRIDE = (426, 426)
    cfg.TEST.AUG.SCALE = (2048, 640)
    cfg.TEST.AUG.SETR_MULTI_SCALE = True
    cfg.TEST.AUG.KEEP_RATIO = True
    cfg.TEST.AUG.SIZE_DIVISOR = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.DEPTH_DECODER_NAME = "BasePixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.SEM_EMBED_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.INST_EMBED_DIM = 256

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.SEG_MIN_SIZE_TRAIN = (192,)
    cfg.INPUT.SEG_MAX_SIZE_TRAIN = 512
    cfg.INPUT.SEG_MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.SEG_MIN_SIZE_TEST = 192
    cfg.INPUT.SEG_MAX_SIZE_TEST = 192
    cfg.INPUT.DEPTH_MIN_SIZE_TRAIN = (192,)
    cfg.INPUT.DEPTH_MAX_SIZE_TRAIN = 512
    cfg.INPUT.DEPTH_MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.DEPTH_MIN_SIZE_TEST = 192
    cfg.INPUT.DEPTH_MAX_SIZE_TEST = 192
    cfg.INPUT.SEG_CROP = CN()
    cfg.INPUT.SEG_CROP.ENABLED = False
    cfg.INPUT.SEG_CROP.TYPE = "absolute"
    cfg.INPUT.SEG_CROP.SIZE = (192, 512)
    cfg.INPUT.SEG_CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.DEPTH_CROP = CN()
    cfg.INPUT.DEPTH_CROP.ENABLED = False
    cfg.INPUT.DEPTH_CROP.TYPE = "absolute"
    cfg.INPUT.DEPTH_CROP.SIZE = (192, 512)
    cfg.INPUT.SEG_COLOR_AUG_SSD = False
    cfg.INPUT.DEPTH_COLOR_JITTER = False

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8


def add_uni_encoder_config(cfg):
    """
    Add config for ONE_FORMER.
    """

    # oneformer model config
    cfg.MODEL.ONE_FORMER = CN()

    # loss
    cfg.MODEL.ONE_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.ONE_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.ONE_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.ONE_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.ONE_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.ONE_FORMER.CONTRASTIVE_WEIGHT = 0.5
    cfg.MODEL.ONE_FORMER.MONODEPTH_WEIGHT = 2.0
    cfg.MODEL.ONE_FORMER.OPTICAL_FLOW_DISTIL_WEIGHT = 1.0
    cfg.MODEL.ONE_FORMER.CONTRASTIVE_TEMPERATURE = 0.07

    # transformer config
    cfg.MODEL.ONE_FORMER.NHEADS = 8
    cfg.MODEL.ONE_FORMER.DROPOUT = 0.1
    cfg.MODEL.ONE_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ONE_FORMER.ENC_LAYERS = 0
    cfg.MODEL.ONE_FORMER.CLASS_DEC_LAYERS = 2
    cfg.MODEL.ONE_FORMER.DEC_LAYERS = 6
    cfg.MODEL.ONE_FORMER.PRE_NORM = False

    cfg.MODEL.ONE_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES = 120
    cfg.MODEL.ONE_FORMER.NUM_OBJECT_CTX = 16
    cfg.MODEL.ONE_FORMER.USE_TASK_NORM = True

    cfg.MODEL.ONE_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.ONE_FORMER.ENFORCE_INPUT_PROJ = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.ONE_FORMER.SIZE_DIVISIBILITY = 32

    # transformer module
    cfg.MODEL.ONE_FORMER.TRANSFORMER_DECODER_NAME = "ContrastiveMultiScaleMaskedTransformerDecoder"

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.ONE_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.ONE_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.ONE_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_swin_config(cfg):
    """
    Add config forSWIN Backbone.
    """

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False


def add_dinat_config(cfg):
    """
    Add config for NAT Backbone.
    """

    # DINAT transformer backbone
    cfg.MODEL.DiNAT = CN()
    cfg.MODEL.DiNAT.DEPTHS = [3, 4, 18, 5]
    cfg.MODEL.DiNAT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.DiNAT.EMBED_DIM = 64
    cfg.MODEL.DiNAT.MLP_RATIO = 3.0
    cfg.MODEL.DiNAT.NUM_HEADS = [2, 4, 8, 16]
    cfg.MODEL.DiNAT.DROP_PATH_RATE = 0.2
    cfg.MODEL.DiNAT.KERNEL_SIZE = 7
    cfg.MODEL.DiNAT.DILATIONS = [[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]]
    cfg.MODEL.DiNAT.OUT_INDICES = (0, 1, 2, 3)
    cfg.MODEL.DiNAT.QKV_BIAS = True
    cfg.MODEL.DiNAT.QK_SCALE = None
    cfg.MODEL.DiNAT.DROP_RATE = 0
    cfg.MODEL.DiNAT.ATTN_DROP_RATE = 0.
    cfg.MODEL.DiNAT.IN_PATCH_SIZE = 4


def add_convnext_config(cfg):
    """
    Add config for ConvNeXt Backbone.
    """

    # swin transformer backbone
    cfg.MODEL.CONVNEXT = CN()
    cfg.MODEL.CONVNEXT.IN_CHANNELS = 3
    cfg.MODEL.CONVNEXT.DEPTHS = [3, 3, 27, 3]
    cfg.MODEL.CONVNEXT.DIMS = [192, 384, 768, 1536]
    cfg.MODEL.CONVNEXT.DROP_PATH_RATE = 0.4
    cfg.MODEL.CONVNEXT.LSIT = 1.0
    cfg.MODEL.CONVNEXT.OUT_INDICES = [0, 1, 2, 3]
    cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]


def add_resnet_posenet_config(cfg):
    """
    Add config for Additional PoseNet.
    """

    # swin transformer backbone
    cfg.MODEL.POSE_RESNETS = CN()
    cfg.MODEL.POSE_RESNETS.NORM = "SyncBN"
    cfg.MODEL.POSE_RESNETS.STEM_OUT_CHANNELS = 64
    cfg.MODEL.POSE_RESNETS.OUT_FEATURES = ["res5"]
    cfg.MODEL.POSE_RESNETS.DEPTH = 18
    cfg.MODEL.POSE_RESNETS.NUM_GROUPS = 1
    cfg.MODEL.POSE_RESNETS.WIDTH_PER_GROUP = 64
    cfg.MODEL.POSE_RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.POSE_RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.POSE_RESNETS.RES5_DILATION = 1
    cfg.MODEL.POSE_RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    cfg.MODEL.POSE_RESNETS.DEFORM_MODULATED = False
    cfg.MODEL.POSE_RESNETS.DEFORM_NUM_GROUPS = 1