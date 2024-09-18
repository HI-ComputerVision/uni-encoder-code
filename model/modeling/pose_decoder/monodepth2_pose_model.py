import torch.distributed as dist
from torch import nn

from collections import OrderedDict
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from model.modeling.backbone.resnet import BasicBlock, BasicStem, BottleneckBlock, DeformBottleneckBlock, ResNet
from model.utils.misc import is_dist_avail_and_initialized


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=2, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(
            num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(
            256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = input_features["res5"]

        out = self.relu(self.convs["squeeze"](last_features))

        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]
        if not is_dist_avail_and_initialized() or dist.get_rank() == 0:
            print(translation[0, 0].detach().cpu().numpy())

        return axisangle, translation


def build_resnet_encoder(cfg, num_input_images=2):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.POSE_RESNETS.NORM
    stem = BasicStem(
        in_channels=num_input_images * 3,
        out_channels=cfg.MODEL.POSE_RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    out_features        = ["stem", "res2", "res3", "res4", "res5"]
    depth               = cfg.MODEL.POSE_RESNETS.DEPTH
    num_groups          = cfg.MODEL.POSE_RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.POSE_RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.POSE_RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.POSE_RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.POSE_RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.POSE_RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.POSE_RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.POSE_RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.POSE_RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.POSE_RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.POSE_RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.POSE_RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.POSE_RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, freeze_at=-1)