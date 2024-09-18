import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.layers import ShapeSpec


def _make_1x1_convs(in_shape, out_shape, groups=1, expand=False):
    out = nn.Module()

    out_shape0 = out_shape
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape

    if expand == True:
        # out_shape0 = out_shape
        # out_shape1 = out_shape * 2
        # out_shape2 = out_shape * 4
        # out_shape3 = out_shape * 8
        # out_shape4 = out_shape * 16

        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    # out.layer0_rn = nn.Conv2d(
    #     in_shape[0],
    #     out_shape1,
    #     kernel_size=1,
    #     stride=1,
    #     padding=0,
    #     bias=False,
    #     groups=groups,
    # )

    out.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=groups,
    )
    out.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=groups,
    )
    out.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=groups,
    )
    out.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=groups,
    )

    return out


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, layer_norm, isFlow=False):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.layer_norm = layer_norm
        self.isFlow = isFlow
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.layer_norm,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.layer_norm,
            groups=self.groups,
        )

        if self.layer_norm == True:
            self.layer_norm1 = nn.BatchNorm2d(features)
            self.layer_norm2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.layer_norm == True:
            out = self.layer_norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.layer_norm == True:
            out = self.layer_norm2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class SoftAttDepth(nn.Module):
    def __init__(self, alpha=0.01, beta=1.0, dim=1, discretization='UD'):
        super(SoftAttDepth, self).__init__()
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.discretization = discretization

    def get_depth_sid(self, depth_labels):
        alpha_ = torch.FloatTensor([self.alpha])
        beta_ = torch.FloatTensor([self.beta])
        t = []
        for K in range(depth_labels):
            K_ = torch.FloatTensor([K])
            t.append(torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * K_ / depth_labels))
        t = torch.FloatTensor(t)
        return t

    def forward(self, input_t, eps=1e-6):
        batch_size, depth, height, width = input_t.shape
        if self.discretization == 'SID':
            grid = self.get_depth_sid(depth).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            grid = torch.linspace(
                self.alpha, self.beta, depth,
                requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        grid = grid.repeat(batch_size, 1, height, width).float()

        z = F.softmax(input_t, dim=self.dim)
        z = z * (grid.to(z.device))
        z = torch.sum(z, dim=1, keepdim=True)

        return z


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        layer_norm=False,
        expand=False,
        align_corners=True,
        scale=1,
        input_length=2,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.scale = scale
        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            if features == 256:
                out_features = features
            else:
                out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )
        self.dim = 1
        if input_length == 2:
            self.resConfUnit1 = ResidualConvUnit(features, activation, layer_norm)
            self.en_atten = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, stride=1, padding=0)
            self.skip_add = nn.quantized.FloatFunctional()
        self.resConfUnit2 = ResidualConvUnit(features, activation, layer_norm)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        df = xs[0]  # 11763

        if len(xs) == 2:
            if self.scale == 1:
                res = self.skip_add.add(df, xs[1])  # 16557
            else:
                import pdb
                pdb.set_trace()
                res = df
            # resatten verison
            # ef = nn.functional.interpolate(
            #      self.resConfUnit1(xs[1]), scale_factor=self.scale, mode="bilinear", align_corners=self.align_corners
            #      )

            att = F.softmax(self.en_atten(self.resConfUnit1(xs[1])), dim=self.dim)
            out = res * att  # 84
            output = self.skip_add.add(self.resConfUnit2(out), res)

            # output = self.resConfUnit2(res)
        else:
            output = self.resConfUnit2(df)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_norm, scale=1, input_length=2):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        layer_norm=use_norm,
        expand=False,
        align_corners=True,
        scale=scale,
        input_length=input_length,
    )


@SEM_SEG_HEADS_REGISTRY.register()
class TransDSSL(nn.Module):
    def __init__(
        self,
        cfg, input_shape: Dict[str, ShapeSpec], *,
        features=256,
        use_norm=False,
    ):
        super(TransDSSL, self).__init__()

        self.layers = _make_1x1_convs(
            [96, 192, 384, 768, 48], features, groups=1, expand=False
        )

        self.upsample = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        self.layers.refinenet0 = _make_fusion_block(features, use_norm, input_length=2)
        self.layers.refinenet1 = _make_fusion_block(features, use_norm, input_length=2)
        self.layers.refinenet2 = _make_fusion_block(features, use_norm, input_length=2)
        self.layers.refinenet3 = _make_fusion_block(features, use_norm, input_length=2)
        self.layers.refinenet4 = _make_fusion_block(features, use_norm, input_length=1)
        self.attn_depth = SoftAttDepth()

        self.layers.output_conv4 = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )
        self.layers.output_conv3 = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )
        self.layers.output_conv2 = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )

        self.layers.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        return ret

    def forward_features(self, features):

        layer_1, layer_2, layer_3, layer_4 = features['res2'], features['res3'], features['res4'], features['res5']

        layer_1_rn = self.layers.layer1_rn(layer_1)
        layer_2_rn = self.layers.layer2_rn(layer_2)
        layer_3_rn = self.layers.layer3_rn(layer_3)
        layer_4_rn = self.layers.layer4_rn(layer_4)

        path_4 = self.layers.refinenet4(layer_4_rn)

        path_3 = self.layers.refinenet3(path_4, layer_3_rn)

        disp_3 = self.layers.output_conv4(path_3)
        disp_3 = self.attn_depth(disp_3)

        path_2 = self.layers.refinenet2(path_3, layer_2_rn)
        disp_2 = self.layers.output_conv3(path_2)
        disp_2 = self.attn_depth(disp_2)

        path_1 = self.layers.refinenet1(path_2, layer_1_rn)

        disp_1 = self.layers.output_conv2(path_1)
        disp_1 = self.attn_depth(disp_1)

        layer_0_rn = self.upsample(layer_1_rn)
        path_0 = self.layers.refinenet0(path_1, layer_0_rn)
        disp_0 = self.layers.output_conv(path_0)

        disp_0 = self.attn_depth(disp_0)
        return {
            ("disp", 3): disp_3,
            ("disp", 2): disp_2,
            ("disp", 1): disp_1,
            ("disp", 0): disp_0,
        }