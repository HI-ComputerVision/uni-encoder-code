import torch
import numbers
import numpy as np
import random
import cv2
import torchvision.transforms.functional as F
from fvcore.transforms.transform import Transform
from detectron2.data.transforms import Augmentation


class CustomColorAugSSDAugment(Augmentation):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()
        self.img_format = img_format
        self.brightness_delta = brightness_delta
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self.hue_delta = hue_delta

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def __call__(self, img):
        beta_brightness = random.uniform(
            -self.brightness_delta, self.brightness_delta) if random.randrange(2) else None
        con_sat_hue_order = True if random.randrange(2) else False
        alpha_contrast = random.uniform(
            self.contrast_low, self.contrast_high) if random.randrange(2) else None
        alpha_saturation = random.uniform(
            self.saturation_low, self.saturation_high) if random.randrange(2) else None
        hue = random.randint(-self.hue_delta, self.hue_delta)
        tfms = [
            CustomColorAugSSDTransform(
                self.img_format,
                beta_brightness=beta_brightness,
                con_sat_hue_order=con_sat_hue_order,
                alpha_contrast=alpha_contrast,
                alpha_saturation=alpha_saturation,
                hue=hue,
            )
        ]
        return img.apply_augmentations(tfms)


class CustomColorAugSSDTransform(Transform):
    def __init__(
        self,
        img_format,
        beta_brightness=None,
        con_sat_hue_order=True,
        alpha_contrast=None,
        alpha_saturation=None,
        hue=None,
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        del img_format
        self._set_attributes(locals())

        self.beta_brightness = beta_brightness
        self.con_sat_hue_order = con_sat_hue_order
        self.alpha_contrast = alpha_contrast
        self.alpha_saturation = alpha_saturation
        self.hue = hue

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]

        if self.beta_brightness is not None:
            img = self.convert(img, beta=self.beta_brightness)
        if self.con_sat_hue_order:
            if self.alpha_contrast is not None:
                img = self.convert(img, alpha=self.alpha_contrast)
        if self.alpha_saturation is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=self.alpha_saturation)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if self.hue is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(
                    int) + self.hue
            ) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if not self.con_sat_hue_order:
            if self.alpha_contrast is not None:
                img = self.convert(img, alpha=self.alpha_contrast)

        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)


class CustomColorJitterAugment(Augmentation):
    """
    A color related data augmentation used in Monodepth2.

    Implementation based on:

     https://github.com/nianticlabs/monodepth2
       /blob/master/datasets/mono_dataset.py

     https://pytorch.org/vision/main/_modules
       /torchvision/transforms/transforms.html#ColorJitter
    """

    def __init__(
        self,
        img_format,
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=(-0.1, 0.1),
    ):
        super().__init__()
        self.img_format = img_format
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def __call__(self, img):
        fn_idx = torch.randperm(4)
        b = None if self.brightness is None else float(
            torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        c = None if self.contrast is None else float(
            torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
        s = None if self.saturation is None else float(
            torch.empty(1).uniform_(self.saturation[0], self.saturation[1]))
        h = None if self.hue is None else float(torch.empty(1).uniform_(self.hue[0], self.hue[1]))
        tfms = [
            CustomColorJitterTransform(
                self.img_format,
                brightness=b,
                contrast=c,
                saturation=s,
                hue=h,
                fn_idx=fn_idx,
            )
        ]
        return img.apply_augmentations(tfms)


class CustomColorJitterTransform(Transform):
    def __init__(
        self,
        img_format,
        brightness=None,
        contrast=None,
        saturation=None,
        hue=None,
        fn_idx=None,
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        del img_format
        self._set_attributes(locals())

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.fn_idx = fn_idx

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = torch.from_numpy(img).permute(2, 0, 1)

        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness is not None:
                img = img = F.adjust_brightness(img, self.brightness)
            elif fn_id == 1 and self.contrast is not None:
                img = F.adjust_contrast(img, self.contrast)
            elif fn_id == 2 and self.saturation is not None:
                img = F.adjust_saturation(img, self.saturation)
            elif fn_id == 3 and self.hue is not None:
                img = F.adjust_hue(img, self.hue)

        img = img.permute(1, 2, 0).numpy()
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img
