import math
import random
from scipy import ndimage
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
import time

from gas.generation.util.image import ensure_mask_3d


def apply_random_augmentations(
    inputs,
    mask=None,
    level_probs=None,
    level=None,
    resize_prob=0.5,
    crop_prob=0.5,
):
    """
    Apply image transformations based on randomly selected difficulty level.

    Args:
        inputs: np.ndarray or tuple of np.ndarray. Image(s) to transform
        mask: np.ndarray, optional. Binary mask to ensure crop contains mask foreground.
            If provided, the returned aug_mask may be 3D (H, W, 1); squeeze to 2D (H, W) for storage or training as needed.
        level_probs: dict with augmentation levels and their probabilities.
            Default probabilities:
            - Level 0 (25%): No augmentations (base transforms)
            - Level 1 (25%): Basic augmentations
            - Level 2 (25%): Medium distortions
            - Level 3 (25%): Hard distortions
        level: set to override level_probs
        resize_prob: probability of resizing image (keeping aspect ratio the same)
        crop_prob: probability of cropping randomly

    Returns:
        tuple: (aug_image, aug_mask, level, transform_params)
            aug_image: Augmented image(s) as np.ndarray
            aug_mask: Augmented mask (if provided, else None)
                (Note: aug_mask may be 3D (H, W, 1); squeeze to 2D for storage/training.)
            level: int, chosen augmentation level
            transform_params: dict, parameters used for the transforms

    Raises:
        ValueError: If probabilities don't sum to 1.0 (within floating point precision)
    """
    if level is None:
        if level_probs is None:
            level_probs = {
                0: 0.25,  # No augmentations (base transforms)
                1: 0.25,  # Basic augmentations
                2: 0.25,  # Medium distortions
                3: 0.25,  # Hard distortions
            }

        if not math.isclose(sum(level_probs.values()), 1.0, rel_tol=1e-9):
            raise ValueError("Probabilities of levels must sum to 1.0")

        # get cumulative probs and select augmentation level
        cumulative_probs = {}
        cumsum = 0
        for level, prob in sorted(level_probs.items()):
            cumsum += prob
            cumulative_probs[level] = cumsum

        rand_val = np.random.random()
        for curr_level, cum_prob in cumulative_probs.items():
            if rand_val <= cum_prob:
                level = curr_level
                break

    # determine target resolution (for resize)
    if isinstance(inputs, tuple):
        original_res = inputs[0].shape[1:3]
    else:
        original_res = inputs.shape[0:2] if inputs.ndim == 3 else inputs.shape[1:3]

    resize_res = original_res
    if np.random.rand() < resize_prob:
        scale = np.random.uniform(0.5, 1.5)
        target_h = min(int(original_res[0] * scale), 2048)
        target_w = min(int(original_res[1] * scale), 2048)
        resize_res = (target_h, target_w)        

    # determine crop scale for h and w
    crop_scale = (1., 1.)
    if np.random.rand() < crop_prob:
        crop_scale = (np.random.uniform(0.35, 0.99), np.random.uniform(0.35, 0.99))
        min_scale_h = max(0.35, 224 / resize_res[0])
        min_scale_w = max(0.35, 224 / resize_res[1])
        crop_scale = (max(crop_scale[0], min_scale_h), max(crop_scale[1], min_scale_w))

    if level == 0:
        tforms = get_base_transforms(resize_res, crop_scale)
    elif level == 1:
        tforms = get_random_augmentations(resize_res, crop_scale)
    elif level == 2:
        tforms = get_random_augmentations_medium(resize_res, crop_scale)
    else:  # level == 3
        tforms = get_random_augmentations_hard(resize_res, crop_scale)

    if isinstance(inputs, tuple):
        transformed_A, _ = tforms(inputs[0], reuse_params=False)
        transformed_B, _ = tforms(inputs[1], reuse_params=True)
        transformed = np.concatenate([transformed_A, transformed_B], axis=0)

        return transformed, None, level, tforms.params
    else:
        transformed_inputs, transformed_masks = tforms(inputs, mask, reuse_params=False)
        return transformed_inputs, transformed_masks, level, tforms.params


def get_base_transforms(target_res, crop_scale):
    """
    Get basic transforms (center crop and resize).

    Args:
        target_image_size: int or tuple. Output size for resize.

    Returns:
        ComposeWithParams: Composed transform pipeline
    """
    return ComposeWithParams(
        [
            ResizeCropWithParams(target_res, crop_scale),
        ]
    )


def get_random_augmentations(target_res, crop_scale):
    """
    Get basic augmentations with geometric transforms.

    Args:
        target_image_size: int or tuple. Output size for resize.

    Returns:
        ComposeWithParams: Composed transform pipeline with basic augmentations
    """
    base_augmentations = [
        RandomRotationWithParams(degrees=20, order=2),
        ResizeCropWithParams(target_res, crop_scale),
        RandomHorizontalFlipWithParams(),
        RandomVerticalFlipWithParams(),
    ]
    return ComposeWithParams(base_augmentations)


def get_random_augmentations_medium(target_res, crop_scale):
    """
    Get medium difficulty transforms with mild distortions.

    Args:
        target_image_size: int or tuple. Output size for resize.

    Returns:
        ComposeWithParams: Composed transform pipeline with medium distortions
    """
    base_augmentations = get_random_augmentations(target_res, crop_scale)
    distortions = [
        ApplyDeeperForensicsDistortion("CS", level_min=0, level_max=1),
        ApplyDeeperForensicsDistortion("CC", level_min=0, level_max=1),
        ApplyDeeperForensicsDistortion("JPEG", level_min=0, level_max=1),
    ]
    return ComposeWithParams(base_augmentations.transforms + distortions)


def get_random_augmentations_hard(target_res, crop_scale):
    """
    Get hard difficulty transforms with more severe distortions.

    Args:
        target_image_size: int or tuple. Output size for resize.

    Returns:
        ComposeWithParams: Composed transform pipeline with severe distortions
    """
    base_augmentations = get_random_augmentations(target_res, crop_scale)
    distortions = [
        ApplyDeeperForensicsDistortion("CS", level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion("CC", level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion("JPEG", level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion("GNC", level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion("GB", level_min=0, level_max=2),
    ]
    return ComposeWithParams(base_augmentations.transforms + distortions)


class ComposeWithParams:
    def __init__(self, transforms):
        """
        Compose multiple transforms while tracking their randomly selected parameters.
        Useful for logging or situations where transforms need to be reapplied with
        the same parameters.

        Args:
            transforms: list of transform objects to compose
        """
        self.transforms = transforms
        self.params = {}

    def __call__(self, frames, masks=None, reuse_params=False):
        """
        Apply composed transforms to frames and optional masks.

        Args:
            frames: np.ndarray, the image(s) to transform
            masks: np.ndarray, optional mask(s) to transform
            reuse_params: bool, if True, reuse previous params (for paired images/mask)

        Returns:
            tuple: (output_frames, output_masks)
                output_frames: np.ndarray, transformed frames
                output_masks: np.ndarray or None, transformed masks if provided
        """
        if not reuse_params:
            self.params = {}

        is_single_image = frames.ndim == 3  # (H, W, C)
        if is_single_image:
            # Add fake temporal dim → (1, H, W, C)
            frames = frames[None, ...]
            masks = masks[None, ...] if masks is not None else None

        output_frames = []
        output_masks = []

        for i in range(frames.shape[0]):
            frame = frames[i]
            mask = ensure_mask_3d(masks[i]) if masks is not None else None

            for transform in self.transforms:
                frame, mask_ = self.apply_transform(
                    image=frame, mask=mask, transform=transform
                )
                mask = mask if mask_ is None else mask_

            output_frames.append(frame)
            if mask is not None:
                if mask.ndim == 3:
                    mask = (mask.sum(axis=2) > 0).astype(np.uint8)
                output_masks.append(mask)

        if is_single_image:
            output_frames = output_frames[0]
            if len(output_masks):
                output_masks = output_masks[0]

        output_frames = np.array(output_frames)
        if len(output_masks):
            output_masks = np.array(output_masks)
            output_masks = np.array(output_masks > 0).astype(np.uint8)

        return output_frames, output_masks

    def apply_transform(self, image, mask, transform):
        """
        Apply a single transform while tracking its parameters.

        Args:
            image: np.ndarray, image to transform
            mask: np.ndarray or None, optional mask to transform
            transform: transform object to apply

        Returns:
            tuple: (transformed_image, transformed_mask)
                transformed_image: np.ndarray, transformed image
                transformed_mask: np.ndarray or None, transformed mask if provided
        """
        transform_name = getattr(transform, "__name__", transform.__class__.__name__)
        if transform_name in self.params:
            output = transform(image, mask=mask, **self.params[transform_name])
        else:
            output = transform(image, mask=mask)
            if hasattr(transform, "params"):
                self.params[transform_name] = transform.params

        if isinstance(output, tuple):
            tform_image = output[0]
            tform_mask = output[1]
        else:
            tform_image = output
            tform_mask = None

        return tform_image, tform_mask


class ApplyDeeperForensicsDistortion:
    """Wrapper for applying DeeperForensics distortions."""

    def __init__(self, distortion_type, level_min=0, level_max=3):
        """
        Initialize distortion transform.

        Args:
            distortion_type: str, type of distortion to apply
            level_min: int, minimum distortion level
            level_max: int, maximum distortion level
        """
        self.__name__ = distortion_type
        self.distortion_type = distortion_type
        self.level = None
        self.level_min = level_min
        self.level_max = level_max
        self.params = {}  # level
        self.distortion_params = {}  # distortion_type specific

    def __call__(self, img, level=None, **kwargs):
        """
        Apply distortion transform.

        Args:
            img: np.ndarray, input image to distort
            level: int or None, optional distortion level
            **kwargs: additional keyword arguments

        Returns:
            np.ndarray: Distorted image
        """
        if level is None and self.level is None:
            self.level = random.randint(self.level_min, self.level_max)
            self.params = {"level": self.level}
        elif self.level is None:
            self.level = level
            self.params = {"level": self.level}

        if self.level > 0:
            self.distortion_func = get_distortion_function(self.distortion_type)
            if len(self.distortion_params) == 0:
                self.distortion_param = get_distortion_parameter(
                    self.distortion_type, self.level
                )
                self.distortion_params = {"param": self.distortion_param}
        else:
            return img

        output = self.distortion_func(img, **self.distortion_params)
        if isinstance(output, tuple):
            self.distortion_params.update(output[1])
            return output[0]
        else:
            return output


# DeeperForensics Distortion Functions
def get_distortion_parameter(distortion_type, level):
    """Get distortion parameter based on type and level.

    Args:
        distortion_type: str, type of distortion
        level: int, distortion level (1-5)

    Returns:
        float or int: Parameter value for the specified distortion

    Parameters are arranged from least severe (level 1) to most severe (level 5).
    Each distortion type has different parameter behavior:

    CS (Color Saturation):
        - Range: [0.4 -> 0.0]
        - Lower values = worse distortion
        - 0.4 = slight desaturation
        - 0.0 = complete desaturation (grayscale)

    CC (Color Contrast):
        - Range: [0.85 -> 0.35]
        - Lower values = worse distortion
        - 0.85 = slight contrast reduction
        - 0.35 = severe contrast reduction

    BW (Block Wise):
        - Range: [16 -> 80]
        - Higher values = worse distortion
        - Controls number of random blocks added
        - 16 = few blocks
        - 80 = many blocks

    GNC (Gaussian Noise Color):
        - Range: [0.001 -> 0.05]
        - Higher values = worse distortion
        - Controls noise variance
        - 0.001 = subtle noise
        - 0.05 = very noisy

    GB (Gaussian Blur):
        - Range: [7 -> 21]
        - Higher values = worse distortion
        - Controls blur kernel size
        - 7 = slight blur
        - 21 = heavy blur

    JPEG (JPEG Compression):
        - Range: [2 -> 6]
        - Higher values = worse distortion
        - Controls downsampling factor
        - 2 = mild compression
        - 6 = severe compression
    """
    param_dict = {
        "CS": [0.4, 0.3, 0.2, 0.1, 0.0],
        "CC": [0.85, 0.725, 0.6, 0.475, 0.35],
        "BW": [16, 32, 48, 64, 80],
        "GNC": [0.001, 0.002, 0.005, 0.01, 0.05],
        "GB": [7, 9, 13, 17, 21],
        "JPEG": [2, 3, 4, 5, 6],
    }
    return param_dict[distortion_type][level - 1]


def get_distortion_function(distortion_type):
    """
    Args:
        distortion_type: str, type of distortion

    Returns:
        callable: Function that implements the specified distortion
    """
    func_dict = {
        "CS": color_saturation,
        "CC": color_contrast,
        "BW": block_wise,
        "GNC": gaussian_noise_color,
        "GB": gaussian_blur,
        "JPEG": jpeg_compression,
    }
    return func_dict[distortion_type]


def rgb2ycbcr(img_rgb):
    """Convert RGB image to YCbCr color space.

    Args:
        img_rgb (np.ndarray): RGB image array of shape (H, W, 3)

    Returns:
        np.ndarray: YCbCr image array of shape (H, W, 3) with values normalized to [0,1]
    """
    img_rgb = img_rgb.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0
    return img_ycbcr


def ycbcr2rgb(img_ycbcr):
    """Convert YCbCr image to RGB color space.

    Args:
        img_ycbcr (np.ndarray): YCbCr image array of shape (H, W, 3)

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) with values in [0,255]
    """
    img_ycbcr = img_ycbcr.astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_rgb = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return img_rgb


def color_saturation(img, param):
    """Apply color saturation distortion.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Saturation multiplier parameter

    Returns:
        np.ndarray: Distorted RGB image array with modified saturation
    """
    ycbcr = rgb2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2rgb(ycbcr).astype(np.uint8)
    return img


def color_contrast(img, param):
    """Apply color contrast distortion.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Contrast multiplier parameter

    Returns:
        np.ndarray: Distorted RGB image array with modified contrast
    """
    img = img.astype(np.float32) * param
    return img.astype(np.uint8)


def block_wise(img, param):
    """Apply block-wise distortion by adding random gray blocks.

    NOTE: CURRENTLY NOT USED

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Number of blocks to add, scaled by image dimensions

    Returns:
        np.ndarray: Distorted RGB image array with added gray blocks
    """
    width = 8
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(img.shape[0], img.shape[1]) // 256 * param
    for _ in range(param):
        r_w = random.randint(0, img.shape[1] - 1 - width)
        r_h = random.randint(0, img.shape[0] - 1 - width)
        img[r_h : r_h + width, r_w : r_w + width, :] = block
    return img


def gaussian_noise_color(img, param):
    """Apply colored Gaussian noise in YCbCr color space.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Variance of the Gaussian noise

    Returns:
        tuple: (distorted_image, params)
            distorted_image: np.ndarray, image with added color noise
    """
    ycbcr = rgb2ycbcr(img) / 255
    size_a = ycbcr.shape
    b = (
        ycbcr + math.sqrt(param) * np.random.randn(size_a[0], size_a[1], size_a[2])
    ) * 255
    b = ycbcr2rgb(b)
    return np.clip(b, 0, 255).astype(np.uint8)


def gaussian_blur(img, param):
    """Apply Gaussian blur with specified kernel size.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Gaussian kernel size (must be odd)

    Returns:
        np.ndarray: Blurred RGB image array
    """
    return cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)


def jpeg_compression(img, param):
    """Apply JPEG compression-like distortion through downsampling.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Downsampling factor

    Returns:
        np.ndarray: Distorted RGB image array with compression artifacts
    """
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    return cv2.resize(img, (w, h))


class CenterCrop:
    """Center crop an image to a square."""

    def crop(self, img):
        """
        Perform center crop.

        Args:
            img (np.ndarray): Input image array

        Returns:
            np.ndarray: Center cropped image array
        """
        h, w = img.shape[:2]
        m = min(h, w)
        i = (h - m) // 2
        j = (w - m) // 2
        return img[i : i + m, j : j + m]

    def __call__(self, img, mask=None):
        """
        Apply center crop transform to an image an optionally an associated mask.

        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array

        Returns:
            np.ndarray or tuple: Cropped image, or tuple of (image, mask) if mask provided
        """
        img = self.crop(img)
        if mask is not None:
            mask = self.crop(mask)
            return img, mask
        return img


class ResizeCropWithParams:
    """Randomly crop and resize an image, ensuring the crop contains the mask foreground if a mask is provided."""

    def __init__(self, resize_res, crop_scale):
        """
        Initialize random resized crop transform.

        Args:
            size (int or tuple): Target output size
            scale (tuple): Range of size of the origin size cropped
        """
        self.params = None
        self.resize_res = resize_res
        self.crop_scale = crop_scale

    def __call__(self, img, mask=None, crop_params=None):
        """
        Apply random resized crop transform.

        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array
            crop_params (tuple, optional): Pre-computed crop parameters

        Returns:
            np.ndarray or tuple: Cropped and resized image, or tuple of (image, mask) if mask provided
        """
        img, mask = self.resize(img, mask)

        img = self.random_crop(
            img, mask=mask, crop_params=crop_params, interpolation=cv2.INTER_LINEAR
        )
        if mask is not None:
            prev_crop_params = self.params["crop_params"]
            mask = self.random_crop(
                mask,
                mask=None,
                crop_params=prev_crop_params,
                interpolation=cv2.INTER_NEAREST,
            )

        return img, mask


    def resize(self, img, mask=None):
        """
        Apply resize transform.

        Args:
            img: np.ndarray, input image
            mask: np.ndarray or None, optional mask to resize

        Returns:
            np.ndarray or tuple: Resized image, or tuple of (image, mask) if mask provided
        """
        if img.shape[:2] != self.resize_res:
            img = cv2.resize(img, (self.resize_res[1], self.resize_res[0]), interpolation=cv2.INTER_LINEAR)
            if mask is not None and mask.shape[:2] != self.resize_res:
                mask = cv2.resize(mask, (self.resize_res[1], self.resize_res[0]), interpolation=cv2.INTER_NEAREST)

        return img, mask

    def random_crop(
        self, img, mask=None, crop_params=None, interpolation=cv2.INTER_LINEAR
    ):
        """
        Randomly crop and resize an image, ensuring the crop contains the mask foreground if a mask is provided.

        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array
            crop_params (tuple, optional): Pre-computed crop parameters (i, j, h, w)
            interpolation (int): Interpolation method

        Returns:
            np.ndarray: Cropped and resized image array
        """
        if crop_params is None:
            height, width = img.shape[:2]
            h = int(height * self.crop_scale[0])
            w = int(width * self.crop_scale[1])

            h = min(h, height)
            w = min(w, width)

            if mask is not None:
                coords = np.where(mask > 0)
                ys, xs = coords[0], coords[1]

                if len(xs) == 0 or len(ys) == 0:
                    # No foreground, fall back to random crop
                    i = np.random.randint(0, height - h + 1)
                    j = np.random.randint(0, width - w + 1)
                else:
                    x0, x1 = xs.min(), xs.max()
                    y0, y1 = ys.min(), ys.max()
                    min_i = max(0, y1 - h + 1)
                    max_i = min(y0, height - h)
                    min_j = max(0, x1 - w + 1)
                    max_j = min(x0, width - w)
                    if min_i > max_i or min_j > max_j:
                        i = max(0, y0 - (h // 2))
                        j = max(0, x0 - (w // 2))
                    else:
                        i = np.random.randint(min_i, max_i + 1) if max_i >= min_i else 0
                        j = np.random.randint(min_j, max_j + 1) if max_j >= min_j else 0
            else:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
        else:
            i, j, h, w = crop_params

        self.params = {"crop_params": (i, j, h, w)}
        if img.ndim == 2:
            return img[i : i + h, j : j + w]
        return img[i : i + h, j : j + w, :]


class RandomHorizontalFlipWithParams:

    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of flipping the image
        """
        self.p = p
        self.params = {}

    def __call__(self, img, mask=None, flip=None):
        """
        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array
            flip (bool, optional): Pre-computed flip decision

        Returns:
            np.ndarray or tuple: Flipped image, or tuple of (image, mask) if mask provided
        """
        if flip is not None:
            self.params = {"flip": flip}
        elif not hasattr(self, "params") or len(self.params) == 0:
            flip = np.random.random() < self.p
            self.params = {"flip": flip}

        if self.params.get("flip", False):
            img = np.fliplr(img)
            mask = None if mask is None else np.fliplr(mask)

        if mask is not None:
            return img, mask
        return img


class RandomVerticalFlipWithParams:

    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of flipping the image
        """
        self.p = p
        self.params = {}

    def __call__(self, img, mask=None, flip=None):
        """
        Apply vertical flip transform.

        Args:
            img (np.ndarray): Input image array
            mask (np.ndarray, optional): Input mask array
            flip (bool, optional): Pre-computed flip decision

        Returns:
            np.ndarray or tuple: Flipped image, or tuple of (image, mask) if mask provided
        """
        if flip is not None:
            self.params = {"flip": flip}
        elif not hasattr(self, "params") or len(self.params) == 0:
            flip = np.random.random() < self.p
            self.params = {"flip": flip}

        if self.params.get("flip", False):
            img = np.flipud(img)
            mask = None if mask is None else np.flipud(mask)

        if mask is not None:
            return img, mask
        return img


class RandomRotationWithParams:
    """Randomly rotate an image."""

    def __init__(self, degrees, p=0.5, reshape=False, mode="reflect", order=2):
        """
        Initialize random rotation transform.

        Args:
            degrees (float or tuple): Range of degrees to select from. If float, uses (-degrees, degrees)
            p (float): Probability of rotating the image
            reshape (bool): If True, expands output image to fit rotated image
            mode (str): How to fill the border ('reflect', 'constant', etc)
            order (int): Interpolation order (0-5)
        """
        if isinstance(degrees, (tuple, list)):
            self.degrees = degrees
        else:
            self.degrees = (-degrees, degrees)
        self.p = p
        self.params = None
        self.reshape = reshape
        self.mode = mode
        self.order = order

    def __call__(self, img, mask, rotate=None, angle=None, order=None, **kwargs):
        """
        Perform rotation transform.

        Args:
            img (np.ndarray): Input RGB image array of shape (H, W, C)
            mask (np.ndarray, optional): Input mask array
            rotate (bool, optional): Pre-computed rotation decision
            angle (float, optional): Pre-computed rotation angle
            order (int, optional): Pre-computed interpolation order
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray or tuple: Rotated image, or tuple of (image, mask) if mask provided
        """
        if rotate is None:
            rotate = np.random.random() < self.p
            self.params = {"rotate": rotate}

        if not rotate:
            return img

        order = self.order if order is None else order
        if isinstance(order, (tuple, list)):
            order = random.randint(order[0], order[1])

        if angle is None:
            angle = random.uniform(self.degrees[0], self.degrees[1])

        self.params.update({"order": order, "angle": angle})
        img = ndimage.rotate(
            img, angle, reshape=self.reshape, mode=self.mode, order=order, axes=(0, 1)
        )
        if mask is not None:
            mask = ndimage.rotate(
                mask,
                angle,
                reshape=self.reshape,
                mode=self.mode,
                order=0,  # Nearest neighbor to preserve binary values
                axes=(0, 1),
            )
            return img, mask
        return img