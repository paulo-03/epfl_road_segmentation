"""
Helper functions to extract features from images
"""

import numpy as np


def extract_images_features(images: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Extract features from a list of images.

    :param images: the list of images
    :param patch_size: the width and height in pixels of each patch
    :return: the list of features
    """
    img_patches = np.array(
        [extract_patches(img, patch_size, patch_size) for img in images]
    )
    img_patches = np.reshape(img_patches, (-1, patch_size, patch_size, 3))
    features = np.array(
        [extract_features(img_patch) for img_patch in img_patches]
    )
    return features


def extract_patches(img, w, h):
    """
    Extract patches from an image.

    :param img: the image
    :param w: the width in pixels of each patch
    :param h: the height in pixels of each patch
    :return: the list of patches
    """
    patches = []
    img_width = img.shape[0]
    img_height = img.shape[1]
    is_2d = len(img.shape) < 3

    for i in range(0, img_height, h):
        for j in range(0, img_width, w):
            if is_2d:
                im_patch = img[j: j + w, i: i + h]
            else:
                im_patch = img[j: j + w, i: i + h, :]
            patches.append(im_patch)
    return patches


def extract_features(patch: np.ndarray) -> np.ndarray:
    """
    Extract features from a patch (the mean and variance of each channel).

    :param patch: the patch
    :return: the array features
    """
    dim = len(patch.shape)
    if dim == 2:
        feat_mean = np.mean(patch)
        feat_var = np.var(patch)
    elif dim == 3:
        feat_mean = np.mean(patch, axis=(0, 1))
        feat_var = np.var(patch, axis=(0, 1))
    else:
        raise AttributeError(f"Images should be of dimension 2 or 3 but are instead of dimension {dim}")
    features = np.append(feat_mean, feat_var)
    return features


def ground_truth_patch_to_int(patch: np.ndarray, threshold: float) -> int:
    """
    Determines if a patch is considered as road or not

    :param patch: a patch of an image
    :param threshold: the minimum fraction of road pixels to consider a patch to be of type road
    :return: 1 if the patch is of type road, 0 otherwise
    """
    return int(np.mean(patch) >= threshold)


def label_to_img(img_width: int, img_height: int, w: int, h: int, labels: list[int]) -> np.ndarray:
    """
    Convert labels for each patch into an image.

    :param img_width: the width in pixels of the image
    :param img_height: the height in pixels of the image
    :param w: the width in pixels of each patch
    :param h: the height in pixels of each patch
    :param labels: the list of labels
    :return: the image constructed from the patches
    """
    im = np.zeros([img_width, img_height])
    idx = 0
    for i in range(0, img_height, h):
        for j in range(0, img_width, w):
            im[j: j + w, i: i + h] = 255 * labels[idx]
            idx = idx + 1
    return im
