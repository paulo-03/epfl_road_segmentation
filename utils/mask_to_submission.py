"""
Script to generate a csv file of predictions from the predicted images
"""

import matplotlib.image as mpimg
import numpy as np


def patch_to_label(patch: np.ndarray, threshold: float) -> int:
    """
    Determines if a patch is considered as road or not

    :param patch: a patch of an image
    :param threshold: the minimum fraction of road pixels to consider a patch to be of type road
    :return:
    - 1 if the patch is of type road, 0 otherwise
    """
    df = np.mean(patch)
    return int(df > threshold)


def mask_to_submission_strings(img_number: int, image_filename: str, threshold: float) -> list[str]:
    """
    Generates a list of strings to be written in the csv file from the given image

    :param img_number: the image number
    :param image_filename: the filename of the predicted image
    :param threshold: the minimum fraction of road pixels to consider a patch to be of type road
    :return:
    a list of strings to be written in the submission csv file
    """
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, threshold)
            yield f"{img_number:03d}_{j}_{i},{label}"


def masks_to_submission(submission_filename: str, image_filenames: list[str], threshold: float):
    """
    Generates a csv file of predictions from the given predicted images

    :param submission_filename: the filename of the csv file
    :param image_filenames: the list of predicted image filenames
    :param threshold: the minimum fraction of road pixels to consider a patch to be of type road
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i, fn in enumerate(image_filenames):
            f.writelines(f'{s}\n' for s in mask_to_submission_strings(i + 1, fn, threshold))


def array_to_submission(submission_filename: str, array: list[int], sqrt_n_patches: int, patch_size: int):
    """
    Generates a csv file of predictions from the given array of patches

    :param submission_filename: the filename of the csv file
    :param array: the array of patches
    :param sqrt_n_patches: the square root of the number of patches per image
    :param patch_size: the width and height in pixels of each patch
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for index, pixel in enumerate(array):
            img_number = 1 + index // (sqrt_n_patches ** 2)
            j = patch_size * ((index // sqrt_n_patches) % sqrt_n_patches)
            i = patch_size * (index % sqrt_n_patches)
            f.writelines(f'{img_number:03d}_{j}_{i},{pixel}\n')


if __name__ == '__main__':
    """
    Write predictions in a csv file from the images found in the given folder
    """
    # percentage of pixels with value 1 required to assign a foreground label to a patch
    foreground_threshold = 0.19

    output_filename = '../submission.csv'
    images_filenames = [f'../predictions/satImage_{i:03d}.png' for i in range(1, 51)]

    masks_to_submission(output_filename, images_filenames, foreground_threshold)
