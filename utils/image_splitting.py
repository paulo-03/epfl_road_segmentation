"""
Helper functions to split images into sub-images and combine sub-images in one image
"""
from os import path, listdir

from PIL import Image
from PIL import ImageOps


def _load_images(folder_path: str) -> list[Image.Image]:
    """
    Load all images of the dataset folder

    :param folder_path: folder containing the images
    :return: list of images
    """
    images_path = list(sorted(listdir(folder_path)))
    images = [Image.open(path.join(folder_path, image_path)) for image_path in images_path]
    return images


def split_training_images(training_folder: str, split_images_folder: str, split_gt_folder: str, patch_size: int):
    """
    Split images and groundtruth into square patches of size patch_size x patch_size and store them as images

    :param training_folder: the folder containing the images and groundtruth
    :param split_images_folder: the folder where to store the split images
    :param split_gt_folder: the folder where to store the split groundtruth
    :param patch_size: the width and height in pixels of each patch
    """
    # load images and groundtruth
    images = _load_images(path.join(training_folder, 'images'))
    groundtruth = _load_images(path.join(training_folder, 'groundtruth'))

    # resize images and groundtruths to be divisible by patch_size
    w, h = groundtruth[0].size
    new_w = round(w / patch_size) * patch_size
    new_h = round(h / patch_size) * patch_size
    images = [img.resize((new_w, new_h)) for img in images]
    groundtruth = [img.resize((new_w, new_h)) for img in groundtruth]

    for image_nb, (img, gt) in enumerate(zip(images, groundtruth)):
        for i in range(0, new_w, patch_size):
            for j in range(0, new_h, patch_size):
                # extract two square patch from the image and its ground truth
                img_patch = img.crop((i, j, i + patch_size, j + patch_size))
                gt_patch = gt.crop((i, j, i + patch_size, j + patch_size))

                # store the created patch as an image
                img_patch.save(path.join(split_images_folder, f'satImage_{image_nb + 1:03d}_{i:03d}_{j:03d}.png'))
                gt_patch.save(path.join(split_gt_folder, f'satImage_{image_nb + 1:03d}_{i:03d}_{j:03d}.png'))


def split_testing_images(testing_folder, patch_size):
    """
    Split test images into square patches of size patch_size x patch_size

    :param testing_folder: the folder containing the test images
    :param patch_size: the width and height in pixels of each patch
    :return: the list of patches, the number of test images, the width and height of the test images and the padding
    """
    # load test images
    test_images = _load_images(testing_folder)

    # pad test images with black pixels to be divisible by patch_size
    n_test_images = len(test_images)
    w, h = test_images[0].size[:2]
    pad = patch_size - (w % patch_size)
    test_images = [ImageOps.expand(img, border=(0, 0, pad, pad), fill='black') for img in test_images]
    w, h = test_images[0].size[:2]

    test_patches = []
    for image in test_images:
        for i in range(0, w, patch_size):
            for j in range(0, h, patch_size):
                # extract a square patch from the test image
                img_patch = image.crop((i, j, i + patch_size, j + patch_size))
                test_patches.append(img_patch)

    return test_patches, n_test_images, w, h, pad


def combine_patches(patches: list[Image.Image], patch_size: int, n_images: int, w: int, h: int, pad: int,
                    folder: str = None, save: bool = True):
    """
    Combine patches into images

    :param patches: the list of patches
    :param patch_size: the width and height in pixels of each patch
    :param n_images: the number of patches
    :param w: the width in pixels of each image
    :param h: the height in pixels of each image
    :param pad: the padding in pixels of each image
    :param folder: the folder where to store the images
    """
    patch_index = 0
    predictions = []
    for n in range(n_images):
        test_images_predictions = Image.new('L', (w, h))
        # assemble the image from its patches
        for i in range(0, w // patch_size):
            for j in range(0, h // patch_size):
                test_images_predictions.paste(
                    patches[patch_index], (i * patch_size, j * patch_size)
                )
                patch_index += 1
        # remove the padding
        test_images_predictions = test_images_predictions.crop((0, 0, w - pad, h - pad))

        predictions.append(test_images_predictions)
        # store the image
        if save:
            test_images_predictions.save(f'{folder}/pred_{n + 1:03d}.png')

    return predictions
