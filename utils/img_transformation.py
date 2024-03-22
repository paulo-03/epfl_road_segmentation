"""Functions to help transforming the images"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def rotate(image: Image.Image, angle: float = 90) -> Image.Image:
    """
    Rotate the given image by the given angle

    :param image: the image to rotate
    :param angle: the angle to rotate the image by
    :return: the rotated image
    """
    return image.rotate(angle)


def crop_resize(image: Image.Image, box: tuple[int, int, int, int] = (100, 100, 300, 300)) -> Image.Image:
    """
    Crop the given image with the given rectangle

    :param image: the image to crop
    :param box: the rectangle we want to extract from the image
    :return: the cropped image
    """
    original_size = image.size
    cropped_img = image.crop(box=box)
    return cropped_img.resize(original_size)


def add_gaussian_noise(image: Image.Image, mean: float = 0, std: float = 25) -> Image.Image:
    """
    Return the original image with gaussian noise added to it

    :param image: the image to alter
    :param mean: the mean of the gaussian noise
    :param std: the standard deviation of the gaussian noise
    :return: the noisy image
    """
    # Create a Gaussian noise array
    image_array = np.array(image)
    height, width, channels = image_array.shape
    gaussian_noise = np.random.normal(mean, std, (height, width, channels)).astype(np.uint8)
    # Add the noise to the image
    noisy_image = image_array + gaussian_noise
    # Clip pixel values to be in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return Image.fromarray(noisy_image)


def color_distortion(image: Image.Image, scale_factors: tuple[float, float, float] = (1.5, 1.2, 0.8)) \
        -> Image.Image:
    """
    Return the original after applying color distortion to it

    :param image: the image to alter
    :param scale_factors: the scale factors for each channel (RGB)
    :return: the distorted image
    """
    # Apply color distortion by scaling the color channels
    distorted_img = np.array(image) * np.array(scale_factors)
    # Clip pixel values to be in the valid range [0, 255]
    distorted_img = np.clip(distorted_img, 0, 255).astype(np.uint8)
    return Image.fromarray(distorted_img)


def flip_vertical(image: Image.Image) -> Image.Image:
    """
    Flip the given image vertically

    :param image: the image to flip
    :return: the original image flipped vertically
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def flip_horizontal(image: Image.Image) -> Image.Image:
    """
    Flip the given image horizontally

    :param image: the image to flip
    :return: the original image flipped horizontally
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def show_transformations(img: Image.Image, gt: Image.Image, angle: float = 90,
                         box: tuple[int, int, int, int] = (100, 100, 300, 300), mean: float = 0, std: float = 20,
                         scale_factors: tuple[float, float, float] = (1.3, 1.1, 0.8)):
    """
    Display all transformation that an image can go through

    :param img: original image
    :param gt: original ground truth image
    :param angle: rotation angle
    :param box: box to crop
    :param mean: gaussian noise mean
    :param std: gaussian noise standard deviation
    :param scale_factors: factors for each channel to multiply intensity of each pixel
    """
    altered_images = [
        (img, gt, 'Original'),
        (rotate(img, angle), rotate(gt, angle), 'Rotated'),
        (crop_resize(img, box), crop_resize(gt, box), 'Cropped'),
        (add_gaussian_noise(img, mean, std), gt, 'Noisy'),
        (color_distortion(img, scale_factors), gt, 'Color Distortion'),
        (flip_horizontal(img), flip_horizontal(gt), 'Flipped')
    ]

    # Create a figure to plot the altered images
    fig, axs = plt.subplots(2, 6, figsize=(20, 8))

    for i, (img, gt, text) in enumerate(altered_images):
        axs[0][i].imshow(img, cmap='Greys_r')
        axs[0][i].set_title(text + ' Image')
        axs[1][i].imshow(gt, cmap='Greys_r')
        axs[1][i].set_title(text + ' Ground Truth')

    plt.show()
