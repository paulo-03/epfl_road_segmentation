"""
Helper functions to load and display images
"""

from os import listdir, path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

from .cnn import CNN
from .image_splitting import split_testing_images, combine_patches


def load_images(folder_path: str) -> list[Image.Image]:
    """
    Load all images of the dataset folder

    :param folder_path: folder containing the images
    :return: list of images
    """
    images_path = list(sorted(listdir(folder_path)))
    images = [Image.open(path.join(folder_path, image_path)) for image_path in images_path]
    return images


def concatenate_images(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Concatenate both images"""

    if image1.height != image2.height:
        raise Exception('Both images should have same height')

    # define the mode as rgb if any of the images are rgb, else greyscale
    mode = 'RGB' if image1.mode == 'RGB' or image2.mode == 'RGB' else 'L'
    dst = Image.new(mode, (image1.width + image2.width, image1.height))
    dst.paste(image1, (0, 0))
    dst.paste(image2, (image1.width, 0))

    return dst


def display_test_predictions(model: CNN, nb: int = 4, test_images_path: str = 'dataset/test_set_images'):
    # Load test images and resized them to go through the model
    test_images = load_images(test_images_path)[:nb]

    # Load test images and split them into patches
    test_patches, n_test_images, w, h, pad = split_testing_images(test_images_path, 96)
    # Compute predictions
    test_patches_predictions = model.predict(images=test_patches)
    # Assemble patches into images and save them
    predictions_test = combine_patches(test_patches_predictions, 96, n_test_images, w, h, pad, save=False)

    # Display the predictions next to the original image
    rows_number = np.ceil(nb / 4).astype(int)
    fig, ax = plt.subplots(rows_number, 4, figsize=(12, 3 * rows_number))

    for i in range(0, nb):
        ip = make_img_overlay(test_images[i], predictions_test[i])
        ax[i // 4][i % 4].imshow(ip, cmap='Greys')

    plt.show()


def make_img_overlay(img: Image.Image, predicted_img: Image.Image, overlay_ratio: float = 0.2) -> Image.Image:
    """
    Add the predicted image as Red overlay on the base image

    :param img: The base image
    :param predicted_img: The predicted image to add as overlay
    :param overlay_ratio: The ratio with which to blend the predicted image
    :return: The resulting image
    """
    w, h = img.size
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = np.array(predicted_img)

    background = img.convert('RGBA')
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, overlay_ratio)
    return new_img


def combine_predictions(predictions: list[list[Image.Image]], threshold: float = 0.1):
    """Try to enhance predictions by combining the predictions of multiple models epoch state

    :param predictions: The predictions to combine
    :param threshold: Fraction of same prediction across models
    :return: The combined predictions of all the models
    """
    # Retrieve the number of model that are combining their predictions
    model_number = len(predictions)

    # Make sure all images have same size
    image_sizes = [model[0].size for model in predictions]
    assert len(set(image_sizes)) == 1, "Not all images have the same size, please make sure before combining them"

    # Convert all image in numpy
    # image: [h, w]
    # list image: [50, h, w]
    # list list image: [nb_model, 50, h, w]
    predictions_numpy = np.array([np.array([np.array(image) for image in model]) for model in predictions])

    # Combine all predictions.
    # Note that only one model need to predict a pixel is on, to put it on in the combined version.
    sum_predictions = np.sum(predictions_numpy, axis=0) / model_number
    clip_predictions = sum_predictions

    thres = 255 * threshold

    clip_predictions[clip_predictions <= thres] = 0
    clip_predictions[clip_predictions > thres] = 255

    return [Image.fromarray(np.uint8(img), "L") for img in clip_predictions]


def get_dataset_mean_std(folder_path: str):
    """
    Return the mean and the standard deviation of the dataset at the given path

    :param folder_path: The path of the folder containing the dataset
    :return: The mean and std of the dataset
    """

    train_set = __RoadDataset(folder_path)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=15, shuffle=False)

    return __batch_mean_and_sd(train_dataloader)


class __RoadDataset(torch.utils.data.Dataset):
    """Class representing our custom Dataset"""

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path
        self.imgs = list(sorted(listdir(folder_path)))

    def __getitem__(self, idx):
        img_path = path.join(self.folder_path, self.imgs[idx])

        img = Image.open(img_path)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        img = transform(img)

        return img

    def __len__(self):
        return len(self.imgs)


# https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html
def __batch_mean_and_sd(loader):
    """Function to calculate the mean and standard deviation given a dataloader"""
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)
    return mean, std
