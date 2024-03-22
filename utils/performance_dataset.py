import math
import os

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from utils.data_augmentation import RoadDatasetAugmentation
from utils.image_splitting import split_training_images


def create_performance_dataset(image_per_fold, k_folds, perform_transformation=True, split_image=True) -> (list, list):
    """Function to create the dataset to compute the performance of a model

    :param image_per_fold: Number of image per fold
    :param k_folds: The number of fold to perform
    :param split_image: Whether to split the image
    :param perform_transformation: Whether to perform the transformation or not
    :return:
    """
    root_training = 'dataset/training'
    _create_split_dataset(root_training)

    performance_dataset = RoadDatasetAugmentation(root_training, split_image, perform_transformation)
    performance_loader = DataLoader(performance_dataset, batch_size=1, shuffle=False)

    performance_image = []
    performance_gt = []
    # Create the validation DataSet
    nb_split_images = 1600
    # number of iterations
    nb_iteration = math.ceil((image_per_fold * k_folds) / nb_split_images)

    to_pil = v2.ToPILImage()
    # Create performance image set
    for i in range(nb_iteration):
        for img, gt in performance_loader:
            performance_image.append(to_pil(img.squeeze(0)))
            performance_gt.append(to_pil(gt.squeeze(0)))

    k_fold_indices = _k_indices(k_folds, image_per_fold, len(performance_image))

    # Create the random folds
    img_folds = [[performance_image[idx] for idx in indices] for indices in k_fold_indices]
    gt_folds = [[performance_gt[idx] for idx in indices] for indices in k_fold_indices]

    return img_folds, gt_folds


def display_performance_distribution(model_names: list[str], metric_names: list[str],
                                     performance_values: list[list[float]], fig_size: tuple[int, int]) -> None:
    """Display box plots to compare models performances

    :param model_names: The list of the names of the model for which we have performances
    :param metric_names: the name of the metrics for which we want to display
    :param performance_values: The performance values of each model
    :param fig_size: The size of the figure

    """
    # Create a dataframe to ease the use of seaborn
    data = {
        'metric': [],
        'values': [],
        'model': []
    }

    # Fill the dictionary
    for idx_name, model_name in enumerate(model_names):
        for idx_value, metric_name in enumerate(metric_names):
            values = performance_values[idx_name]
            for value in values[idx_value]:
                data['metric'].append(metric_name)
                data['values'].append(value)
                data['model'].append(model_name)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Display results
    fig, ax = plt.subplots(figsize=fig_size)
    sn.boxplot(data=df, x="metric", y="values", hue="model", ax=ax)

    plt.show()


def _create_split_dataset(root_training):
    # if split images folder doesn't exist, create it and split image
    split_image_folder = f'{root_training}/images_split'
    split_groundtruth_folder = f'{root_training}/groundtruth_split'
    if not os.path.exists(split_image_folder):
        os.makedirs(split_image_folder, exist_ok=True)
        os.makedirs(split_groundtruth_folder, exist_ok=True)
        split_training_images(root_training, f'{root_training}/images_split', f'{root_training}/groundtruth_split',
                              96)


def _k_indices(k_folds: int, image_per_fold: int, image_number: int) -> list[list[int]]:
    """Description"""
    return [np.random.choice(range(0, image_number),
                             size=image_per_fold,
                             replace=True
                             ).tolist() for _ in range(k_folds)]
