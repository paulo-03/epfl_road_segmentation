"""Script creating the class for the Random Model"""

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from utils.helpers import concatenate_images
from utils.performance_metrics import get_performance_distribution


class RandomModel:
    def __init__(self, training_image: list[Image.Image]):
        """Initialize a model that will create randomly images with white and black pixel

        :param training_data: list of all images for train
        """
        # Transform the Image data in numpy array
        self.data = [np.array(img) / 255 for img in training_image]
        self.hist_values = []
        self.bin_edges = []
        self.training_img_number = len(training_image)

    def fit_distribution(self, granularity: int = 30):
        one_fraction = []
        for img in self.data:
            rounded_img = img.round()
            one_pixels = rounded_img.sum()
            one_fraction.append(one_pixels / img.size)

        fig, ax = plt.subplots(1, figsize=(7, 3))
        self.hist_values, self.bin_edges, _ = ax.hist(one_fraction, bins=granularity)
        ax.set_title("Distribution of road pixel's fraction per training image")
        ax.set_xlabel("fraction of ones")
        ax.set_ylabel("count")

        plt.show()

    def _compute_pred_tensor(self, img_fold, gt_fold):
        to_float = transforms.ToTensor()

        pred_img = self.predict(img_fold)
        pred_img_tensor = torch.stack([to_float(img) for img in pred_img])
        pred_gt_tensor = torch.stack([to_float(gt) for gt in gt_fold])

        return pred_img_tensor, pred_gt_tensor

    def performance_distribution(self, img_folds: list[list[Image]], gt_folds: list[list[Image]]) -> list[list[float]]:
        """
        Function that performs a performance distribution of the model, by making prediction of k fold validation
        images.

        :param img_folds: The image folds list
        :param gt_folds: The groundtruth fold list

        :return: The performance metrics of each k_fold as a list
        """
        return get_performance_distribution(img_folds, gt_folds, self._compute_pred_tensor)

    def predict(self, test_images: list[Image.Image]):
        width, height = (test_images[0].width, test_images[0].height)
        random_fracs = np.random.choice(a=self.bin_edges[1:],
                                        size=len(test_images),
                                        p=self.hist_values / self.training_img_number)
        predictions = []
        values = np.array([0, 255], dtype=np.uint8)
        for one_frac in random_fracs:
            probabilities = [1 - one_frac, one_frac]
            predictions.append(np.random.choice(values, size=(width, height), p=probabilities, ))

        # Create transformer to transform prediction to pil image
        to_pil_img = transforms.ToPILImage()

        return [to_pil_img(numpy_img) for numpy_img in predictions]

    @staticmethod
    def display_test_predictions(test_images: list[Image], predictions: list[Image], nb: int = 8):
        # Display the predictions next to the original image
        rows_number = np.ceil(nb / 4).astype(int)
        fig, ax = plt.subplots(rows_number, 4, figsize=(20, 3 * rows_number))
        for i in range(0, nb):
            ip = concatenate_images(test_images[i], predictions[i])
            ax[i // 4][i % 4].imshow(ip, cmap='Greys')

        plt.show()
