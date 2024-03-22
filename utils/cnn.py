"""
Class that regroups all private and public function of our cnn models.

private functions are the ones inherent at its training
public functions are the ones such as fit, predict, and so on.
"""

from abc import ABC, abstractmethod

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from .D_Link_Net import DLinkNet
from .U_Net import UNet


class CNN(ABC):
    def __init__(self, model_name: str, device: str) -> None:

        self.model_name = model_name
        self.device = device
        self.train_set_size = 0
        self.val_set_size = 0
        self.cur_epoch = 0
        self.batch_size = 0
        self.mean = 0
        self.std = 0
        self.training_batch_number = 0

        # Set the architecture of model
        if model_name == "UNet":
            self.model = UNet().to(torch.device(self.device))
        elif model_name == "DLinkNet":
            self.model = DLinkNet().to(torch.device(self.device))
        else:
            raise AttributeError("This model has not been implemented. Please use one of the following models :\n"
                                 "UNet or DLinkNet")

        # declare evaluation metrics array
        self.train_loss_history = []
        self.train_acc_history = []
        self.lr_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.train_prf1_p_history = []
        self.train_prf1_n_history = []
        self.val_prf1_p_history = []
        self.val_prf1_n_history = []

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[Image.Image]:
        """Function to predict output from a given input

        :param images: The images for which to do the prediction
        :return: The list of predicted images
        """
        # Set the model in evaluation mode (turn-off the auto gradient computation, ...)
        self.model.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # Normalize the image to predict with the training mean and std
        images = torch.stack([transform(img) for img in images])

        # Create DataLoader with all images to predict
        images_dataset = TensorDataset(images)
        images_dataloader = DataLoader(images_dataset, self.batch_size)

        # Create transformer to transform prediction to pil image
        to_pil_img = transforms.ToPILImage()

        predictions = []
        for img in images_dataloader:
            # Move the data to the device
            img = img[0].to(self.device)
            # Compute model output
            prediction = self.model(img)
            # Force the image to be composed of zeros and ones only (threshold set to 0.5)
            prediction = prediction.clamp(min=0, max=1).round()
            # Store the prediction
            predictions.extend([to_pil_img(numpy_img) for numpy_img in prediction.cpu().detach()])

        return predictions

    @abstractmethod
    def restore_model(self, model_path: str) -> None:
        """
        Restore model with saved parameter to either continue training more,
        or to use the model to do some predictions

        :param model_path: The path where to find the model parameters to use
        """
        pass

    def print_training_stats(self, start_epoch: int = 0, end_epoch: int = None):
        """Print the training statistics

        :param start_epoch: The starting epoch to use to print the stats
        :param  end_epoch: The end epoch to use to print the stats
        """
        if end_epoch is not None:
            self._plot_training_curves(end_epoch, start_epoch)
        else:
            self._plot_training_curves(self.cur_epoch)

    def _plot_training_curves(self, num_epoch: int, start_epoch: int = 0):
        """Plot train and val accuracy and loss evolution"""

        start_train_idx = start_epoch * self.training_batch_number
        end_train_idx = num_epoch * self.training_batch_number

        n_train = len(self.train_acc_history[start_train_idx:end_train_idx])
        # space train data evenly
        t_train = start_epoch + ((num_epoch - start_epoch) * np.arange(n_train) / n_train)
        # space val data evenly
        t_val = start_epoch + np.arange(1, (num_epoch - start_epoch) + 1)

        fig, ax = plt.subplots(4, 2, figsize=(15, 12))

        # plot accuracy evolution
        ax[0][0].plot(t_train, self.train_acc_history[start_train_idx:end_train_idx], label='Train')
        ax[0][0].plot(t_val, self.val_acc_history[start_epoch:num_epoch], label='Val')
        ax[0][0].legend()
        ax[0][0].set_xlabel('Epoch')
        ax[0][0].set_ylabel('Accuracy')

        # plot loss evolution
        ax[0][1].plot(t_train, self.train_loss_history[start_train_idx:end_train_idx], label='Train')
        ax[0][1].plot(t_val, self.val_loss_history[start_epoch:num_epoch], label='Val')
        ax[0][1].legend()
        ax[0][1].set_xlabel('Epoch')
        ax[0][1].set_ylabel('Loss')

        # Check min value for the two performance metric (training)
        min_neg = np.array(self.train_prf1_n_history[start_train_idx:end_train_idx]).min()
        min_pos = np.array(self.train_prf1_p_history[start_train_idx:end_train_idx]).min()
        min_train = np.min([min_pos, min_neg])
        min_train = 0 if np.isnan(min_train) else min_train

        # plot positive train precision, recall and f1 evolution
        ax[1][0].plot(t_train, self.train_prf1_p_history[start_train_idx:end_train_idx],
                      label=['Precision positive', 'Recall positive', 'f1 positive'])
        ax[1][0].legend()
        ax[1][0].set_xlabel('Epoch')
        ax[1][0].set_ylabel('Training Positive metrics')
        ax[1][0].set_ylim(min_train)

        # plot negative train precision, recall and f1 evolution
        ax[1][1].plot(t_train, self.train_prf1_n_history[start_train_idx:end_train_idx],
                      label=['Precision negative', 'Recall negative', 'f1 negative'])
        ax[1][1].legend()
        ax[1][1].set_xlabel('Epoch')
        ax[1][1].set_ylabel('Training Negative metrics')
        ax[1][1].set_ylim(min_train)

        # Check min value for the two performance metric (validation)
        min_neg = np.array(self.val_prf1_n_history[start_epoch:num_epoch]).min()
        min_pos = np.array(self.val_prf1_p_history[start_epoch:num_epoch]).min()
        min_test = np.min([min_pos, min_neg])
        min_test = 0 if np.isnan(min_test) else min_test

        # plot positive val precision, recall and f1 evolution
        ax[2][0].plot(t_val, self.val_prf1_p_history[start_epoch:num_epoch],
                      label=['Precision positive', 'Recall positive', 'f1 positive'])
        ax[2][0].legend()
        ax[2][0].set_xlabel('Epoch')
        ax[2][0].set_ylabel('Val Positive metrics')
        ax[2][0].set_ylim(min_test)

        # plot negative val precision, recall and f1 evolution
        ax[2][1].plot(t_val, self.val_prf1_n_history[start_epoch:num_epoch],
                      label=['Precision negative', 'Recall negative', 'f1 negative'])
        ax[2][1].legend()
        ax[2][1].set_xlabel('Epoch')
        ax[2][1].set_ylabel('Val Negative metrics')
        ax[2][1].set_ylim(min_test)

        # plot learning rate evolution
        ax[3][0].plot(t_train, self.lr_history[start_train_idx:end_train_idx], label='learning rate')
        ax[3][0].legend()
        ax[3][0].set_xlabel('Epoch')
        ax[3][0].set_ylabel('Learning rate evolution')

        plt.show()
