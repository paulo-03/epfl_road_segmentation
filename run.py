"""
Python script to easily train one of the two implemented Convolutional Neural Networks.
"""

import os
import random
import shutil
from datetime import datetime

from utils.cnn_trainer import CnnTrainer
from utils.data_augmentation import *
from utils.helpers import get_dataset_mean_std, load_images, combine_predictions
from utils.image_splitting import *
from utils.mask_to_submission import *


def training_model(model_name: str,
                   num_epochs: int,
                   batch_size: int,
                   patch_size: int,
                   device: str = 'cuda',
                   seed: int = 42,
                   test_size: float = 0.15,
                   learning_rate: float = 1e-3,
                   weight_decay: float = 1e-2,
                   augmentation_factor: int = 0,
                   root_training_folder: str = 'dataset/training',
                   augmented_image_folder: str = 'images_augmented',
                   augmented_groundtruth_folder: str = 'groundtruth_augmented',
                   model_saving_root_folder: str = 'models',
                   model_saved_file: str = None,
                   root_testing_folder: str = 'dataset/test_set_images',
                   root_predictions: str = 'predictions',
                   output_filename: str = 'submission.csv',
                   foreground_threshold: float = 0.4):
    """Train the selected model according to the given hyperparameters, predicts outputs
    and generate the csv file for submission on AIcrowd.

    :param model_name: Name of the chosen model to train, one of ['UNet', 'DLinkNet']
    :param num_epochs: Number of training epochs
    :param batch_size: Size of each training and validation batch
    :param split_images: Set this boolean to true to split images in patches
    :param patch_size: Size of the (square) patches
    :param device: Choose in which device to train the model
    :param seed: Set the seed for reproducibility reasons
    :param test_size: Fraction of validation dataset to use for testing
    :param learning_rate: Set the initial learning rate
    :param weight_decay: Set weight decay
    :param augmentation_factor: How many times to augment the training data set
    :param root_training_folder: Path to the training folder containing data for training
    :param augmented_image_folder: Name of (augmented) training image folder
    :param augmented_groundtruth_folder: Name of (augmented) training groundtruth folder
    :param model_saving_root_folder: Path where to save model metrics (weights, evolution of loss, ...)
    :param model_saved_file: The file where the model has been saved along with the folder where it has been saved
    :param root_testing_folder: Path to the testing folder containing data to predict
    :param root_predictions: Path where to save the prediction images
    :param output_filename: Name of the csv file to generate
    :param foreground_threshold: Threshold to consider a 16 by 16 patch as road or not
    """
    # Set all seeds for reproducibility (see
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    image_augmented_folder = path.join(root_training_folder, augmented_image_folder)
    groundtruth_augmented_folder = path.join(root_training_folder, augmented_groundtruth_folder)
    split_images_folder = path.join(root_training_folder, 'images_split')
    split_gt_folder = path.join(root_training_folder, 'groundtruth_split')

    # Cleanup training folder
    shutil.rmtree(image_augmented_folder, ignore_errors=True)
    shutil.rmtree(groundtruth_augmented_folder, ignore_errors=True)
    os.makedirs(image_augmented_folder)
    os.makedirs(groundtruth_augmented_folder)

    shutil.rmtree(split_images_folder, ignore_errors=True)
    shutil.rmtree(split_gt_folder, ignore_errors=True)

    # create model saving folder and prediction folder
    os.makedirs(model_saving_root_folder, exist_ok=True)
    os.makedirs(root_predictions, exist_ok=True)

    # Format the current date to avoid path problems
    date_str = str(datetime.now()) \
        .replace(':', 'h', 1) \
        .replace(':', 'm', 1) \
        .replace('.', 's', 1)

    # Create folder for current model weights and predictions
    model_save_folder = path.join(model_saving_root_folder, date_str)
    predictions_folder = path.join(root_predictions, date_str)
    os.makedirs(model_save_folder)
    os.makedirs(predictions_folder)

    print('Folder preparation done')

    os.makedirs(split_images_folder)
    os.makedirs(split_gt_folder)

    # split images and groundtruth
    split_training_images(root_training_folder, split_images_folder, split_gt_folder, patch_size)

    print('Image splitting done')

    # perform data augmentation
    data_augmentation(nb=augmentation_factor, training_folder=root_training_folder,
                      augmented_image_folder=augmented_image_folder,
                      augmented_groundtruth_folder=augmented_groundtruth_folder, split_images=True)

    print('Data augmentation done')

    dataset_mean, dataset_std = get_dataset_mean_std(image_augmented_folder)

    # Set the data loader and optimizer parameters

    data_kwargs_pred = dict(
        batch_size=batch_size,
        mean=dataset_mean,
        std=dataset_std,
    )

    data_kwargs = data_kwargs_pred | dict(
        img_folder=image_augmented_folder,
        gt_folder=groundtruth_augmented_folder
    )

    optimizer_kwargs = dict(
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Initialize the model and its environment
    cnn = CnnTrainer(
        model_name=model_name,
        data_kwargs=data_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        num_epochs=num_epochs,
        model_saving_path=model_save_folder,
        test_size=test_size,
        device=device)

    # Restore previous model to continue training if needed
    if model_saved_file is not None:
        print('Restoring previous model')

        cnn.restore_model(model_path=path.join(model_saving_root_folder, model_saved_file))

    print('Begin Training')

    # Fit the model
    cnn.fit()

    print('Begin Predictions')

    # Load test images and split them into patches
    test_patches, n_test_images, w, h, pad = split_testing_images(root_testing_folder, patch_size)

    # Create folders to store predictions
    for i in range(37, 86):
        os.makedirs(path.join(predictions_folder, f'prediction_{i}'))

    for i in range(37, 86):
        # Restore model saves
        cnn.restore_model(path.join(model_save_folder, f'training_save_epoch_{i}.tar'))
        # Compute predictions
        test_patches_predictions = cnn.predict(images=test_patches)
        # Assemble patches into images and save them
        combine_patches(test_patches_predictions, patch_size, n_test_images, w, h, pad,
                        path.join(predictions_folder, f'prediction_{i}'))

    images_to_combine = [load_images(path.join(predictions_folder, f'prediction_{i}')) for i in range(37, 86)]

    combined_predictions = combine_predictions(images_to_combine, 0)

    os.makedirs(path.join(predictions_folder, 'combined_prediction'))
    for idx, comb_pred in enumerate(combined_predictions):
        comb_pred.save(path.join(predictions_folder, 'combined_prediction', f'pred_{idx+1:03d}.png'))

    print('Generating csv file')

    predictions_filenames = [f'{predictions_folder}/combined_prediction/pred_{i:03d}.png' for i in range(1, 51)]
    masks_to_submission(output_filename, predictions_filenames, foreground_threshold)

    print(f'Everything done! Find the output in {output_filename}')


if __name__ == '__main__':
    training_model(
        model_name='UNet',
        num_epochs=100,
        batch_size=500,
        patch_size=96,
        device='cuda',
        seed=42,
        test_size=0.15,
        learning_rate=1e-3,
        weight_decay=1e-2,
        augmentation_factor=99,
        root_training_folder='dataset/training',
        augmented_image_folder='images_augmented_96x96',
        augmented_groundtruth_folder='groundtruth_augmented_96x96',
        model_saving_root_folder='models',
        model_saved_file=None,
        root_testing_folder='dataset/test_set_images',
        root_predictions='predictions',
        output_filename='submission.csv',
        foreground_threshold=0.3
    )
