"""
Helper functions used to increase the samples/images in the training data set.
"""

from os import listdir, path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2


def __initiate_augmented_folder(training_folder: str, augmented_image_folder: str,
                                augmented_groundtruth_folder: str, split_images: bool):
    """
    Function that initiate the folder where to save the augmented images and store the original images

    :param training_folder: the folder containing the images and ground truth
    :param augmented_image_folder: the folder where to store the augmented images
    :param augmented_groundtruth_folder: the folder where to store the augmented groundtruth
    :param split_images: whether the images are split or not
                         (if they are already split, we won't perform random cropping)

    """
    train_set = RoadDatasetAugmentation(root=training_folder, split_images=split_images, perform_transformations=False)
    train_dataloader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)

    to_pil = v2.ToPILImage()

    for idx, (img, gt) in enumerate(train_dataloader):
        # get transformed image from dataloader
        img = to_pil(img.squeeze(0))
        # get transformed ground truth from dataloader
        gt = to_pil(gt.squeeze(0))

        # save image to disk
        img.save(path.join(training_folder, augmented_image_folder, f'satImage_{idx + 1:06d}.png'))
        # save gt to disk
        gt.save(path.join(training_folder, augmented_groundtruth_folder, f'satImage_{idx + 1:06d}.png'))


def data_augmentation(nb: int, training_folder: str, augmented_image_folder: str, augmented_groundtruth_folder: str,
                      add_base_images: bool = True, split_images: bool = False):
    """
    Function that augments the dataset with some transformation applied to images, and save them to disk

    :param nb: The number of time to pass the whole dataset into transformation process
               (i.e. nb=10 goes from 100 to 1100 images, by adding 1000 new images to the 100 existing ones)
    :param training_folder: the folder containing the images and groundtruth
    :param augmented_image_folder: the folder where to store the augmented images
    :param augmented_groundtruth_folder: the folder where to store the augmented groundtruth
    :param add_base_images: whether to add the base images to the augmented dataset or not
    :param split_images: whether the images are split or not
    """

    train_set = RoadDatasetAugmentation(root=training_folder, split_images=split_images, perform_transformations=True)
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=False)
    nb_images = len(train_dataloader)

    start_output_idx = 1
    if add_base_images:
        # add base images to augmented dataset
        __initiate_augmented_folder(training_folder, augmented_image_folder, augmented_groundtruth_folder, split_images)
        start_output_idx += nb_images

    to_pil = v2.ToPILImage()

    for i in range(nb):
        for idx, (img, gt) in enumerate(train_dataloader):
            # get transformed image from dataloader
            img = to_pil(img.squeeze(0))
            # get transformed ground truth from dataloader
            gt = to_pil(gt.squeeze(0))

            # compute the image index
            img_idx = start_output_idx + i * nb_images + idx
            # save image to disk
            img.save(path.join(training_folder, augmented_image_folder, f'satImage_{img_idx:06d}.png'))

            # save gt to disk
            gt.save(
                path.join(training_folder, augmented_groundtruth_folder, f'satImage_{img_idx:06d}.png'))


class RoadDatasetAugmentation(Dataset):
    """Class representing our custom road Dataset used to perform data augmentation"""

    def __init__(self, root, split_images, perform_transformations):
        self.root = root
        self.perform_transformations = perform_transformations
        self.split_images = split_images
        self.imgs = list(sorted(listdir(path.join(root, 'images_split' if split_images else 'images'))))
        self.gt = list(sorted(listdir(path.join(root, 'groundtruth_split' if split_images else 'groundtruth'))))

        self.transform_base = v2.Compose([
            # only perform random crop if images are not already split
            v2.Identity() if split_images else v2.RandomResizedCrop(size=400, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ])

        # apply a set of transformations with a certain probability
        self.transform_rotation = v2.RandomApply([
            v2.RandomRotation(360)
        ], p=0.5)

        self.transform_blur = v2.RandomApply([
            v2.GaussianBlur(5)
        ], p=0.5)

        self.transform_brightness = v2.RandomApply([
            v2.ColorJitter(brightness=(0.5, 1.5))
        ], p=0.5)

        self.transform_contrast = v2.RandomApply([
            v2.ColorJitter(contrast=(0.5, 1.5))
        ], p=0.25)

        self.transform_saturation = v2.RandomApply([
            v2.ColorJitter(saturation=(0.5, 1.5))
        ], p=0.25)

    def __getitem__(self, idx):
        # load image and ground truth
        img_path = path.join(self.root, 'images_split' if self.split_images else 'images', self.imgs[idx])
        gt_path = path.join(self.root, 'groundtruth_split' if self.split_images else 'groundtruth', self.gt[idx])

        img = read_image(img_path, ImageReadMode.RGB)
        gt = read_image(gt_path, ImageReadMode.RGB)

        # concatenate both images, so that transformation is the same for img and gt
        both = torch.cat((img.unsqueeze(0), gt.unsqueeze(0)), 0)

        if self.perform_transformations:
            both = self.transform_base(both)

            both = self.transform_rotation(both)

        img, gt = both[0], both[1]

        # transformations to perform only on image, since the ground truth stays the same
        if self.perform_transformations:
            img = self.transform_blur(img)
            img = self.transform_brightness(img)
            img = self.transform_contrast(img)
            img = self.transform_saturation(img)

        # gt converted back to grayscale as it had to be loaded as rgb to concatenate for transformation
        return img, v2.Grayscale()(gt)

    def __len__(self):
        return len(self.imgs)
