"""Definition of U-Net class"""

from math import floor, ceil

import torch
import torch.nn as nn
import torch.nn.functional as f


class UNet(nn.Module):
    """Architecture of UNet model as explained in https://arxiv.org/pdf/1505.04597.pdf"""

    def __init__(self):
        super().__init__()

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.double_conv1 = self.double_conv(3, 64)
        self.double_conv2 = self.double_conv(64, 128)
        self.double_conv3 = self.double_conv(128, 256)
        self.double_conv4 = self.double_conv(256, 512)

        self.center = self.double_conv(512, 1024)

        self.double_conv5 = self.double_conv(1024, 512)
        self.double_conv6 = self.double_conv(512, 256)
        self.double_conv7 = self.double_conv(256, 128)
        self.double_conv8 = self.double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    @staticmethod
    def double_conv(in_channels, out_channels):
        """Performs two convolutions"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def crop_concat(tens_left, tens_right):
        """To be sure that tensors are correctly concatenated"""
        diffy = tens_right.size()[2] - tens_left.size()[2]
        diffx = tens_right.size()[3] - tens_left.size()[3]

        tens_left = f.pad(input=tens_left,
                          pad=[floor(diffx / 2), ceil(diffx / 2),
                               floor(diffy / 2), ceil(diffx / 2)])

        return torch.cat([tens_left, tens_right], dim=1)

    def forward(self, x):
        """Performs a forward pass"""
        down_conv1 = self.double_conv1(x)
        down_pool1 = self.down(down_conv1)
        down_conv2 = self.double_conv2(down_pool1)
        down_pool2 = self.down(down_conv2)
        down_conv3 = self.double_conv3(down_pool2)
        down_pool3 = self.down(down_conv3)
        down_conv4 = self.double_conv4(down_pool3)
        down_pool4 = self.down(down_conv4)

        center = self.center(down_pool4)

        up1 = self.up1(center)
        up_conv1 = self.double_conv5(self.crop_concat(down_conv4, up1))
        up2 = self.up2(up_conv1)
        up_conv2 = self.double_conv6(self.crop_concat(down_conv3, up2))
        up3 = self.up3(up_conv2)
        up_conv3 = self.double_conv7(self.crop_concat(down_conv2, up3))
        up4 = self.up4(up_conv3)
        up_conv4 = self.double_conv8(self.crop_concat(down_conv1, up4))

        final = self.final_conv(up_conv4)

        return final
