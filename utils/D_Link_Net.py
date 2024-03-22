"""Definition of the D-Link"""

import torch.nn as nn


class DLinkNet(nn.Module):
    """Architecture of DLink Net model"""

    def __init__(self):
        super().__init__()

        self.init = self.init_block(3, 64)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_res_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_res_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_res_4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.res1 = self.res_block(64)
        self.res2 = self.res_block(128)
        self.res3 = self.res_block(256)
        self.res4 = self.res_block(512)

        self.dilation_1 = self.dilation_conv(512, padding=1, dilation=1)
        self.dilation_2 = self.dilation_conv(512, padding=2, dilation=2)
        self.dilation_3 = self.dilation_conv(512, padding=4, dilation=4)
        self.dilation_4 = self.dilation_conv(512, padding=8, dilation=8)

        self.c4 = self.c_block(512, 256)
        self.c3 = self.c_block(256, 128)
        self.c2 = self.c_block(128, 64)
        self.c1 = self.c_block(64, 32)

        self.final_conv = self.final(32, 1)

    @staticmethod
    def init_block(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def res_block(channel):
        return nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def dilation_conv(channel, padding, dilation):
        return nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def c_block(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def final(in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Initial convolution
        init = self.init(x)  # shape : (48, 48, 64)

        # First res block
        down_1 = self.down(init)  # shape : (24, 24, 64)

        res_block_3_1 = down_1.clone()
        for _ in range(3):
            res_block_3_1 += self.res1(res_block_3_1.clone())  # shape : (24, 24, 64)

        # Second res block
        down_2 = self.down(res_block_3_1)  # shape : (12, 12, 64)
        down_2 = self.conv_res_2(down_2)  # shape : (12, 12, 128)
        res_block_4 = down_2.clone()
        for _ in range(4):
            res_block_4 += self.res2(res_block_4.clone())  # shape : (12, 12, 128)

        # Third res block
        down_3 = self.down(res_block_4)  # shape : (6, 6, 128)
        down_3 = self.conv_res_3(down_3)  # shape : (6, 6, 256)
        res_block_6 = down_3.clone()
        for _ in range(6):
            res_block_6 += self.res3(res_block_6.clone())  # shape : (6, 6, 256)

        # Fourth res block
        down_4 = self.down(res_block_6)  # shape : (3, 3, 256)
        down_4 = self.conv_res_4(down_4)  # shape : (3, 3, 512)
        res_block_3_2 = down_4.clone()
        for _ in range(3):
            res_block_3_2 += self.res4(res_block_3_2.clone())  # shape : (3, 3, 512)

        # Dilation blocks
        dilation_1 = self.dilation_1(res_block_3_2)
        dilation_2 = self.dilation_2(dilation_1)
        dilation_3 = self.dilation_3(dilation_2)
        dilation_4 = self.dilation_4(dilation_3)  # shape : (3, 3, 512)

        # C blocks
        c5 = res_block_3_2 + dilation_1 + dilation_2 + dilation_3 + dilation_4  # shape : (3, 3, 512)
        c4 = res_block_6 + self.c4(c5)  # shape : (6, 6, 256)
        c3 = res_block_4 + self.c3(c4)  # shape : (12, 12, 128)
        c2 = res_block_3_1 + self.c2(c3)  # shape : (24, 24, 64)
        c1 = self.c1(c2)  # shape : (48, 48, 32)

        # Final transformation
        final = self.final_conv(c1)  # shape : (96, 96, 1)

        return final
