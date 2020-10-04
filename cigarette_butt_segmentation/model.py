from torch import nn
import torch


class UNet(nn.Module):
    """
    UNet implementation from the article https://arxiv.org/abs/1505.04597
    """
    def __init__(self, size=256):
        super().__init__()

        def get_conv_block(inp, out):
            """
            Convolution block of UNet. Conv 3x3 -> ReLU -> Conv 3x3 -> ReLU
            :param inp: Number of input channels
            :param out: Number of output channels
            :return:
            """
            block = nn.Sequential(
                nn.Conv2d(in_channels=inp,
                          out_channels=out,
                          kernel_size=3,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out,
                          out_channels=out,
                          kernel_size=3,
                          padding=1),
                nn.ReLU()
            )
            return block

        def get_dec_block(inp):
            """
            Decoder block of UNet. Conv 3x3 -> ReLU -> Conv 3x3 -> ReLU
            :param inp: Number of input channels
            :return: Reduced by 2 number of channels
            """
            block = nn.Sequential(
                nn.Conv2d(in_channels=inp,
                          out_channels=inp // 2,
                          kernel_size=3,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=inp // 2,
                          out_channels=inp // 2,
                          kernel_size=3,
                          padding=1),
                nn.ReLU()
            )
            return block

        def get_up_block(inp, out_size):
            """
            UNet upsampling block. Upsample -> Conv 2x2 -> ReLU
            :param inp: Number of input channels
            :param out_size: Output size
            :return:
            """
            block = nn.Sequential(
                nn.Upsample(out_size),
                nn.Conv2d(in_channels=inp,
                          out_channels=inp // 2,
                          kernel_size=3,
                          padding=1),
                nn.ReLU()
            )
            return block

        # encoder
        self.enc_conv0 = get_conv_block(3, 32)
        self.pool0 = nn.MaxPool2d(2)
        self.enc_conv1 = get_conv_block(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = get_conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc_conv3 = get_conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck_conv = get_conv_block(256, 512)

        # decoder
        self.upsample0 = get_up_block(512, size // 8)
        self.dec_conv0 = get_dec_block(512)
        self.upsample1 = get_up_block(256, size // 4)
        self.dec_conv1 = get_dec_block(256)
        self.upsample2 = get_up_block(128, size // 2)
        self.dec_conv2 = get_dec_block(128)
        self.upsample3 = get_up_block(64, size)
        self.dec_conv3 = get_dec_block(64)
        self.out_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck_conv(self.pool3(e3))

        # decoder
        d0 = self.dec_conv0(torch.cat([e3, self.upsample0(b)], dim=1))
        d1 = self.dec_conv1(torch.cat([e2, self.upsample1(d0)], dim=1))
        d2 = self.dec_conv2(torch.cat([e1, self.upsample2(d1)], dim=1))
        d3 = self.out_conv(self.dec_conv3(torch.cat([e0, self.upsample3(d2)], dim=1)))
        return d3
