""" Full assembly of the parts to form the complete network """

from torch import Tensor, nn

from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels: int, out_channels: int, bilinear: bool = False):

        super(UNet, self).__init__()

        self.n_channels: int = n_channels
        self.n_classes: int = out_channels
        self.bilinear: bool = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, n_channels * 2)

        self.down1 = Down(n_channels * 2, n_channels * 4)
        self.down2 = Down(n_channels * 4, n_channels * 8)
        self.down3 = Down(n_channels * 8, n_channels * 16)
        self.down4 = Down(n_channels * 16, n_channels * 32 // factor)

        self.up1 = Up(n_channels * 32, n_channels * 16 // factor, bilinear)
        self.up2 = Up(n_channels * 16, n_channels * 8 // factor, bilinear)
        self.up3 = Up(n_channels * 8, n_channels * 4 // factor, bilinear)
        self.up4 = Up(n_channels * 4, n_channels * 2 // factor, bilinear)

        self.outc = OutConv(n_channels * 2, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1: Tensor = self.inc(x)
        x2: Tensor = self.down1(x1)
        x3: Tensor = self.down2(x2)
        x4: Tensor = self.down3(x3)
        x5: Tensor = self.down4(x4)
        x: Tensor = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits: Tensor = self.outc(x)
        return logits
