from torch import nn


class BaseREBNCONV(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dirate: int = None,
    ):
        super().__init__()
        self._in_ch = in_ch
        self._out_ch = out_ch
        self._stride = stride
        self._kernel_size = kernel_size
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

        self.conv = nn.Conv2d(
            self._in_ch,
            self._out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=1 * dirate if dirate else padding,
            dilation=1 * dirate if dirate else padding,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(self._out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.rl(x)


class REBNCONV(BaseREBNCONV):
    def __init__(
        self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1, stride: int = 1
    ):
        super().__init__(in_ch, out_ch, dirate=dirate, stride=stride)

        self.conv_s1 = self.conv
        delattr(self, "conv")

        self.bn_s1 = self.bn
        delattr(self, "bn")

        self.relu_s1 = self.rl
        delattr(self, "rl")

    def forward(self, x):
        x = self.conv_s1(x)
        x = self.bn_s1(x)
        return self.relu_s1(x)
