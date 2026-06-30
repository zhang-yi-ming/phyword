import math

import torch.nn as nn

__all__ = ["StackedDownsample2d", "StackedUpsample2d"]


def _check_channels(channels):
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        raise ValueError("channels must be a positive integer")


def _check_square_power_of_2(height, width):
    if height != width:
        raise ValueError("height and width must be equal")
    if height <= 0 or height & (height - 1) != 0:
        raise ValueError("height and width must be positive powers of 2")
    return int(math.log2(height))


class StackedDownsample2d(nn.Module):
    """Stacked 2D downsampling to [B, output_channels, 1, 1].

    Input shape: [B, C, H, W].
    H and W must be equal powers of 2. Each downsample layer doubles channels
    and halves H/W, then a final 1x1 conv projects to output_channels.
    """

    def __init__(self, channels=None, output_channels=2048, spatial_size=None):
        super().__init__()
        if channels is not None:
            _check_channels(channels)
        _check_channels(output_channels)

        self.channels = channels
        self.output_channels = output_channels
        self.spatial_size = None
        self.num_layers = None
        self.net = None

        if spatial_size is not None:
            if channels is None:
                raise ValueError("channels must be set when spatial_size is set")
            num_layers = _check_square_power_of_2(spatial_size, spatial_size)
            self._build(channels, spatial_size, num_layers, None, None)

    def _build(self, channels, spatial_size, num_layers, device, dtype):
        layers = []
        for i in range(num_layers):
            in_channels = channels * (2**i)
            out_channels = channels * (2 ** (i + 1))
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.SiLU(),
                ]
            )
        layers.append(
            nn.Conv2d(
                channels * (2**num_layers),
                self.output_channels,
                kernel_size=1,
            )
        )

        self.channels = channels
        self.spatial_size = spatial_size
        self.num_layers = num_layers
        self.net = nn.Sequential(*layers)
        if device is not None or dtype is not None:
            self.net = self.net.to(device=device, dtype=dtype)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("input must have shape [B, C, H, W]")
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        num_layers = _check_square_power_of_2(height, width)

        if self.channels is not None and channels != self.channels:
            raise ValueError(f"expected {self.channels} input channels, got {channels}")

        if self.net is None:
            self._build(channels, height, num_layers, x.device, x.dtype)
        elif height != self.spatial_size or num_layers != self.num_layers:
            raise ValueError(
                f"expected spatial size {self.spatial_size}x{self.spatial_size}, "
                f"got {height}x{width}"
            )

        return self.net(x)


class StackedUpsample2d(nn.Module):
    """Stacked 2D upsampling from [B, input_channels, 1, 1].

    The first 1x1 conv projects input_channels back to the channel count before
    downsample projection, then each upsample layer halves channels and doubles H/W.
    """

    def __init__(self, channels, spatial_size, input_channels=2048):
        super().__init__()
        _check_channels(channels)
        _check_channels(input_channels)
        num_layers = _check_square_power_of_2(spatial_size, spatial_size)
        expanded_channels = channels * (2**num_layers)

        layers = []
        layers.append(
            nn.Conv2d(
                input_channels,
                expanded_channels,
                kernel_size=1,
            )
        )
        for i in range(num_layers):
            in_channels = expanded_channels // (2**i)
            out_channels = expanded_channels // (2 ** (i + 1))
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0,
                    ),
                    nn.SiLU(),
                ]
            )

        self.channels = channels
        self.spatial_size = spatial_size
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("input must have shape [B, C, H, W]")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[2] != 1 or x.shape[3] != 1:
            raise ValueError("input spatial size must be 1x1")
        return self.net(x)
