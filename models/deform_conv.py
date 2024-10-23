import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # Offset convolution (2 coordinates per pixel per kernel point)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1
        )

        # Main convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Initialize offsets to 0
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        # Get batch size and spatial dimensions
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)

        #print("Input shape:", x.shape)

        # Get offsets for each position
        offsets = self.offset_conv(x)
        #print("Offset shape:", offsets.shape)

        # Create sampling grid for deformable convolution
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=torch.float32),
            torch.arange(width, device=x.device, dtype=torch.float32)
        )
        y_grid = y_grid.contiguous().view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
        x_grid = x_grid.contiguous().view(1, 1, height, width).repeat(batch_size, 1, 1, 1)

        # Split offsets into x and y
        offsets = offsets.view(batch_size, -1, 2, height, width)

        # Apply offsets to create sampling locations
        x_offset = x_grid + offsets[:, :, 0].sum(dim=1, keepdim=True)
        y_offset = y_grid + offsets[:, :, 1].sum(dim=1, keepdim=True)

        # Normalize coordinates to [-1, 1] for grid_sample
        x_offset = 2.0 * x_offset / (width - 1) - 1.0
        y_offset = 2.0 * y_offset / (height - 1) - 1.0

        # Stack coordinates for grid_sample
        grid = torch.stack([x_offset, y_offset], dim=-1)
        #print("Grid shape:", grid.shape)

        # Sample the input feature map
        x_deformed = F.grid_sample(
            x,
            grid.squeeze(1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # Apply convolution
        return self.conv(x_deformed)
