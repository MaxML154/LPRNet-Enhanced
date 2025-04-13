#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STNet (Spatial Transformer Network) module for license plate image transformation.
This helps correct image distortions and make recognition more robust.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STNet(nn.Module):
    """
    Spatial Transformer Network for correcting license plate distortions.
    Adapted from the implementation by Liu Junkai and Chen Ang, UESTC.
    """

    def __init__(self):
        super(STNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True)
        )
        
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 14 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        """
        Apply spatial transformation to the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Transformed tensor of the same shape as input
        """
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 14 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Generate grid for sampling
        # Using align_corners=False as per PyTorch recommendations
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        
        # Sample the input at grid points
        x = F.grid_sample(x, grid, align_corners=False)

        return x


if __name__ == '__main__':
    # Simple test
    model = STNet()
    
    # Create random input with dimensions matching license plate images
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 24, 94)
    
    # Pass through model
    output = model(input_tensor)
    
    # Print shapes to verify
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}") 