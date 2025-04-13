#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LPRNet model for license plate recognition.
Includes original LPRNet and enhanced versions with STNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    """Initialize weights for convolutional and linear layers."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.constant_(module.bias, 0)


class SmallBasicBlock(nn.Module):
    """Original small basic block from LPRNet."""
    def __init__(self, ch_in, ch_out):
        super(SmallBasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.block(x)


class SmallBasicBlockV2(nn.Module):
    """Enhanced small basic block with residual connection."""
    def __init__(self, ch_in, ch_out):
        super(SmallBasicBlockV2, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
        self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.relu = nn.ReLU()
        self.apply(init_weights)

    def forward(self, x):
        # Main path
        residual = self.block(x)
        # Shortcut path
        shortcut = self.shortcut(x)
        # Residual connection
        out = residual + shortcut
        # Activation
        out = self.relu(out)
        return out


class LPRNet(nn.Module):
    """
    License Plate Recognition Network.
    
    Supports four configurations:
    1. Original LPRNet (use_origin_block=True, add_stnet=False)
    2. LPRNet with STNet (use_origin_block=True, add_stnet=True)
    3. LPRNetPlus (use_origin_block=False, add_stnet=False)
    4. LPRNetPlus with STNet (use_origin_block=False, add_stnet=True)
    """
    def __init__(self, num_classes, lpr_max_len=8, in_channels=3, dropout_rate=0.5, 
                 use_origin_block=False, add_stnet=False):
        super(LPRNet, self).__init__()
        self.num_classes = num_classes
        self.lpr_max_len = lpr_max_len

        # Select block type
        block_module = SmallBasicBlock if use_origin_block else SmallBasicBlockV2

        # Add STNet if requested
        self.add_stnet = add_stnet
        if self.add_stnet:
            from .stnet import STNet
            self.stnet = STNet()

        # Main backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            block_module(ch_in=64, ch_out=128),  # 4
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            block_module(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            block_module(ch_in=256, ch_out=256),  # 11
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(13, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=num_classes),
            nn.ReLU(),  # 22
        )
        
        # Final output layer
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.num_classes, out_channels=self.num_classes, kernel_size=(1, 1),
                      stride=(1, 1)),
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Log probabilities for each position and class
        """
        # Apply STNet if enabled
        if self.add_stnet:
            x = self.stnet(x)

        # Extract features from important layers
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # Important feature maps
                keep_features.append(x)

        # Global context processing
        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        # Concatenate and process features
        x = torch.cat(global_context, 1)
        # Final convolution
        x = self.container(x)
        # Average across height dimension
        x = torch.mean(x, dim=2)
        
        # Permute to (batch, width, classes) and apply log_softmax
        x = x.permute(0, 2, 1)
        logits = F.log_softmax(x, dim=2)

        return logits


def build_lprnet(model_type, num_classes, lpr_max_len=8, dropout_rate=0.5):
    """
    Build an LPRNet model based on the specified type.
    
    Args:
        model_type (str): One of ['lprnet', 'lprnet_plus', 'lprnet_stnet', 'lprnet_plus_stnet']
        num_classes (int): Number of classes (characters) to recognize
        lpr_max_len (int): Maximum length of license plate
        dropout_rate (float): Dropout rate
        
    Returns:
        LPRNet: Initialized model
    """
    if model_type == 'lprnet':
        return LPRNet(num_classes=num_classes, lpr_max_len=lpr_max_len, 
                      dropout_rate=dropout_rate, use_origin_block=True, add_stnet=False)
    elif model_type == 'lprnet_plus':
        return LPRNet(num_classes=num_classes, lpr_max_len=lpr_max_len, 
                      dropout_rate=dropout_rate, use_origin_block=False, add_stnet=False)
    elif model_type == 'lprnet_stnet':
        return LPRNet(num_classes=num_classes, lpr_max_len=lpr_max_len, 
                      dropout_rate=dropout_rate, use_origin_block=True, add_stnet=True)
    elif model_type == 'lprnet_plus_stnet':
        return LPRNet(num_classes=num_classes, lpr_max_len=lpr_max_len, 
                      dropout_rate=dropout_rate, use_origin_block=False, add_stnet=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test all model variants
    num_classes = 78
    input_tensor = torch.randn(4, 3, 24, 94)
    
    model_types = ['lprnet', 'lprnet_plus', 'lprnet_stnet', 'lprnet_plus_stnet']
    
    for model_type in model_types:
        print(f"\nTesting {model_type}:")
        model = build_lprnet(model_type, num_classes)
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}") 