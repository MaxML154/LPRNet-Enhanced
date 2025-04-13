#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loss functions for license plate recognition.
"""

import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    """
    Connectionist Temporal Classification Loss.
    Used for sequence recognition without explicit alignment.
    """
    def __init__(self, blank_label=0, reduction='mean'):
        """
        Initialize CTC loss.
        
        Args:
            blank_label: Index of blank label
            reduction: Type of reduction ('none', 'mean', 'sum')
        """
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_label, reduction=reduction, zero_infinity=True)
        
    def forward(self, preds, targets, pred_lengths, target_lengths):
        """
        Forward pass.
        
        Args:
            preds: Predictions from model (N, T, C)
            targets: Target labels
            pred_lengths: Lengths of predictions
            target_lengths: Lengths of targets
            
        Returns:
            CTC loss value
        """
        # Predictions must be in the shape of (T, N, C) for CTCLoss
        preds_log_softmax = preds.permute(1, 0, 2)
        
        # Compute loss
        loss = self.ctc_loss(preds_log_softmax, targets, pred_lengths, target_lengths)
        
        return loss


if __name__ == '__main__':
    # Test CTCLoss
    # Set up random inputs
    batch_size = 4
    max_length = 10
    num_classes = 26
    
    # Random log probabilities tensor (N, T, C)
    logits = torch.randn(batch_size, max_length, num_classes).log_softmax(2)
    
    # Random target indices
    targets = torch.randint(1, num_classes, (batch_size, 5))
    
    # Random lengths
    pred_lengths = torch.full((batch_size,), max_length, dtype=torch.long)
    target_lengths = torch.randint(2, 6, (batch_size,), dtype=torch.long)
    
    # Flatten targets
    targets_flat = torch.cat([targets[i, :target_lengths[i]] for i in range(batch_size)])
    
    # Create loss
    ctc_loss = CTCLoss(blank_label=0)
    
    # Calculate loss
    loss = ctc_loss(logits, targets_flat, pred_lengths, target_lengths)
    
    print(f"Loss shape: {loss.shape}")
    print(f"Loss value: {loss.item()}") 