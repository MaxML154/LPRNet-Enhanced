#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training logger module for LPRNet-Enhanced.
Handles progress display, metrics logging, and visualization.
"""

import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


class Logger:
    """
    Logger for training process.
    Handles progress display, metrics tracking, and visualization.
    """
    def __init__(self, log_interval=10, bar_length=30):
        """
        Initialize logger.
        
        Args:
            log_interval: How often to log metrics (batches)
            bar_length: Length of progress bar
        """
        self.log_interval = log_interval
        self.bar_length = bar_length
        self.start_time = time.time()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def _format_time(self, seconds):
        """Format time in human-readable format."""
        return str(datetime.timedelta(seconds=int(seconds)))
        
    def log_batch(self, epoch, batch_idx, num_batches, batch_time, batch_size, losses, metrics, lr=None):
        """
        Log information for a single batch.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            num_batches: Total number of batches
            batch_time: Time taken for current batch
            batch_size: Batch size
            losses: Dictionary of loss values
            metrics: Dictionary of metric values
            lr: Current learning rate
        """
        if (batch_idx + 1) % self.log_interval != 0:
            return
            
        # Calculate progress and time stats
        progress = (batch_idx + 1) / num_batches
        progress_bar = '=' * int(progress * self.bar_length) + '>' + ' ' * (self.bar_length - int(progress * self.bar_length) - 1)
        
        # Calculate time statistics
        elapsed_time = time.time() - self.start_time
        iter_per_sec = (batch_idx + 1) / elapsed_time
        eta = (num_batches - batch_idx - 1) / iter_per_sec if iter_per_sec > 0 else 0
        
        # Format information
        loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in losses.items()])
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        lr_str = f", lr: {lr:.6f}" if lr is not None else ""
        
        # Create the log line
        log_str = f"Epoch {epoch} Batch {batch_idx+1}/{num_batches} [{progress_bar}] {batch_idx+1}/{num_batches} "
        log_str += f"[{self._format_time(elapsed_time)}<{self._format_time(eta)}, {iter_per_sec:.2f} it/s] "
        log_str += f"{loss_str}, {metrics_str}{lr_str}"
        
        # Print log line
        print(log_str, end='\r')
        sys.stdout.flush()
        
    def log_epoch(self, epoch, train_losses, train_metrics, val_losses=None, val_metrics=None):
        """
        Log information for an entire epoch.
        
        Args:
            epoch: Current epoch
            train_losses: Dictionary of training loss values
            train_metrics: Dictionary of training metric values
            val_losses: Dictionary of validation loss values
            val_metrics: Dictionary of validation metric values
        """
        # Create the log line
        log_str = f"\nEpoch {epoch} completed. "
        
        # Add training information
        train_loss_str = ', '.join([f"Train {k}: {v:.4f}" for k, v in train_losses.items()])
        train_metrics_str = ', '.join([f"Train {k}: {v:.4f}" for k, v in train_metrics.items()])
        log_str += f"{train_loss_str}, {train_metrics_str}"
        
        # Add validation information if available
        if val_losses is not None and val_metrics is not None:
            val_loss_str = ', '.join([f"Val {k}: {v:.4f}" for k, v in val_losses.items()])
            val_metrics_str = ', '.join([f"Val {k}: {v:.4f}" for k, v in val_metrics.items()])
            log_str += f", {val_loss_str}, {val_metrics_str}"
            
        # Print log line
        print(log_str)
        
        # Update history
        self.history['train_loss'].append(list(train_losses.values())[0] if train_losses else 0)
        self.history['train_acc'].append(list(train_metrics.values())[0] if train_metrics else 0)
        
        if val_losses is not None and val_metrics is not None:
            self.history['val_loss'].append(list(val_losses.values())[0] if val_losses else 0)
            self.history['val_acc'].append(list(val_metrics.values())[0] if val_metrics else 0)
            
    def plot_metrics(self, save_path=None):
        """
        Plot training and validation metrics.
        
        Args:
            save_path: Path to save the plots
        """
        # Create figure with 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot training and validation loss
        epochs = range(1, len(self.history['train_loss']) + 1)
        axs[0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        if self.history['val_loss']:
            axs[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axs[0].set_title('Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot training and validation accuracy
        axs[1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        if self.history['val_acc']:
            axs[1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        axs[1].set_title('Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()
        axs[1].grid(True)
        
        # Set tight layout
        plt.tight_layout()
        
        # Save plots if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        # Display plots
        plt.show() 