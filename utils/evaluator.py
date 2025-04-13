#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluator module for license plate recognition.
"""

import torch
import numpy as np
from .configs.config import PLATE_CHARS


class Evaluator:
    """
    Evaluator for license plate recognition models.
    Handles prediction decoding and accuracy calculation.
    """
    def __init__(self, blank_label=0):
        """
        Initialize evaluator.
        
        Args:
            blank_label: Index of blank label in predictions
        """
        self.blank_label = blank_label
        self.plate_chars = PLATE_CHARS
        
    def decode(self, preds):
        """
        Decode predictions using greedy search.
        
        Args:
            preds: Model predictions (batch_size, seq_len, num_classes)
            
        Returns:
            List of predicted plate strings
        """
        # Get the most likely class at each step
        preds_argmax = preds.argmax(dim=2)
        
        batch_size = preds_argmax.size(0)
        results = []
        
        for i in range(batch_size):
            indices = preds_argmax[i]
            out = []
            last_index = -1
            # Remove duplicates and blanks
            for index in indices:
                index_val = index.item()
                if index_val != self.blank_label and index_val != last_index:
                    out.append(index_val)
                last_index = index_val
                
            # Convert indices to characters
            plate_text = ''.join([self.plate_chars[idx] for idx in out])
            results.append(plate_text)
            
        return results
        
    def calculate_accuracy(self, preds, targets):
        """
        Calculate accuracy for predictions.
        
        Args:
            preds: Model predictions (batch_size, seq_len, num_classes)
            targets: Target labels list of strings
            
        Returns:
            Accuracy values (character-level and sequence-level)
        """
        decoded_preds = self.decode(preds)
        
        # Calculate sequence accuracy (exact match)
        seq_correct = 0
        # Calculate character accuracy
        total_chars = 0
        correct_chars = 0
        
        for i, (pred, target) in enumerate(zip(decoded_preds, targets)):
            # For exact match
            if pred == target:
                seq_correct += 1
                
            # For character-level accuracy
            min_len = min(len(pred), len(target))
            total_chars += max(len(pred), len(target))
            
            # Count correct characters
            for j in range(min_len):
                if pred[j] == target[j]:
                    correct_chars += 1
                    
        seq_accuracy = seq_correct / len(targets)
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        
        return seq_accuracy, char_accuracy
        
    def evaluate_batch(self, model, batch, device):
        """
        Evaluate model on a single batch.
        
        Args:
            model: The model to evaluate
            batch: Batch of data (images, labels, label_lengths)
            device: Device to run on
            
        Returns:
            Evaluation metrics
        """
        model.eval()
        
        with torch.no_grad():
            images, labels, label_lengths = batch
            images = images.to(device)
            
            # Get predictions
            preds = model(images)
            
            # Convert target indices to strings
            target_strings = []
            for i, length in enumerate(label_lengths):
                target = labels[i][:length]
                target_chars = [self.plate_chars[idx.item()] for idx in target]
                target_strings.append("".join(target_chars))
            
            # Calculate accuracy
            seq_accuracy, char_accuracy = self.calculate_accuracy(preds, target_strings)
            
        return {
            'sequence_accuracy': seq_accuracy,
            'character_accuracy': char_accuracy,
            'predictions': self.decode(preds),
            'targets': target_strings
        }
        
    def evaluate_loader(self, model, data_loader, device):
        """
        Evaluate model on an entire data loader.
        
        Args:
            model: The model to evaluate
            data_loader: DataLoader containing test data
            device: Device to run on
            
        Returns:
            Evaluation metrics
        """
        model.eval()
        
        total_seq_accuracy = 0
        total_char_accuracy = 0
        total_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                images, labels, label_lengths = batch
                images = images.to(device)
                
                # Get predictions
                preds = model(images)
                
                # Convert target indices to strings
                target_strings = []
                for i, length in enumerate(label_lengths):
                    target = labels[i][:length]
                    target_chars = [self.plate_chars[idx.item()] for idx in target]
                    target_strings.append("".join(target_chars))
                
                # Calculate accuracy
                seq_accuracy, char_accuracy = self.calculate_accuracy(preds, target_strings)
                
                total_seq_accuracy += seq_accuracy
                total_char_accuracy += char_accuracy
                total_batches += 1
                
                all_predictions.extend(self.decode(preds))
                all_targets.extend(target_strings)
        
        # Calculate average accuracy
        avg_seq_accuracy = total_seq_accuracy / total_batches if total_batches > 0 else 0
        avg_char_accuracy = total_char_accuracy / total_batches if total_batches > 0 else 0
        
        return {
            'sequence_accuracy': avg_seq_accuracy,
            'character_accuracy': avg_char_accuracy,
            'predictions': all_predictions,
            'targets': all_targets
        } 