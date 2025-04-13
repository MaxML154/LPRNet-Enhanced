#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration module for LPRNet-Enhanced.
Defines parameters, character sets, and command-line argument parsing.
"""

import os
import argparse

# CBLPRD-330k dataset parameters
PLATE_CHARS = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学港澳挂使领0123456789ABCDEFGHJKLMNPQRSTUVWXYZ临"
NUM_CLASSES = len(PLATE_CHARS)

def get_plate_dict():
    """Generate a dictionary mapping plate characters to indices."""
    plate_dict = {}
    for i, char in enumerate(PLATE_CHARS):
        plate_dict[char] = i
    return plate_dict

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CBLPRD-330k LPRNet Training')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, required=True, help='Path to CBLPRD-330k dataset')
    parser.add_argument('--train-file', type=str, default='data/train.txt', help='Train file name')
    parser.add_argument('--val-file', type=str, default='data/val.txt', help='Validation file name')
    parser.add_argument('--test-file', type=str, default='data/test.txt', help='Test file name')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='lprnet_plus', 
                      choices=['lprnet', 'lprnet_plus', 'lprnet_stnet', 'lprnet_plus_stnet'],
                      help='Model architecture to use')
    parser.add_argument('--input-size', type=str, default='94x24', help='Input size for model (WxH)')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate for model')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--devices', type=str, default='0', help='Device ids to use (e.g., "0,1")')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    
    # 学习率调度器参数
    parser.add_argument('--lr-scheduler', type=str, default='multistep', 
                        choices=['step', 'multistep', 'cosine', 'reduce', 'plateau', 'onecycle'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr-step', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr-steps', type=str, default='',
                        help='Comma-separated list of epochs to reduce LR for MultiStepLR (e.g., "10,20,25"), if empty will auto-compute')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='LR reduction factor at each milestone')
    parser.add_argument('--lr-min', type=float, default=1e-6,
                        help='Minimum learning rate for schedulers like CosineAnnealingLR')
    parser.add_argument('--lr-warmup-epochs', type=int, default=0,
                        help='Number of epochs for learning rate warmup')
    parser.add_argument('--lr-decay-points', type=int, default=3,
                        help='Number of decay points for auto-computed MultiStepLR')
    
    # Data augmentation options
    parser.add_argument('--use-resampling', action='store_true', help='Use resampling to balance data')
    parser.add_argument('--correct-skew', action='store_true', help='Apply skew correction to license plates')
    parser.add_argument('--no-double-process', action='store_false', dest='process_double', 
                      help='Disable double-layer plate processing')
    
    # Saving and logging
    parser.add_argument('--save-dir', type=str, default='weights', help='Directory to save checkpoints')
    parser.add_argument('--log-interval', type=int, default=10, help='Print interval (batches)')
    parser.add_argument('--save-interval', type=int, default=1, help='Save checkpoint interval (epochs)')
    parser.add_argument('--resume', type=str, default='', help='Resume training from checkpoint')
    
    # Early stopping
    parser.add_argument('--early-stopping', type=int, default=10, 
                      help='Number of epochs with no improvement after which training will be stopped')
    
    # Testing after training
    parser.add_argument('--test-after-train', action='store_true', help='Run test evaluation after training')
    parser.add_argument('--no-test-after-train', action='store_false', dest='test_after_train', 
                      help='Skip test evaluation after training')
    parser.set_defaults(test_after_train=True)
    
    # Inference
    parser.add_argument('--weights', type=str, default='', help='Path to model weights for inference')
    parser.add_argument('--image', type=str, default='', help='Path to image for inference')
    
    args = parser.parse_args()
    
    # Parse input size
    w, h = args.input_size.split('x')
    args.input_width = int(w)
    args.input_height = int(h)
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    return args

# 计算动态学习率调度点
def compute_lr_milestones(total_epochs, num_decay_points=3):
    """
    根据总训练周期和降低点数量，动态计算学习率降低的里程碑。
    
    参数:
        total_epochs: 总训练周期数
        num_decay_points: 学习率降低点的数量
        
    返回:
        降低学习率的epoch列表
    """
    if num_decay_points <= 0:
        return []
    
    milestones = []
    
    # 分割训练过程
    if num_decay_points == 1:
        # 只有一个降低点，放在2/3处
        milestones.append(int(total_epochs * 2/3))
    elif num_decay_points == 2:
        # 两个降低点，分别在1/3和2/3处
        milestones.append(int(total_epochs * 1/3))
        milestones.append(int(total_epochs * 2/3))
    else:
        # 第一个降低点：总次数的1/3
        first_milestone = int(total_epochs / 3)
        milestones.append(first_milestone)
        
        # 中间降低点：均匀分布
        remaining_epochs = total_epochs - first_milestone
        remaining_points = num_decay_points - 1
        
        for i in range(1, remaining_points):
            next_point = first_milestone + int(remaining_epochs * i / remaining_points)
            milestones.append(next_point)
        
        # 确保最后一个降低点不会太接近训练结束
        # 至少留5%的训练周期或至少1个epoch用于最后的学习率
        last_milestone = int(total_epochs * 0.95)
        if last_milestone > milestones[-1] and last_milestone < total_epochs:
            milestones[-1] = last_milestone
    
    # 确保里程碑递增且没有重复
    milestones = sorted(list(set(milestones)))
    
    # 确保所有里程碑都小于总训练周期
    milestones = [m for m in milestones if m < total_epochs]
    
    return milestones

# Default configuration that can be used programmatically
def get_default_config():
    """Get default configuration without command line arguments."""
    config = {
        # Data parameters
        'data_dir': '/mnt/dataset/CBLPRD-330k',
        'train_file': 'data/train.txt',
        'val_file': 'data/val.txt',
        'test_file': 'data/test.txt',
        
        # Model parameters
        'model_type': 'lprnet_plus_stnet',
        'input_width': 94,
        'input_height': 24,
        'dropout_rate': 0.5,
        'lpr_max_len': 8,
        
        # Training parameters
        'batch_size': 128,
        'epochs': 200,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'momentum': 0.9,
        'devices': '0',
        'workers': 4,
        
        # 学习率调度器参数
        'lr_scheduler': 'multistep',
        'lr_step': 30,
        'lr_steps': '',
        'lr_gamma': 0.1,
        'lr_min': 1e-6,
        'lr_warmup_epochs': 0,
        'lr_decay_points': 3,
        
        # Data augmentation options
        'use_resampling': True,
        'correct_skew': True,
        'process_double': True,
        
        # Saving and logging
        'save_dir': './weights',
        'log_interval': 10,
        'save_interval': 1,
        'resume': '',
        
        # Early stopping
        'early_stopping': 10,
        
        # Plate characters
        'plate_chars': PLATE_CHARS,
        'num_classes': NUM_CLASSES,
        
        # Inference
        'weights': '',
        'image': '',
    }
    
    return config

if __name__ == '__main__':
    args = get_args()
    print(args) 