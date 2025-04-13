#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing script for LPRNet-Enhanced.
"""

import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

from utils.configs.config import get_args, PLATE_CHARS, NUM_CLASSES
from utils.dataset.cblprd_dataset import CBLPRDDataset, collate_fn
from utils.model.lprnet import build_lprnet
from utils.evaluator import Evaluator


def test_model(args=None):
    """Test the model on the test set."""
    # Get arguments
    if args is None:
        args = get_args()
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = build_lprnet(
        model_type=args.model_type,
        num_classes=NUM_CLASSES,
        lpr_max_len=8,
        dropout_rate=args.dropout_rate
    )
    model = model.to(device)
    
    # Load weights
    if not args.weights:
        print("Error: Please provide weights path using --weights")
        return
        
    if not os.path.isfile(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        return
        
    print(f"Loading weights from {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded weights from epoch {checkpoint['epoch']}")
    
    # 打印文件路径信息
    print(f"Data directory: {args.data_dir}")
    print(f"Test file: {args.test_file}")
    
    # Create test dataset
    test_dataset = CBLPRDDataset(
        data_root=args.data_dir,
        txt_file=args.test_file,
        is_train=False,
        input_shape=(args.input_width, args.input_height),
        use_resampling=False,
        correct_skew=args.correct_skew,
        process_double=args.process_double
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Evaluate model
    evaluator = Evaluator(blank_label=0)
    model.eval()
    
    test_metrics = evaluator.evaluate_loader(model, test_loader, device)
    
    print("\nTest Results:")
    print(f"Sequence Accuracy: {test_metrics['sequence_accuracy']:.4f}")
    print(f"Character Accuracy: {test_metrics['character_accuracy']:.4f}")
    
    # 根据参数决定是否显示样本图像
    if args.visualize_samples:
        # Show some examples
        visualize_samples(model, test_loader, evaluator, device, num_samples=args.num_visualize)
    
    return test_metrics


def test_single_image(args=None):
    """Test the model on a single image."""
    # Get arguments
    if args is None:
        args = get_args()
    
    if not args.image:
        print("Error: Please provide image path using --image")
        return
    
    if not os.path.isfile(args.image):
        print(f"Error: Image file not found at {args.image}")
        return
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = build_lprnet(
        model_type=args.model_type,
        num_classes=NUM_CLASSES,
        lpr_max_len=8,
        dropout_rate=args.dropout_rate
    )
    model = model.to(device)
    
    # Load weights
    if not args.weights:
        print("Error: Please provide weights path using --weights")
        return
        
    if not os.path.isfile(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        return
        
    print(f"Loading weights from {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded weights from epoch {checkpoint['epoch']}")
    
    # Load and preprocess image
    img = cv2.imread(args.image)
    original_img = img.copy()
    
    # Apply skew correction if enabled
    if args.correct_skew:
        from utils.dataset.cblprd_dataset import correct_plate_skew
        img = correct_plate_skew(img)
    
    # Process double-layer plate if needed
    if args.process_double:
        from utils.dataset.cblprd_dataset import check_if_double_layer, process_double_layer_plate
        if check_if_double_layer(img, ""):
            img = process_double_layer_plate(img)
    
    # Resize to model input size
    img = cv2.resize(img, (args.input_width, args.input_height))
    
    # Convert to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        preds = model(img)
    
    # Decode prediction
    evaluator = Evaluator(blank_label=0)
    predicted_plate = evaluator.decode(preds)[0]
    
    # Display results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cv2.resize(img.cpu().squeeze().permute(1, 2, 0).numpy(), 
                                     (original_img.shape[1], original_img.shape[0])), 
                          cv2.COLOR_BGR2RGB))
    plt.title(f'Preprocessed Image\nPredicted: {predicted_plate}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Predicted license plate: {predicted_plate}")


def visualize_samples(model, data_loader, evaluator, device, num_samples=5):
    """Visualize sample predictions."""
    model.eval()
    all_images = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, label_lengths in data_loader:
            if len(all_images) >= num_samples:
                break
                
            # Get predictions
            images_device = images.to(device)
            preds = model(images_device)
            
            # Convert target indices to strings
            target_strings = []
            for i, length in enumerate(label_lengths):
                target = labels[i][:length]
                target_chars = [PLATE_CHARS[idx.item()] for idx in target]
                target_strings.append("".join(target_chars))
            
            # Decode predictions
            pred_strings = evaluator.decode(preds)
            
            # Store samples
            batch_size = min(images.size(0), num_samples - len(all_images))
            all_images.extend([images[i].permute(1, 2, 0).numpy() for i in range(batch_size)])
            all_preds.extend(pred_strings[:batch_size])
            all_targets.extend(target_strings[:batch_size])
    
    # Create figure
    rows = min(5, num_samples)
    cols = (num_samples + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    # Plot samples
    for i in range(len(all_images)):
        ax = axes[i]
        
        # Denormalize image
        img = all_images[i] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Show image
        ax.imshow(img)
        ax.set_title(f"Pred: {all_preds[i]}\nTrue: {all_targets[i]}")
        ax.axis('off')
        
        # Add color indicator for correct/incorrect
        color = 'green' if all_preds[i] == all_targets[i] else 'red'
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['top'].set_linewidth(5)
        ax.spines['left'].set_linewidth(5)
        ax.spines['right'].set_linewidth(5)
    
    # Hide unused axes
    for i in range(len(all_images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 创建自定义的命令行参数解析器
    parser = argparse.ArgumentParser(description='LPRNet-Enhanced Test Script')
    
    # 基本选项
    parser.add_argument('--single-image', action='store_true', help='Test on a single image')
    parser.add_argument('--weights', type=str, help='Path to model weights')
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--test-file', type=str, help='Path to test annotation file')
    parser.add_argument('--image', type=str, help='Path to single image for testing')
    
    # 可视化选项
    parser.add_argument('--visualize-samples', action='store_true', default=True, 
                        help='Visualize sample predictions')
    parser.add_argument('--no-visualize-samples', action='store_false', dest='visualize_samples', 
                        help='Skip sample visualization')
    parser.add_argument('--num-visualize', type=int, default=10, 
                        help='Number of samples to visualize')
    
    # 解析命令行参数
    cmd_args = parser.parse_args()
    
    # 合并命令行参数和配置文件参数
    config_args = get_args()
    
    # 如果命令行提供了参数，则覆盖配置文件中的参数
    for key, value in vars(cmd_args).items():
        if value is not None:
            setattr(config_args, key, value)
    
    # 根据参数决定运行模式
    if config_args.single_image:
        test_single_image(config_args)
    else:
        test_model(config_args) 