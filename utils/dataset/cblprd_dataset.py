#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset module for CBLPRD-330k.
Supports various processing options including:
- Double-layer plate handling
- Skew correction using Hough transforms
- Data resampling to address class imbalance
"""

import os
import random
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter

from ..configs.config import PLATE_CHARS, get_plate_dict

# Global variables
PLATE_DICT = get_plate_dict()


def is_plate_right(plate_name):
    """Check if the plate name contains only valid characters."""
    assert isinstance(plate_name, str), plate_name
    for ch in plate_name:
        if ch not in PLATE_CHARS:
            return False
    return True


def correct_plate_skew(img, max_angle=15):
    """
    Correct skew in license plate images using Hough transform.
    
    Args:
        img: Input image
        max_angle: Maximum correction angle in degrees
        
    Returns:
        Corrected image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Edge detection
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # Find lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    # If no lines detected, return original image
    if lines is None or len(lines) == 0:
        return img
    
    # Calculate skew angle
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Only consider nearly horizontal lines
        if (theta < np.pi/4 or theta > 3*np.pi/4):
            angle = theta * 180 / np.pi
            if angle > 90:
                angle = angle - 180
            angles.append(angle)
    
    # If no suitable lines, return original image
    if not angles:
        return img
    
    # Use median angle to reduce outlier influence
    angle = np.median(angles)
    
    # Limit maximum correction angle
    if abs(angle) > max_angle:
        angle = max_angle if angle > 0 else -max_angle
    
    # Apply rotation to correct skew
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def process_double_layer_plate(img):
    """
    Process double-layer license plates by splitting and concatenating horizontally.
    
    Args:
        img: Input double-layer plate image
        
    Returns:
        Processed single-layer image
    """
    h, w, c = img.shape
    # Upper part of plate
    img_upper = img[0:int(5/12*h), :]
    # Lower part of plate
    img_lower = img[int(1/3*h):, :]
    # Resize upper part to match lower part
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    # Concatenate horizontally
    new_img = np.hstack((img_upper, img_lower))
    return new_img


def check_if_double_layer(img, plate_text, plate_type=""):
    """
    Check if a license plate is double-layered.
    
    Args:
        img: Input image
        plate_text: License plate text
        plate_type: Type of license plate (if provided)
        
    Returns:
        Boolean indicating if plate is double-layered
    """
    # Check based on plate type string
    if plate_type:
        if "双层" in plate_type or "拖拉机" in plate_type:
            return True
    
    # Check based on text patterns
    if "挂" in plate_text:
        return True
        
    # Check for tractor plates (pattern: province + 2 digits + letters)
    if len(plate_text) >= 3 and plate_text[0] in "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新" and plate_text[1:3].isdigit():
        return True
    
    # Check based on aspect ratio
    h, w = img.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio < 2.0:
        return True
    
    return False


def load_txt_data(txt_path, data_root):
    """
    Load data paths and labels from text file.
    
    Args:
        txt_path: Path to text file
        data_root: Root directory for images
        
    Returns:
        List of data items and dictionary of labels
    """
    assert os.path.isfile(txt_path), f"File not found: {txt_path}"
    
    data_list = []
    label_dict = {}
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 2:
                continue
                
            img_path = parts[0]  # Image path
            label_name = parts[1]  # Plate text
            
            # Get plate type if available
            plate_type = parts[2] if len(parts) > 2 else ""
            
            # Validate plate text
            if len(label_name) < 3:
                continue
            if not is_plate_right(label_name):
                continue
                
            # Build full image path
            full_img_path = os.path.join(data_root, img_path)
            if not os.path.isfile(full_img_path):
                print(f"Warning: Image file not found: {full_img_path}")
                continue
                
            # Store plate text and corresponding character indices
            if label_name not in label_dict:
                label = []
                for i in range(len(label_name)):
                    label.append(PLATE_DICT[label_name[i]])
                label_dict[label_name] = label
                
            data_list.append([full_img_path, label_name, plate_type])
            
    return data_list, label_dict


class CBLPRDDataset(Dataset):
    """
    CBLPRD-330k dataset loader with various processing options.
    """
    def __init__(self, data_root, txt_file, is_train=True, input_shape=(94, 24),
                 use_resampling=True, correct_skew=True, process_double=True):
        """
        Initialize CBLPRD-330k dataset.
        
        Args:
            data_root: Root directory of dataset
            txt_file: Text file with image paths and labels (can be absolute path or relative to project root)
            is_train: Whether this is for training (enables augmentation)
            input_shape: Model input shape (width, height)
            use_resampling: Whether to use resampling to balance classes
            correct_skew: Whether to apply skew correction
            process_double: Whether to process double-layer plates
        """
        # 处理txt_file路径：如果是绝对路径或已包含完整路径则直接使用，否则拼接
        if os.path.isabs(txt_file) or os.path.exists(txt_file):
            txt_path = txt_file
        else:
            # 假定txt_file是在项目根目录下的data文件夹中
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 获取项目根目录（当前脚本目录的上两级）
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            # 构建txt文件的完整路径
            txt_path = os.path.join(project_root, 'data', txt_file)
        
        # 确保txt文件存在
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"TXT file not found: {txt_path}")
            
        self.data_list, self.label_dict = load_txt_data(txt_path, data_root)
        self.is_train = is_train
        self.input_shape = input_shape
        self.correct_skew = correct_skew
        self.process_double = process_double
        
        # Create transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # [0, 255] -> [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet mean/std
        ])
        
        # Apply resampling if enabled
        if is_train and use_resampling:
            self.data_list = self._resample_data(self.data_list)
            
        # Report dataset statistics
        print(f"Loaded {len(self.data_list)} images from {txt_path}")
        print(f"Processing options: correct_skew={correct_skew}, process_double={process_double}")

    def _resample_data(self, data_list, min_samples=5, max_factor=10):
        """
        Resample data to balance class distribution.
        
        Args:
            data_list: List of data items
            min_samples: Minimum number of samples per class to include
            max_factor: Maximum oversampling factor to avoid excessive duplication
            
        Returns:
            Resampled data list
        """
        # Count first character (province) frequencies
        first_char_counts = Counter()
        for item in data_list:
            label = item[1]
            first_char = label[0]
            first_char_counts[first_char] += 1
            
        # Determine target count for each class
        total_count = len(data_list)
        num_classes = len(first_char_counts)
        target_per_class = total_count / num_classes
        
        # Create groups for each first character
        groups = {}
        for item in data_list:
            label = item[1]
            first_char = label[0]
            if first_char not in groups:
                groups[first_char] = []
            groups[first_char].append(item)
            
        # Create resampled list
        resampled_list = []
        
        # Add samples to resampled list
        for char, items in groups.items():
            current_count = len(items)
            if current_count < min_samples:
                # Skip classes with too few samples
                continue
                
            # Calculate how many times to duplicate this class
            factor = min(max_factor, math.ceil(target_per_class / current_count))
            
            # Add original samples
            resampled_list.extend(items)
            
            # Add additional samples if needed
            if factor > 1:
                # Calculate how many additional copies needed
                additional_needed = int((factor - 1) * current_count)
                # Duplicate with replacement
                additional_items = random.choices(items, k=additional_needed)
                resampled_list.extend(additional_items)
                
        print(f"Resampling: original={len(data_list)}, resampled={len(resampled_list)}")
        return resampled_list

    def __getitem__(self, index):
        """Get a sample from the dataset."""
        img_path, label_name, plate_type = self.data_list[index]
        img = cv2.imread(img_path)
        
        # Apply skew correction if enabled
        if self.correct_skew:
            img = correct_plate_skew(img)
            
        # Process double-layer plates if needed
        if self.process_double and check_if_double_layer(img, label_name, plate_type):
            img = process_double_layer_plate(img)
            
        # Resize to model input size
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        
        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        img = self.transforms(img)
        
        # Get label
        label = self.label_dict[label_name]
        
        # Convert to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label, len(label)

    def __len__(self):
        """Get dataset size."""
        return len(self.data_list)
        
def collate_fn(batch):
    """
    自定义批处理函数，处理不同长度的标签。
    
    参数:
        batch: 批次数据，每个元素包含(图像, 标签, 标签长度)
        
    返回:
        处理后的批次数据：(图像张量, 填充后的标签张量, 标签长度张量)
    """
    images = []
    labels = []
    label_lengths = []
    
    for img, label, length in batch:
        images.append(img)
        labels.append(label)
        label_lengths.append(length)
    
    # 堆叠图像
    images = torch.stack(images, 0)
    
    # 找出最长标签的长度
    max_length = max(label_lengths)
    
    # 创建填充后的标签张量
    batch_size = len(labels)
    padded_labels = torch.zeros(batch_size, max_length, dtype=torch.long)
    
    # 填充标签
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    
    # 转换长度为张量
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    
    return images, padded_labels, label_lengths 