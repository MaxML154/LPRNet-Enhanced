import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random

class LPRDataset(Dataset):
    def __init__(self, data_file, chars_dict, img_size=(94, 24), is_train=True, augmentation=None, dataset_root=None):
        """
        Initialize LPR dataset from CBLPRD-330k format
        Args:
            data_file (str): Path to data file (train.txt or val.txt)
            chars_dict (dict): Dictionary mapping characters to indices
            img_size (tuple): Input image size (width, height)
            is_train (bool): Whether it's training set
            augmentation (dict): Augmentation parameters
            dataset_root (str): Root directory of the dataset
        """
        self.data_list = []
        self.chars_dict = chars_dict
        self.img_size = img_size
        self.is_train = is_train
        self.augmentation = augmentation
        self.dataset_root = dataset_root
        
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:  # Image path and license plate text
                    img_path = parts[0]
                    plate_text = parts[1]
                    # Plate type is optional (parts[2] if available)
                    plate_type = parts[2] if len(parts) > 2 else ""
                    self.data_list.append((img_path, plate_text, plate_type))

        # Basic transformations
        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path, plate_text, plate_type = self.data_list[index]
        
        try:
            # 创建可能的图片路径列表，按优先级尝试
            possible_paths = []
            
            # 1. 首先尝试从配置的dataset_root加载
            if self.dataset_root:
                # 处理纯文件名 (不包含路径的情况)
                if os.path.basename(img_path) == img_path:  # 如果img_path只是一个文件名
                    possible_paths.append(os.path.join(self.dataset_root, img_path))
            
            # 2. 尝试原始路径
            possible_paths.append(img_path)
            
            # 3. 尝试其他常见位置
            possible_paths.extend([
                os.path.join("data/CBLPRD-330k", os.path.basename(img_path)),
                os.path.join("CBLPRD-330k", os.path.basename(img_path)),
                os.path.join("data", os.path.basename(img_path))
            ])
            
            # 尝试所有可能的路径
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    found = True
                    break
            
            if not found:
                # 如果所有尝试都失败，则打印详细错误信息并返回默认值
                error_msg = f"Image file not found. Tried paths: {possible_paths}"
                raise FileNotFoundError(error_msg)
            
            # 加载和预处理图像
            img = Image.open(img_path).convert('RGB')
            
            # 应用数据增强
            if self.is_train and self.augmentation and self.augmentation.get('ENABLE', False):
                img = self._augment_image(img)
                
            # 应用变换
            img = self.transforms(img)
            
            # 处理双层车牌
            if plate_type == "双层黄牌" or plate_type == "双层蓝牌":
                # 可以实现双层车牌的处理逻辑
                pass
                
            # 将车牌文本转换为索引
            plate_label, original_length = self._text_to_indices(plate_text)
            
            return img, plate_label, original_length
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回默认值
            dummy_img = torch.zeros(3, self.img_size[1], self.img_size[0])
            dummy_label = torch.zeros(8, dtype=torch.int)
            return dummy_img, dummy_label, 0

    def _text_to_indices(self, text):
        """Convert license plate text to indices"""
        indices = []
        
        # 检查文本长度，避免过长的车牌号
        if len(text) > 8:
            text = text[:8]
            print(f"Warning: Truncating license plate text to 8 characters: {text}")
            
        for char in text:
            if char in self.chars_dict:
                indices.append(self.chars_dict[char])
            else:
                # Handle unknown characters
                print(f"Warning: Unknown character '{char}' in license plate text '{text}'")
                indices.append(0)  # Use 0 as index for unknown characters
                
        # 原始文本长度（实际有效长度）
        original_length = len(indices)
        
        # Pad to max length (assuming 8 is max length for Chinese license plates)
        while len(indices) < 8:
            indices.append(0)  # Padding index
            
        return torch.tensor(indices, dtype=torch.long), original_length

    def _augment_image(self, img):
        """Apply data augmentation to image"""
        if not self.augmentation:
            return img
        
        img_np = np.array(img)
        
        # Random rotation
        if random.random() < 0.5 and 'ROTATION_RANGE' in self.augmentation:
            angle = random.uniform(-self.augmentation['ROTATION_RANGE'], self.augmentation['ROTATION_RANGE'])
            h, w = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img_np = cv2.warpAffine(img_np, M, (w, h), borderValue=(114, 114, 114))
        
        # Random brightness and contrast adjustment
        if random.random() < 0.5 and 'BRIGHTNESS_RANGE' in self.augmentation:
            brightness = random.uniform(self.augmentation['BRIGHTNESS_RANGE'][0], 
                                       self.augmentation['BRIGHTNESS_RANGE'][1])
            img_np = img_np * brightness
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
        if random.random() < 0.5 and 'CONTRAST_RANGE' in self.augmentation:
            contrast = random.uniform(self.augmentation['CONTRAST_RANGE'][0], 
                                     self.augmentation['CONTRAST_RANGE'][1])
            img_np = img_np * contrast
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Add Gaussian noise
        if random.random() < 0.3 and 'GAUSSIAN_NOISE' in self.augmentation:
            noise = np.random.normal(0, self.augmentation['GAUSSIAN_NOISE'] * 255, img_np.shape).astype(np.uint8)
            img_np = cv2.add(img_np, noise)
        
        # Apply blur
        if random.random() < self.augmentation.get('BLUR_PROBABILITY', 0) and 'BLUR_SIZE' in self.augmentation:
            blur_size = self.augmentation['BLUR_SIZE']
            img_np = cv2.GaussianBlur(img_np, (blur_size, blur_size), 0)
            
        return Image.fromarray(img_np)


def get_train_dataloader(config):
    """
    Create training dataloader based on configuration
    """
    # Build character dictionary
    chars_list = (
        config['CHARS']['PROVINCES'] + 
        config['CHARS']['ALPHABETS'] + 
        config['CHARS']['DIGITS'] + 
        config['CHARS']['SEPARATOR']
    )
    chars_dict = {char: i+1 for i, char in enumerate(chars_list)}  # 0 is reserved for padding/unknown
    
    # Create datasets
    train_dataset = LPRDataset(
        data_file=config['DATASET']['TRAIN_FILE'],
        chars_dict=chars_dict,
        img_size=config['MODEL']['INPUT_SIZE'],
        is_train=True,
        augmentation=config['AUGMENTATION'],
        dataset_root=config['DATASET']['ROOT']  # 添加数据集根目录
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['TRAIN']['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['TRAIN']['WORKERS'],
        pin_memory=True
    )
    
    return train_loader, chars_dict


def get_val_dataloader(config, chars_dict):
    """
    Create validation dataloader based on configuration
    """
    val_dataset = LPRDataset(
        data_file=config['DATASET']['VAL_FILE'],
        chars_dict=chars_dict,
        img_size=config['MODEL']['INPUT_SIZE'],
        is_train=False,
        dataset_root=config['DATASET']['ROOT']  # 添加数据集根目录
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['TEST']['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['TRAIN']['WORKERS'],
        pin_memory=True
    )
    
    return val_loader


def idx_to_text(indices, chars_list):
    """
    Convert indices to license plate text
    """
    chars = []
    for idx in indices:
        if idx > 0 and idx <= len(chars_list):
            chars.append(chars_list[idx - 1])
    
    return ''.join(chars) 