#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集加载测试脚本
用于验证CBLPRD-330k数据集是否可以正确加载
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from utils.dataset import LPRDataset
from utils.utils import load_config, build_char_dict

def parse_args():
    parser = argparse.ArgumentParser(description='测试数据集加载')
    parser.add_argument('--config', type=str, default='config/lprnet_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='要检查的样本数量')
    parser.add_argument('--save_dir', type=str, default='output/dataset_test',
                        help='保存测试结果的目录')
    
    return parser.parse_args()

def test_dataset_loading(config, num_samples=5, save_dir=None):
    """测试数据集加载功能"""
    # 创建字符字典
    chars_dict, chars_list = build_char_dict(config)
    
    # 测试训练数据集
    print("\n正在测试训练数据集...")
    train_dataset = LPRDataset(
        data_file=config['DATASET']['TRAIN_FILE'],
        chars_dict=chars_dict,
        img_size=config['MODEL']['INPUT_SIZE'],
        is_train=False,  # 关闭数据增强以便可视化
        dataset_root=config['DATASET']['ROOT']
    )
    
    print(f"训练集样本数量: {len(train_dataset)}")
    
    # 测试验证数据集
    print("\n正在测试验证数据集...")
    val_dataset = LPRDataset(
        data_file=config['DATASET']['VAL_FILE'],
        chars_dict=chars_dict,
        img_size=config['MODEL']['INPUT_SIZE'],
        is_train=False,
        dataset_root=config['DATASET']['ROOT']
    )
    
    print(f"验证集样本数量: {len(val_dataset)}")
    
    # 调试: 检查第一个样本的形状
    if len(train_dataset) > 0:
        sample_img, _, _ = train_dataset[0]
        print(f"\n调试信息 - 训练集第一个样本:")
        print(f"  形状: {sample_img.shape}")
        print(f"  维度数: {sample_img.dim()}")
        print(f"  类型: {sample_img.dtype}")
    
    if len(val_dataset) > 0:
        sample_img, _, _ = val_dataset[0]
        print(f"\n调试信息 - 验证集第一个样本:")
        print(f"  形状: {sample_img.shape}")
        print(f"  维度数: {sample_img.dim()}")
        print(f"  类型: {sample_img.dtype}")
    
    # 可视化一些样本
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n正在可视化 {num_samples} 个训练样本...")
        visualize_samples(train_dataset, chars_list, num_samples, 
                         os.path.join(save_dir, 'train_samples.png'))
        
        print(f"\n正在可视化 {num_samples} 个验证样本...")
        visualize_samples(val_dataset, chars_list, num_samples, 
                         os.path.join(save_dir, 'val_samples.png'))
    
    # 测试所有样本能否加载
    test_all_samples(train_dataset, "训练集")
    test_all_samples(val_dataset, "验证集")
    
    return True

def test_all_samples(dataset, name):
    """测试所有样本是否可以加载"""
    print(f"\n正在测试所有 {name} 样本加载...")
    errors = 0
    
    for i in tqdm(range(len(dataset)), desc=f"验证 {name} 样本"):
        try:
            img, label, length = dataset[i]
            if torch.sum(img) == 0:
                errors += 1
        except Exception as e:
            errors += 1
    
    if errors == 0:
        print(f"✓ 所有 {len(dataset)} 个 {name} 样本加载成功!")
    else:
        print(f"✗ {errors} 个 {name} 样本无法正确加载 (共 {len(dataset)} 个)")

def visualize_samples(dataset, chars_list, num_samples, save_path=None):
    """可视化数据集样本"""
    indices = torch.randperm(len(dataset))[:num_samples]
    
    plt.figure(figsize=(15, 3*num_samples))
    
    for i, idx in enumerate(indices):
        img, label, length = dataset[idx]
        
        # 检查维度并相应处理
        if img.dim() == 4:  # 如果是 [batch, channels, height, width]
            img = img.squeeze(0)  # 移除批次维度，变成 [channels, height, width]
        elif img.dim() != 3:  # 输出错误详细信息以便调试
            print(f"警告: 图像维度异常 - 形状为 {img.shape}, dim={img.dim()}")
            continue  # 跳过这个样本
            
        # 将张量转换为可显示的图像
        img_np = img.permute(1, 2, 0).numpy()
        
        # 使用numpy数组进行反归一化处理
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # 将标签索引转换为文本
        label_text = ''.join([chars_list[idx-1] for idx in label[:length] if idx > 0])
        
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img_np)
        plt.title(f"样本 #{idx}: {label_text}")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"样本图像已保存至 {save_path}")
    else:
        plt.show()

def main():
    args = parse_args()
    
    # 加载配置
    print(f"正在加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 显示数据集路径信息
    print("\n数据集配置:")
    print(f"数据集根目录: {config['DATASET']['ROOT']}")
    print(f"训练文件: {config['DATASET']['TRAIN_FILE']}")
    print(f"验证文件: {config['DATASET']['VAL_FILE']}")
    
    # 检查文件是否存在
    if not os.path.exists(config['DATASET']['TRAIN_FILE']):
        print(f"错误: 训练文件不存在: {config['DATASET']['TRAIN_FILE']}")
    
    if not os.path.exists(config['DATASET']['VAL_FILE']):
        print(f"错误: 验证文件不存在: {config['DATASET']['VAL_FILE']}")
    
    # 测试数据集加载
    test_dataset_loading(config, args.num_samples, args.save_dir)
    
    print("\n测试完成!")

if __name__ == '__main__':
    main() 