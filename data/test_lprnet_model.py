#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LPRNet模型脚本
用于验证修复后的LPRNet模型能否正确处理各种输入尺寸
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import argparse
from models.lprnet import build_lprnet
from utils.utils import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='测试LPRNet模型')
    parser.add_argument('--config', type=str, default='config/lprnet_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小')
    
    return parser.parse_args()

def test_model_with_different_inputs(model, input_size, batch_sizes=[1, 2, 4, 8]):
    """测试模型对不同批次大小和输入尺寸的处理能力"""
    width, height = input_size
    device = next(model.parameters()).device
    
    print(f"\n测试不同批次大小的输入处理能力:")
    for batch_size in batch_sizes:
        try:
            # 创建随机输入
            dummy_input = torch.randn(batch_size, 3, height, width).to(device)
            
            # 前向传播
            output = model(dummy_input)
            
            print(f"  批次大小 {batch_size}: 成功 (输入形状: {dummy_input.shape}, 输出形状: {output.shape})")
        except Exception as e:
            print(f"  批次大小 {batch_size}: 失败 - 错误: {e}")
    
    print("\n测试不同输入尺寸的处理能力:")
    # 测试不同的宽高比
    test_sizes = [
        (94, 24),   # 默认尺寸
        (100, 30),  # 略大尺寸
        (80, 20),   # 略小尺寸
        (120, 35),  # 更大尺寸
        (70, 15)    # 更小尺寸
    ]
    
    for size in test_sizes:
        w, h = size
        try:
            # 创建随机输入
            dummy_input = torch.randn(2, 3, h, w).to(device)
            
            # 前向传播
            output = model(dummy_input)
            
            print(f"  输入尺寸 {size}: 成功 (输入形状: {dummy_input.shape}, 输出形状: {output.shape})")
        except Exception as e:
            print(f"  输入尺寸 {size}: 失败 - 错误: {e}")

def main():
    # 解析参数
    args = parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建模型
    print("构建LPRNet模型...")
    model = build_lprnet(config)
    model = model.to(device)
    model.eval()
    
    # 打印模型信息
    print(f"模型构建成功，共 {sum(p.numel() for p in model.parameters() if p.requires_grad):,} 个可训练参数")
    
    # 测试单个输入
    input_size = config['MODEL']['INPUT_SIZE']
    batch_size = args.batch_size
    dummy_input = torch.randn(batch_size, 3, input_size[1], input_size[0]).to(device)
    
    try:
        # 前向传播
        output = model(dummy_input)
        print(f"\n基础测试: 成功")
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  输出形状: {output.shape}")
    except Exception as e:
        print(f"\n基础测试: 失败")
        print(f"  错误: {e}")
    
    # 测试不同的输入尺寸
    test_model_with_different_inputs(model, input_size)
    
    print("\n测试完成!")

if __name__ == '__main__':
    main() 
