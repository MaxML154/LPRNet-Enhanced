#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复数据集路径问题的辅助脚本
这个脚本帮助处理CBLPRD-330k数据集路径问题，生成正确的train.txt和val.txt文件
"""

import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='修复CBLPRD-330k数据集路径问题')
    parser.add_argument('--dataset_dir', type=str, default='F:/CBLPRD-330k_v1/CBLPRD-330k',
                        help='CBLPRD-330k数据集目录路径')
    parser.add_argument('--train_file', type=str, default='F:/CBLPRD-330k_v1/train1.txt',
                        help='原始train.txt文件路径')
    parser.add_argument('--val_file', type=str, default='F:/CBLPRD-330k_v1/val1.txt',
                        help='原始val.txt文件路径')
    parser.add_argument('--output_train', type=str, default='data/train.txt',
                        help='输出train.txt文件路径')
    parser.add_argument('--output_val', type=str, default='data/val.txt',
                        help='输出val.txt文件路径')
    parser.add_argument('--check_files', action='store_true',
                        help='检查图片文件是否存在')
    
    return parser.parse_args()

def process_data_file(input_file, output_file, dataset_dir, check_files=False):
    """处理数据文件，转换路径格式"""
    print(f"处理文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return 0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取数据文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 移除空行并去除空白
    lines = [line.strip() for line in lines if line.strip()]
    
    processed_lines = []
    missing_files = 0
    total_files = len(lines)
    
    for line in tqdm(lines, desc="处理中"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        img_path = parts[0]
        plate_text = parts[1]
        plate_type = parts[2] if len(parts) > 2 else ""
        
        # 构建绝对路径
        img_filename = os.path.basename(img_path)
        abs_img_path = os.path.join(dataset_dir, img_filename)
        
        # 检查文件是否存在
        if check_files and not os.path.exists(abs_img_path):
            print(f"警告: 找不到图片文件 {abs_img_path}")
            missing_files += 1
        
        # 创建处理后的行
        processed_line = f"{abs_img_path} {plate_text}"
        if plate_type:
            processed_line += f" {plate_type}"
        
        processed_lines.append(processed_line)
    
    # 写入处理后的行到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(f"{line}\n")
    
    print(f"已处理 {len(processed_lines)} 行")
    if check_files and missing_files > 0:
        print(f"警告: 有 {missing_files} 个图片文件缺失 (共 {total_files} 个)")
    
    print(f"结果已保存至: {output_file}")
    return len(processed_lines)

def update_config_file():
    """更新配置文件中的路径"""
    config_file = 'config/lprnet_config.yaml'
    if not os.path.exists(config_file):
        print(f"错误: 找不到配置文件 {config_file}")
        return False
    
    print(f"更新配置文件: {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        if 'TRAIN_FILE:' in line:
            updated_lines.append('  TRAIN_FILE: "data/train.txt" # Training data list file\n')
        elif 'VAL_FILE:' in line:
            updated_lines.append('  VAL_FILE: "data/val.txt" # Validation data list file\n')
        else:
            updated_lines.append(line)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print("配置文件已更新")
    return True

def main():
    args = parse_args()
    
    # 处理训练文件
    train_count = process_data_file(args.train_file, args.output_train, args.dataset_dir, args.check_files)
    
    # 处理验证文件
    val_count = process_data_file(args.val_file, args.output_val, args.dataset_dir, args.check_files)
    
    # 更新配置文件
    updated = update_config_file()
    
    # 输出结果
    print("\n========== 数据集路径修复完成 ==========")
    print(f"处理了 {train_count} 条训练样本")
    print(f"处理了 {val_count} 条验证样本")
    print(f"数据集目录: {args.dataset_dir}")
    print(f"训练文件: {args.output_train}")
    print(f"验证文件: {args.output_val}")
    print("配置文件已更新" if updated else "配置文件未更新")
    
    print("\n下一步:")
    print("1. 运行训练: python train.py")

if __name__ == '__main__':
    main() 
