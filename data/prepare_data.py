import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import random
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare CBLPRD-330k dataset for LPRNet')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to CBLPRD-330k dataset directory (containing images)')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to train.txt from CBLPRD-330k dataset')
    parser.add_argument('--val_file', type=str, required=True,
                        help='Path to val.txt from CBLPRD-330k dataset')
    parser.add_argument('--output_dir', type=str, default='data/CBLPRD-330k',
                        help='Path to output directory for processed data')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to use (for testing with smaller dataset)')
    parser.add_argument('--no_copy', action='store_true',
                        help='Do not copy image files, just create data files with correct paths')
    
    return parser.parse_args()

def copy_data_files(args):
    """
    Copy and process train.txt and val.txt files
    """
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_train_file = 'data/train.txt'
    output_val_file = 'data/val.txt'
    
    # Process train file
    process_data_file(args.train_file, output_train_file, args.dataset_dir, args.output_dir, 
                     sample_size=args.sample_size, is_train=True, no_copy=args.no_copy)
    
    # Process val file
    sample_size_val = max(100, args.sample_size // 10) if args.sample_size else None
    process_data_file(args.val_file, output_val_file, args.dataset_dir, args.output_dir, 
                     sample_size=sample_size_val, is_train=False, no_copy=args.no_copy)

def process_data_file(input_file, output_file, input_dir, output_dir, sample_size=None, is_train=True, no_copy=False):
    """
    Process data file (train.txt or val.txt)
    """
    print(f"Processing {'train' if is_train else 'val'} file: {input_file}")
    
    # Read data file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]
    
    # Sample if needed
    if sample_size and sample_size < len(lines):
        print(f"Sampling {sample_size} out of {len(lines)} lines")
        lines = random.sample(lines, sample_size)
    
    processed_lines = []
    missing_files = 0
    
    # Process each line and copy images
    for line in tqdm(lines, desc=f"Processing {'train' if is_train else 'val'} file"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        img_path = parts[0]
        plate_text = parts[1]
        plate_type = parts[2] if len(parts) > 2 else ""
        
        # 处理不同格式的输入文件路径
        if os.path.isabs(img_path):
            # 已经是绝对路径
            abs_img_path = img_path
            img_filename = os.path.basename(img_path)
        else:
            img_filename = os.path.basename(img_path)
            # 尝试多种可能的路径
            possible_paths = [
                os.path.join(input_dir, img_path),  # 完整相对路径
                os.path.join(input_dir, img_filename),  # 仅使用文件名
            ]
            
            abs_img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    abs_img_path = path
                    break
            
            if not abs_img_path:
                missing_files += 1
                print(f"Warning: Cannot find image file: {img_path}")
                print(f"Tried paths: {possible_paths}")
                continue
        
        # 创建输出文件行
        if no_copy:
            # 如果不复制图片，使用原始图片的绝对路径
            processed_line = f"{abs_img_path} {plate_text}"
        else:
            # 复制图片到输出目录
            dst_path = os.path.join(output_dir, img_filename)
            try:
                shutil.copy(abs_img_path, dst_path)
                rel_path = os.path.join('CBLPRD-330k', img_filename)
                processed_line = f"{rel_path} {plate_text}"
            except Exception as e:
                print(f"Error copying {abs_img_path} to {dst_path}: {e}")
                # 使用原始路径
                processed_line = f"{abs_img_path} {plate_text}"
        
        # 添加车牌类型
        if plate_type:
            processed_line += f" {plate_type}"
        
        processed_lines.append(processed_line)
    
    # 创建目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入处理后的行到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(f"{line}\n")
    
    print(f"Processed {len(processed_lines)} {'train' if is_train else 'val'} samples")
    if missing_files > 0:
        print(f"Warning: {missing_files} image files were missing")
    
    print(f"Output file saved to {output_file}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process data files
    copy_data_files(args)
    
    print("Data preparation completed.")
    if not args.no_copy:
        print(f"Dataset directory: {args.output_dir}")
    print(f"Train file: data/train.txt")
    print(f"Val file: data/val.txt")
    
    # Print instructions
    print("\nNext steps:")
    print("1. Train the model: python train.py")
    print("2. Evaluate the model: python evaluate.py --weights weights/best.pth")
    print("3. Run the demo: python demo.py --image images/your_image.jpg --weights weights/best.pth")

if __name__ == '__main__':
    main() 
