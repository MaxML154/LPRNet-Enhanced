import os
import argparse
import random
import yaml
from tqdm import tqdm
from collections import defaultdict
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Reduce CBLPRD-330k dataset size while maintaining plate type proportions')
    parser.add_argument('--train_file', type=str, default='data/train.txt',
                        help='Path to train.txt file (default: data/train.txt)')
    parser.add_argument('--val_file', type=str, default='data/val.txt',
                        help='Path to val.txt file (default: data/val.txt)')
    parser.add_argument('--output_train_file', type=str, default='data/train2.txt',
                        help='Path to output reduced train file (default: data/train2.txt)')
    parser.add_argument('--output_val_file', type=str, default='data/val2.txt',
                        help='Path to output reduced val file (default: data/val2.txt)')
    parser.add_argument('--config', type=str, default='config/lprnet_config.yaml',
                        help='Path to config file (default: config/lprnet_config.yaml)')
    parser.add_argument('--reduction_ratio', type=float, default=0.08,
                        help='Ratio to reduce dataset to (default: 0.08 = 8%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def read_data_file(file_path):
    """Read data file and return lines"""
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def parse_lines(lines):
    """Parse lines and group them by plate type"""
    plate_types_dict = defaultdict(list)
    unknown_type_count = 0
    
    for line in tqdm(lines, desc="Parsing dataset"):
        parts = line.strip().split()
        
        # Check if line has plate type
        if len(parts) >= 3:
            plate_type = parts[2]
            plate_types_dict[plate_type].append(line)
        else:
            plate_types_dict["unknown"].append(line)
            unknown_type_count += 1
    
    if unknown_type_count > 0:
        print(f"Warning: {unknown_type_count} entries have unknown plate type")
    
    return plate_types_dict

def sample_proportionally(plate_types_dict, reduction_ratio):
    """Sample entries from each plate type proportionally"""
    sampled_lines = []
    
    total_lines = sum(len(lines) for lines in plate_types_dict.values())
    print(f"Total entries: {total_lines}")
    
    target_total = math.floor(total_lines * reduction_ratio)
    print(f"Target after reduction: {target_total} (reduction ratio: {reduction_ratio:.2%})")
    
    # Calculate how many samples to take from each plate type
    type_counts = {}
    for plate_type, lines in plate_types_dict.items():
        count = len(lines)
        type_counts[plate_type] = {
            'original': count,
            'ratio': count / total_lines if total_lines > 0 else 0,
            'target': math.floor(count * reduction_ratio)
        }
    
    # Sample from each plate type
    print("\nSampling by plate type:")
    for plate_type, info in type_counts.items():
        original_count = info['original']
        target_count = info['target']
        print(f"  {plate_type}: {original_count} -> {target_count} ({info['ratio']:.2%})")
        
        # Sample lines for this plate type
        if target_count > 0:
            sampled = random.sample(plate_types_dict[plate_type], target_count)
            sampled_lines.extend(sampled)
    
    print(f"Total sampled entries: {len(sampled_lines)}")
    return sampled_lines

def write_output_file(lines, output_file):
    """Write sampled lines to output file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write lines to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")
    
    print(f"Output file saved to: {output_file}")

def reduce_dataset(input_file, output_file, reduction_ratio):
    """Reduce dataset size while maintaining plate type proportions"""
    # Read input file
    lines = read_data_file(input_file)
    print(f"Original dataset size: {len(lines)} entries")
    
    # Parse lines by plate type
    plate_types_dict = parse_lines(lines)
    
    # Sample proportionally
    sampled_lines = sample_proportionally(plate_types_dict, reduction_ratio)
    
    # Write output file
    write_output_file(sampled_lines, output_file)
    
    return len(lines), len(sampled_lines)

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    plate_types = config['DATASET']['PLATE_TYPES']
    print(f"Found {len(plate_types)} plate types in config: {', '.join(plate_types)}")
    
    # Process train file
    print("\n=== Processing Training Data ===")
    train_original, train_reduced = reduce_dataset(
        args.train_file, args.output_train_file, args.reduction_ratio
    )
    
    # Process val file
    print("\n=== Processing Validation Data ===")
    val_original, val_reduced = reduce_dataset(
        args.val_file, args.output_val_file, args.reduction_ratio
    )
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Training data: {train_original} -> {train_reduced} entries ({train_reduced/train_original:.2%})")
    print(f"Validation data: {val_original} -> {val_reduced} entries ({val_reduced/val_original:.2%})")
    print(f"Total: {train_original + val_original} -> {train_reduced + val_reduced} entries ({(train_reduced + val_reduced)/(train_original + val_original):.2%})")
    
    print("\nReduced dataset files:")
    print(f"  Training data: {args.output_train_file}")
    print(f"  Validation data: {args.output_val_file}")
    
    print("\nTo use the reduced dataset, modify the config file:")
    print(f"  DATASET:")
    print(f"    TRAIN_FILE: \"{args.output_train_file}\"")
    print(f"    VAL_FILE: \"{args.output_val_file}\"")

if __name__ == '__main__':
    main() 