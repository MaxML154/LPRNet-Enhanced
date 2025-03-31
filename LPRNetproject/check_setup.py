import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import argparse

from models.lprnet import build_lprnet
from utils.utils import load_config, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Check LPRNet project setup')
    parser.add_argument('--config', type=str, default='config/lprnet_config.yaml',
                        help='Path to config file')
    
    return parser.parse_args()


def check_pytorch():
    """Check PyTorch installation"""
    print("\n=== Checking PyTorch Installation ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available, using CPU")
    
    # Create a small tensor to test
    try:
        x = torch.rand(5, 3)
        print("Created test tensor successfully")
        device = get_device()
        print(f"Using device: {device}")
        x = x.to(device)
        print("Moved tensor to device successfully")
    except Exception as e:
        print(f"Error creating/moving tensor: {e}")


def check_model(config):
    """Check model creation"""
    print("\n=== Checking Model Creation ===")
    try:
        model = build_lprnet(config)
        print("Model created successfully")
        print(f"Model architecture:\n{model}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {params:,}")
        
        # Test forward pass
        device = get_device()
        model = model.to(device)
        dummy_input = torch.randn(1, 3, config['MODEL']['INPUT_SIZE'][1], config['MODEL']['INPUT_SIZE'][0], device=device)
        output = model(dummy_input)
        print(f"Forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"Error creating/testing model: {e}")


def check_directories():
    """Check if required directories exist"""
    print("\n=== Checking Directories ===")
    required_dirs = [
        'config',
        'data',
        'models',
        'utils',
        'weights',
        'output',
        'logs',
        'images'
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print(f"✓ Directory '{dir_name}' exists")
        else:
            print(f"✗ Directory '{dir_name}' does not exist")


def check_files():
    """Check if required files exist"""
    print("\n=== Checking Files ===")
    required_files = [
        'config/lprnet_config.yaml',
        'models/lprnet.py',
        'utils/dataset.py',
        'utils/utils.py',
        'train.py',
        'evaluate.py',
        'demo.py',
        'export.py',
        'requirements.txt'
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name) and os.path.isfile(file_name):
            print(f"✓ File '{file_name}' exists")
        else:
            print(f"✗ File '{file_name}' does not exist")


def check_data_files():
    """Check data files"""
    print("\n=== Checking Data Files ===")
    data_files = [
        'data/train.txt',
        'data/val.txt'
    ]
    
    for file_name in data_files:
        if os.path.exists(file_name) and os.path.isfile(file_name):
            with open(file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 0 and not lines[0].startswith('#'):
                    print(f"✓ File '{file_name}' exists and contains data")
                else:
                    print(f"⚠ File '{file_name}' exists but appears to be empty or just contains comments")
                    print(f"  Please copy the actual data from CBLPRD-330k dataset")
        else:
            print(f"✗ File '{file_name}' does not exist")
            
    dataset_dir = 'data/CBLPRD-330k'
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        files = os.listdir(dataset_dir)
        if len(files) > 0:
            print(f"✓ Directory '{dataset_dir}' exists and contains {len(files)} files")
        else:
            print(f"⚠ Directory '{dataset_dir}' exists but is empty")
            print("  Please download the CBLPRD-330k dataset and place it in this directory")
    else:
        print(f"⚠ Directory '{dataset_dir}' does not exist or is empty")
        print("  Please create this directory and add the CBLPRD-330k dataset")


def main():
    # Parse arguments
    args = parse_args()
    
    # Print header
    print("="*50)
    print("LPRNet Project Setup Check")
    print("="*50)
    
    # Check directories
    check_directories()
    
    # Check files
    check_files()
    
    # Check data files
    check_data_files()
    
    # Load config
    try:
        config = load_config(args.config)
        print(f"\n✓ Config file '{args.config}' loaded successfully")
    except Exception as e:
        print(f"\n✗ Error loading config file: {e}")
        return
    
    # Check PyTorch installation
    check_pytorch()
    
    # Check model creation
    check_model(config)
    
    # Print final message
    print("\n" + "="*50)
    print("Setup Check Complete")
    print("="*50)
    
    print("\nNext steps:")
    print("1. Download the CBLPRD-330k dataset")
    print("2. Place the images in data/CBLPRD-330k directory")
    print("3. Copy the contents of train.txt and val.txt from the dataset to data/train.txt and data/val.txt")
    print("4. Install the required dependencies: pip install -r requirements.txt")
    print("5. Train the model: python train.py")
    print("6. Evaluate the model: python evaluate.py --weights weights/best.pth")
    print("7. Run the demo: python demo.py --image images/your_image.jpg --weights weights/best.pth")
    print("8. Export to ONNX: python export.py --weights weights/best.pth --output weights/lprnet.onnx")


if __name__ == '__main__':
    main() 