import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image

from models.lprnet import build_lprnet
from utils.utils import load_config, process_image


def parse_args():
    parser = argparse.ArgumentParser(description='Export LPRNet model to ONNX')
    parser.add_argument('--config', type=str, default='config/lprnet_config.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--output', type=str, default='weights/lprnet.onnx',
                        help='Path to save ONNX model')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model (requires onnx-simplifier)')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to test image for verifying the exported model')
    
    return parser.parse_args()


def export_to_onnx(model, output_path, input_shape, simplify=False):
    """Export PyTorch model to ONNX format"""
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_shape[1], input_shape[0], device='cpu')
    
    # Export to ONNX
    torch.onnx.export(
        model,                                       # model being run
        dummy_input,                                 # model input (or a tuple for multiple inputs)
        output_path,                                 # where to save the model
        export_params=True,                          # store the trained parameter weights inside the model file
        opset_version=12,                            # the ONNX version to export the model to
        do_constant_folding=True,                    # whether to execute constant folding for optimization
        input_names=['input'],                       # the model's input names
        output_names=['output'],                     # the model's output names
        dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                      'output': {0: 'batch_size'}}
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Simplify the model if requested
    if simplify:
        try:
            from onnxsim import simplify
            print("Simplifying ONNX model...")
            simplified_model, check = simplify(onnx_model)
            if check:
                onnx.save(simplified_model, output_path)
                print(f"Simplified model saved to {output_path}")
            else:
                print("Simplified model could not be validated. Using original model.")
        except ImportError:
            print("onnx-simplifier not installed. Skipping simplification.")
            print("Install with: pip install onnx-simplifier")
    
    print(f"ONNX model exported to {output_path}")
    return output_path


def test_onnx_model(onnx_path, test_image_path, input_shape):
    """Test the exported ONNX model with a sample image"""
    # Load and preprocess the image
    img = process_image(test_image_path, input_shape)
    input_data = img.numpy()
    
    # Create ONNX runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    output = session.run([output_name], {input_name: input_data})[0]
    
    # Print output shape
    print(f"ONNX model output shape: {output.shape}")
    
    return True


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Build model
    print("Building model...")
    model = build_lprnet(config)
    
    # Load weights
    print(f"Loading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    # Export to ONNX
    input_shape = config['MODEL']['INPUT_SIZE']
    onnx_path = export_to_onnx(model, args.output, input_shape, args.simplify)
    
    # Test exported model
    if args.test_image:
        print(f"Testing ONNX model with image: {args.test_image}")
        success = test_onnx_model(onnx_path, args.test_image, input_shape)
        if success:
            print("ONNX model test passed!")
    
    print("Export completed successfully!")


if __name__ == '__main__':
    main() 