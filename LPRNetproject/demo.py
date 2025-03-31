import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from models.lprnet import build_lprnet, decode_ctc
from utils.utils import load_config, get_device, process_image, build_char_dict, process_double_layer_plate


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for LPRNet license plate recognition')
    parser.add_argument('--config', type=str, default='config/lprnet_config.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, default='weights/best.pth',
                        help='Path to model weights')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output image (if not specified, image will be displayed)')
    parser.add_argument('--double_layer', action='store_true',
                        help='Process as double-layer license plate')
    
    return parser.parse_args()


def visualize_result(image_path, pred_text, output_path=None, double_layer=False):
    """Visualize recognition result"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for better text rendering
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font that supports Chinese characters
    try:
        # Try to find a suitable font (may need to be adjusted based on OS)
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # Windows
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            'simhei.ttf',  # Try current directory
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 30)
                break
                
        if font is None:
            # Fallback to default
            font = ImageFont.load_default()
            
    except Exception as e:
        print(f"Error loading font: {e}")
        font = ImageFont.load_default()
    
    # Calculate text position (at the bottom of the image)
    img_width, img_height = pil_img.size
    text_width = len(pred_text) * 20  # Approximate text width
    x = max(0, (img_width - text_width) // 2)
    y = img_height - 40
    
    # Add info text
    plate_type = "双层牌照" if double_layer else "单层牌照"
    info_text = f"识别结果: {pred_text} ({plate_type})"
    
    # Draw semi-transparent background for text
    text_bg = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw_bg = ImageDraw.Draw(text_bg)
    draw_bg.rectangle([(0, y-5), (img_width, y+35)], fill=(0, 0, 0, 128))
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), text_bg).convert('RGB')
    
    # Create new draw object for the composite image
    draw = ImageDraw.Draw(pil_img)
    
    # Draw text
    draw.text((x, y), info_text, font=font, fill=(255, 255, 255))
    
    # Convert back to numpy for display/save with matplotlib
    img = np.array(pil_img)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"License Plate Recognition: {pred_text}")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Result saved to {output_path}")
    else:
        plt.show()


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Build character dictionary
    chars_dict, chars_list = build_char_dict(config)
    
    # Build model
    model = build_lprnet(config)
    model = model.to(device)
    
    # Load weights
    print(f"Loading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process image
    print(f"Processing image: {args.image}")
    img_size = config['MODEL']['INPUT_SIZE']
    img = process_image(args.image, img_size)
    
    # Process as double-layer plate if specified
    if args.double_layer:
        img = process_double_layer_plate(img)
    
    # Move to device
    img = img.to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img)
        pred_texts = decode_ctc(logits, chars_list)
        pred_text = pred_texts[0]
    
    # Print result
    print(f"Predicted license plate: {pred_text}")
    
    # Visualize result
    visualize_result(args.image, pred_text, args.output, args.double_layer)


if __name__ == '__main__':
    main() 