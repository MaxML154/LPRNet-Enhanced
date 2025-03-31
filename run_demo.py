import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Run LPRNet demo with a sample license plate')
    parser.add_argument('--weights', type=str, default='weights/best.pth',
                        help='Path to model weights')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate the sample data without running demo')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of sample images to generate')
    parser.add_argument('--double_layer', action='store_true',
                        help='Process as double-layer license plate')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Check if the required directories exist
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
    
    if not os.path.exists('images'):
        os.makedirs('images', exist_ok=True)
    
    if not os.path.exists('data/CBLPRD-330k'):
        os.makedirs('data/CBLPRD-330k', exist_ok=True)
    
    # Generate sample data if needed
    sample_image_path = 'images/sample_plate.jpg'
    
    if not os.path.exists(sample_image_path) or not os.path.exists('data/train.txt') or not os.path.exists('data/val.txt'):
        print("Generating sample license plate data...")
        generate_cmd = [sys.executable, 'generate_samples.py', '--num_samples', str(args.num_samples)]
        subprocess.run(generate_cmd, check=True)
    
    if args.generate_only:
        print("Sample data generated. Exiting without running demo.")
        return
    
    # Run the demo
    print(f"Running demo with sample license plate: {sample_image_path}")
    
    demo_cmd = [
        sys.executable, 
        'demo.py',
        '--image', sample_image_path,
        '--weights', args.weights
    ]
    
    if args.double_layer:
        demo_cmd.append('--double_layer')
    
    try:
        subprocess.run(demo_cmd, check=True)
        print("Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        # Check if weights file exists
        if not os.path.exists(args.weights):
            print(f"Weights file not found: {args.weights}")
            print("You need to train the model first or specify the correct weights path.")
            print("To train the model, run: python train.py")
    
    print("\nNext steps:")
    print("1. To train the model with the generated samples: python train.py")
    print("2. To evaluate the model: python evaluate.py --weights weights/best.pth")
    print("3. To use your own license plate image: python demo.py --image path/to/your/image.jpg --weights weights/best.pth")

if __name__ == '__main__':
    main() 