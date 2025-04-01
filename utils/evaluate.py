import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

from models.lprnet import build_lprnet, decode_ctc
from utils.dataset import get_val_dataloader, idx_to_text
from utils.utils import load_config, setup_logger, get_device, visualize_prediction, build_char_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LPRNet for license plate recognition')
    parser.add_argument('--config', type=str, default='config/lprnet_config.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, default='weights/best.pth',
                        help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default='output/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample predictions')
    parser.add_argument('--num_visualize', type=int, default=10,
                        help='Number of samples to visualize')
    
    return parser.parse_args()


def evaluate(model, val_loader, device, chars_list, output_dir=None, visualize=False, num_visualize=10):
    """Evaluate the model on the validation set"""
    model.eval()
    correct = 0
    total = 0
    
    all_pred_texts = []
    all_target_texts = []
    
    # Visualization setup
    if visualize and output_dir:
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        vis_indices = np.random.choice(len(val_loader.dataset), min(num_visualize, len(val_loader.dataset)), replace=False)
        vis_count = 0
    
    with torch.no_grad():
        for i, (images, targets, target_lengths) in enumerate(tqdm(val_loader, desc='Evaluating')):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(images)
            
            # Decode predictions
            batch_size = images.size(0)
            pred_texts = decode_ctc(logits, chars_list)
            target_texts = [idx_to_text(targets[j][:target_lengths[j]].tolist(), chars_list) for j in range(batch_size)]
            
            # Collect all predictions and targets for metrics
            all_pred_texts.extend(pred_texts)
            all_target_texts.extend(target_texts)
            
            # Update metrics
            correct += sum(pred == target for pred, target in zip(pred_texts, target_texts))
            total += batch_size
            
            # Visualize predictions if requested
            if visualize and output_dir and vis_count < num_visualize:
                for j in range(batch_size):
                    if vis_count >= num_visualize:
                        break
                    
                    # Visualize prediction
                    visualize_prediction(
                        images[j],
                        target_texts[j],
                        pred_texts[j],
                        os.path.join(output_dir, 'visualizations', f'sample_{vis_count}.png')
                    )
                    vis_count += 1
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Calculate per-character accuracy
    char_correct = 0
    char_total = 0
    for pred, target in zip(all_pred_texts, all_target_texts):
        for p, t in zip(pred[:len(target)], target):
            if p == t:
                char_correct += 1
            char_total += 1
    
    char_accuracy = char_correct / max(1, char_total)
    
    # Calculate metrics for each character position
    position_metrics = {}
    for pos in range(8):  # Assuming max plate length is 8
        pos_correct = 0
        pos_total = 0
        for pred, target in zip(all_pred_texts, all_target_texts):
            if pos < len(target) and pos < len(pred):
                pos_total += 1
                if pred[pos] == target[pos]:
                    pos_correct += 1
        
        if pos_total > 0:
            position_metrics[pos] = pos_correct / pos_total
    
    # Calculate character-level confusion matrix (for top N most common characters)
    top_n_chars = 20
    all_chars = set()
    for text in all_target_texts:
        all_chars.update(text)
    
    char_freq = {}
    for text in all_target_texts:
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1
    
    top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:top_n_chars]
    top_chars = [char for char, _ in top_chars]
    
    # Collect character-level predictions and targets
    char_preds = []
    char_targets = []
    for pred, target in zip(all_pred_texts, all_target_texts):
        for p, t in zip(pred[:len(target)], target):
            if t in top_chars:
                char_preds.append(p)
                char_targets.append(t)
    
    # Calculate confusion matrix
    if output_dir and len(char_targets) > 0:
        cm = confusion_matrix(char_targets, char_preds, labels=top_chars)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=top_chars, yticklabels=top_chars)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Character-level Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
    
    # Generate classification report
    if output_dir and len(char_targets) > 0:
        report = classification_report(char_targets, char_preds, labels=top_chars, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Character Accuracy: {char_accuracy:.4f}")
    print("Position-wise Accuracy:")
    for pos, acc in position_metrics.items():
        print(f"  Position {pos+1}: {acc:.4f}")
    
    # Save metrics to file
    if output_dir:
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Character Accuracy: {char_accuracy:.4f}\n")
            f.write("Position-wise Accuracy:\n")
            for pos, acc in position_metrics.items():
                f.write(f"  Position {pos+1}: {acc:.4f}\n")
    
        # Plot position-wise accuracy
        plt.figure(figsize=(10, 6))
        positions = list(position_metrics.keys())
        accuracies = [position_metrics[pos] for pos in positions]
        plt.bar([pos+1 for pos in positions], accuracies)
        plt.xlabel('Character Position')
        plt.ylabel('Accuracy')
        plt.title('Position-wise Character Accuracy')
        plt.xticks([pos+1 for pos in positions])
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'position_accuracy.png'))
        plt.close()
    
    return accuracy, char_accuracy, position_metrics


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.output_dir, name='lprnet_eval')
    logger.info(f"Config: {config}")
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Build character dictionary
    chars_dict, chars_list = build_char_dict(config)
    
    # Setup data loader
    logger.info("Setting up data loader...")
    val_loader = get_val_dataloader(config, chars_dict)
    logger.info(f"Num validation samples: {len(val_loader.dataset)}")
    
    # Build model
    logger.info("Building model...")
    model = build_lprnet(config)
    model = model.to(device)
    
    # Load weights
    logger.info(f"Loading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate model
    logger.info("Evaluating model...")
    accuracy, char_accuracy, position_metrics = evaluate(
        model,
        val_loader,
        device,
        chars_list,
        output_dir=args.output_dir,
        visualize=args.visualize,
        num_visualize=args.num_visualize
    )
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Character Accuracy: {char_accuracy:.4f}")
    logger.info("Position-wise Accuracy:")
    for pos, acc in position_metrics.items():
        logger.info(f"  Position {pos+1}: {acc:.4f}")
    
    logger.info(f"Evaluation completed. Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 
