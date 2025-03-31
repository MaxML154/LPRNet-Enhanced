import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import random

from models.lprnet import build_lprnet, CTCLoss, decode_ctc
from utils.dataset import get_train_dataloader, get_val_dataloader, idx_to_text
from utils.utils import load_config, setup_logger, get_device, save_checkpoint, load_checkpoint, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description='Train LPRNet for license plate recognition')
    parser.add_argument('--config', type=str, default='config/lprnet_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dirs(config):
    """Create necessary directories for outputs"""
    os.makedirs(config['OUTPUT']['DIR'], exist_ok=True)
    os.makedirs(config['OUTPUT']['WEIGHTS_DIR'], exist_ok=True)
    os.makedirs(config['OUTPUT']['LOGS_DIR'], exist_ok=True)


def get_lr_scheduler(optimizer, config):
    """Get learning rate scheduler based on config"""
    scheduler_type = config['TRAIN']['LR_SCHEDULER']
    
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['TRAIN']['LR_STEP_SIZE'], 
            gamma=0.1
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config['TRAIN']['LR_PLATEAU_FACTOR'], 
            patience=config['TRAIN']['LR_PLATEAU_PATIENCE'],
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['TRAIN']['EPOCHS'], 
            eta_min=1e-6
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, config, chars_list):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for i, (images, targets, target_lengths) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = torch.tensor(target_lengths, device=device)
        
        # 验证数据格式
        batch_size = images.size(0)
        if i == 0:
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Images shape: {images.shape}")
            logger.info(f"Targets shape: {targets.shape}")
            logger.info(f"Target lengths: {target_lengths}")
            
        # 确保target_lengths不超过targets的实际长度
        max_target_len = targets.size(1)
        for b in range(batch_size):
            if target_lengths[b] > max_target_len:
                logger.warning(f"Target length {target_lengths[b]} exceeds max length {max_target_len}")
                target_lengths[b] = max_target_len
        
        # Forward pass
        logits = model(images)  # shape: (batch_size, seq_len, num_classes)
        if i == 0:
            logger.info(f"Logits shape: {logits.shape}")
        
        # 计算logits_lengths
        logits_length = logits.size(1)
        logits_lengths = torch.full(size=(batch_size,), fill_value=logits_length, dtype=torch.long, device=device)
        
        # 添加额外检查
        if i == 0:
            logger.info(f"Logits lengths: {logits_lengths}")
            sum_target_lengths = target_lengths.sum().item()
            logger.info(f"Sum of target lengths: {sum_target_lengths}")
            
        try:
            # Compute loss with error handling
            loss = criterion(logits, targets, logits_lengths, target_lengths)
            
            # Compute accuracy
            pred_texts = decode_ctc(logits.detach(), chars_list)
            target_texts = [idx_to_text(targets[j][:target_lengths[j]].tolist(), chars_list) for j in range(batch_size)]
            correct = sum(pred == target for pred, target in zip(pred_texts, target_texts))
            acc = correct / batch_size
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_acc += acc
            
        except RuntimeError as e:
            logger.error(f"Runtime error in batch {i}: {str(e)}")
            logger.error(f"Batch size: {batch_size}")
            logger.error(f"Images shape: {images.shape}")
            logger.error(f"Targets shape: {targets.shape}")
            logger.error(f"Target lengths: {target_lengths}")
            logger.error(f"Logits shape: {logits.shape}")
            
            # 跳过这个批次
            continue
        
        # Update progress bar
        if (i + 1) % config['TRAIN']['PRINT_INTERVAL'] == 0:
            progress_bar.set_postfix({
                'Loss': f'{epoch_loss / (i + 1):.4f}',
                'Acc': f'{epoch_acc / (i + 1):.4f}'
            })
            
            # Log some predictions
            logger.info(f"Sample predictions (Epoch {epoch}, Batch {i+1}):")
            for j in range(min(3, batch_size)):
                logger.info(f"  True: {target_texts[j]}, Pred: {pred_texts[j]}")
    
    # Calculate epoch metrics
    num_batches = len(train_loader)
    epoch_loss /= num_batches
    epoch_acc /= num_batches
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, chars_list):
    """Validate the model"""
    model.eval()
    val_loss = 0
    val_acc = 0
    valid_batches = 0
    
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(val_loader, desc='Validation'):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = torch.tensor(target_lengths, device=device)
            
            # 验证数据格式
            batch_size = images.size(0)
            
            # 确保target_lengths不超过targets的实际长度
            max_target_len = targets.size(1)
            for b in range(batch_size):
                if target_lengths[b] > max_target_len:
                    print(f"Warning: Target length {target_lengths[b]} exceeds max length {max_target_len}")
                    target_lengths[b] = max_target_len
            
            # Forward pass
            logits = model(images)
            
            try:
                # Compute loss
                logits_lengths = torch.full(size=(batch_size,), fill_value=logits.size(1), dtype=torch.long, device=device)
                loss = criterion(logits, targets, logits_lengths, target_lengths)
                
                # Compute accuracy
                pred_texts = decode_ctc(logits, chars_list)
                target_texts = [idx_to_text(targets[j][:target_lengths[j]].tolist(), chars_list) for j in range(batch_size)]
                correct = sum(pred == target for pred, target in zip(pred_texts, target_texts))
                acc = correct / batch_size
                
                # Update metrics
                val_loss += loss.item()
                val_acc += acc
                valid_batches += 1
                
            except RuntimeError as e:
                print(f"Runtime error during validation: {str(e)}")
                print(f"Skipping batch with shapes: images={images.shape}, targets={targets.shape}, logits={logits.shape}")
                continue
    
    # Calculate validation metrics (only count valid batches)
    if valid_batches > 0:
        val_loss /= valid_batches
        val_acc /= valid_batches
    else:
        val_loss = float('inf')
        val_acc = 0
    
    return val_loss, val_acc


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Create directories
    create_dirs(config)
    
    # Setup logger
    logger = setup_logger(config['OUTPUT']['LOGS_DIR'])
    logger.info(f"Config: {config}")
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Setup data loaders
    logger.info("Setting up data loaders...")
    train_loader, chars_dict = get_train_dataloader(config)
    chars_list = [char for char in chars_dict.keys()]
    val_loader = get_val_dataloader(config, chars_dict)
    logger.info(f"Num training samples: {len(train_loader.dataset)}")
    logger.info(f"Num validation samples: {len(val_loader.dataset)}")
    
    # Build model
    logger.info("Building model...")
    model = build_lprnet(config)
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model)}")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['TRAIN']['LEARNING_RATE'],
        weight_decay=config['TRAIN']['WEIGHT_DECAY']
    )
    criterion = CTCLoss()
    scheduler = get_lr_scheduler(optimizer, config)
    
    # Resume training if specified
    start_epoch = 0
    best_acc = 0
    best_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_acc, best_loss = load_checkpoint(model, optimizer, scheduler, args.resume)
        logger.info(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}, best loss: {best_loss:.4f}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=config['OUTPUT']['LOGS_DIR'])
    
    # Training loop
    logger.info("Starting training...")
    no_improve_epochs = 0
    
    for epoch in range(start_epoch, config['TRAIN']['EPOCHS']):
        # Train for one epoch
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, config, chars_list)
        epoch_time = time.time() - epoch_start_time
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, chars_list)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{config['TRAIN']['EPOCHS']} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Time: {epoch_time:.2f}s")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            else:
                writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Save checkpoint periodically
        if (epoch + 1) % config['TRAIN']['SAVE_INTERVAL'] == 0 or is_best:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'best_acc': best_acc,
                'best_loss': best_loss
            }
            save_checkpoint(state, is_best, config['OUTPUT']['WEIGHTS_DIR'], f'checkpoint_epoch{epoch+1}.pth')
        
        # Early stopping
        if no_improve_epochs >= config['TRAIN']['EARLY_STOPPING_PATIENCE']:
            logger.info(f"No improvement for {no_improve_epochs} epochs. Early stopping.")
            break
    
    writer.close()
    logger.info(f"Training completed. Best validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main() 