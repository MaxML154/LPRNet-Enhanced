#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training script for LPRNet-Enhanced.
"""

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from utils.configs.config import get_args, PLATE_CHARS, NUM_CLASSES, compute_lr_milestones
from utils.dataset.cblprd_dataset import CBLPRDDataset, collate_fn
from utils.model.lprnet import build_lprnet
from utils.loss import CTCLoss
from utils.evaluator import Evaluator
from utils.logger import Logger

def main():
    # Get arguments
    args = get_args()
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    np.random.seed(0)
    
    # Create model
    model = build_lprnet(
        model_type=args.model_type,
        num_classes=NUM_CLASSES,
        lpr_max_len=8,  # Typical max length for Chinese license plates
        dropout_rate=args.dropout_rate
    )
    model = model.to(device)
    print(f"Model: {args.model_type}")
    
    # 打印文件路径信息
    print(f"Data directory: {args.data_dir}")
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Test file: {args.test_file}")
    
    # Create datasets and data loaders
    train_dataset = CBLPRDDataset(
        data_root=args.data_dir,
        txt_file=args.train_file,
        is_train=True,
        input_shape=(args.input_width, args.input_height),
        use_resampling=args.use_resampling,
        correct_skew=args.correct_skew,
        process_double=args.process_double
    )
    
    val_dataset = CBLPRDDataset(
        data_root=args.data_dir,
        txt_file=args.val_file,
        is_train=False,
        input_shape=(args.input_width, args.input_height),
        use_resampling=False,
        correct_skew=args.correct_skew,
        process_double=args.process_double
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create loss function, optimizer, and evaluator
    criterion = CTCLoss(blank_label=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, args)
    
    evaluator = Evaluator(blank_label=0)
    
    # Create logger
    logger = Logger(log_interval=args.log_interval)
    
    # Training state
    start_epoch = 1
    best_accuracy = 0
    no_improvement_count = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device, logger
        )
        
        # Evaluate on validation set
        val_loss, val_acc = validate(
            model, val_loader, criterion, evaluator, device
        )
        
        # Update learning rate based on scheduler type
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_loss)  # ReduceLROnPlateau needs validation loss
            else:
                scheduler.step()
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        logger.log_epoch(
            epoch,
            {'loss': train_loss},
            {'accuracy': train_acc},
            {'loss': val_loss},
            {'accuracy': val_acc, 'lr': current_lr}
        )
        
        # Save checkpoint
        is_best = val_acc > best_accuracy
        best_accuracy = max(val_acc, best_accuracy)
        
        if is_best or epoch % args.save_interval == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': args.lr_scheduler,  # 保存调度器类型
            }
            
            # 保存调度器状态（如果存在）
            if scheduler is not None:
                checkpoint_dict['scheduler'] = scheduler.state_dict()
            
            save_checkpoint(checkpoint_dict, is_best, args.save_dir)
            
        # Early stopping
        if is_best:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= args.early_stopping:
            print(f"No improvement for {args.early_stopping} epochs, stopping training...")
            break
            
    # Plot training history
    logger.plot_metrics(save_path=os.path.join(args.save_dir, 'training_metrics.png'))
    
    # Load best model
    best_model_path = os.path.join(args.save_dir, 'model_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded best model with validation accuracy: {checkpoint['best_accuracy']:.4f}")
    
    # 根据参数决定是否进行测试评估
    test_metrics = None
    if args.test_after_train:
        print("\nRunning test evaluation...")
        # Final evaluation
        test_dataset = CBLPRDDataset(
            data_root=args.data_dir,
            txt_file=args.test_file,
            is_train=False,
            input_shape=(args.input_width, args.input_height),
            use_resampling=False,
            correct_skew=args.correct_skew,
            process_double=args.process_double
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        test_metrics = evaluator.evaluate_loader(model, test_loader, device)
        print("\nTest Results:")
        print(f"Sequence Accuracy: {test_metrics['sequence_accuracy']:.4f}")
        print(f"Character Accuracy: {test_metrics['character_accuracy']:.4f}")
    else:
        print("\nSkipping test evaluation as requested.")
        print(f"To evaluate the model, run: python test.py --weights {best_model_path} --data-dir {args.data_dir} --test-file {args.test_file}")
        
    return test_metrics

def create_scheduler(optimizer, args):
    """
    根据命令行参数创建学习率调度器
    
    参数:
        optimizer: 优化器
        args: 命令行参数
    
    返回:
        学习率调度器
    """
    scheduler_type = args.lr_scheduler.lower()
    
    if scheduler_type == 'step':
        # 每固定步长降低学习率
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_step, 
            gamma=args.lr_gamma
        )
    
    elif scheduler_type == 'multistep':
        # 在指定epoch降低学习率
        if args.lr_steps:
            # 使用命令行指定的降低点
            milestones = [int(x) for x in args.lr_steps.split(',')]
        else:
            # 根据总epoch数自动计算降低点
            milestones = compute_lr_milestones(args.epochs, args.lr_decay_points)
        
        print(f"Using MultiStepLR with milestones at epochs: {milestones}")
        return optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=milestones, 
            gamma=args.lr_gamma
        )
    
    elif scheduler_type == 'cosine':
        # 余弦退火学习率
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min
        )
    
    elif scheduler_type == 'reduce':
        # 当性能停止提升时降低学习率
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_gamma,
            patience=5,
            min_lr=args.lr_min
        )
    
    elif scheduler_type == 'plateau':
        # 当性能停止提升时降低学习率（与验证损失结合使用）
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=args.lr_gamma,
            patience=3,
            verbose=True
        )
    
    elif scheduler_type == 'onecycle':
        # OneCycleLR策略
        steps_per_epoch = 100  # 估计值，实际应该是len(train_loader)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,  # 最大学习率是初始学习率的10倍
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            pct_start=0.3  # 30%的训练用于预热
        )
    
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}', not using any scheduler.")
        return None

def train_epoch(model, train_loader, criterion, optimizer, epoch, device, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    batch_time = 0
    start_time = time.time()
    num_batches = len(train_loader)
    
    for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
        batch_start = time.time()
        
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(images)
        
        # Prepare for loss calculation
        batch_size = images.size(0)
        pred_lengths = torch.full((batch_size,), preds.size(1), dtype=torch.long, device=device)
        
        # Flatten labels for CTC loss
        labels_flat = torch.cat([labels[i, :label_lengths[i]] for i in range(batch_size)])
        
        # Calculate loss
        loss = criterion(preds, labels_flat, pred_lengths, label_lengths)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            target_strings = []
            for i, length in enumerate(label_lengths):
                target = labels[i][:length]
                target_chars = [PLATE_CHARS[idx.item()] for idx in target]
                target_strings.append("".join(target_chars))
                
            _, accuracy = Evaluator(blank_label=0).calculate_accuracy(preds, target_strings)
        
        # Update statistics
        total_loss += loss.item()
        total_acc += accuracy
        batch_time = time.time() - batch_start
        
        # Log progress
        logger.log_batch(
            epoch, batch_idx, num_batches, batch_time, batch_size,
            {'loss': loss.item()},
            {'accuracy': accuracy},
            optimizer.param_groups[0]['lr']
        )
        
    # Calculate average loss and accuracy
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, evaluator, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        val_metrics = evaluator.evaluate_loader(model, val_loader, device)
        
        # Calculate average loss
        for images, labels, label_lengths in val_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            preds = model(images)
            
            # Prepare for loss calculation
            batch_size = images.size(0)
            pred_lengths = torch.full((batch_size,), preds.size(1), dtype=torch.long, device=device)
            
            # Flatten labels for CTC loss
            labels_flat = torch.cat([labels[i, :label_lengths[i]] for i in range(batch_size)])
            
            # Calculate loss
            loss = criterion(preds, labels_flat, pred_lengths, label_lengths)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, val_metrics['sequence_accuracy']

def save_checkpoint(state, is_best, save_dir):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'checkpoint_epoch{state["epoch"]}.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_filename)

if __name__ == '__main__':
    main() 