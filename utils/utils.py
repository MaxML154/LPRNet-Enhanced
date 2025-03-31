import os
import yaml
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def load_config(config_path):
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(log_dir, name='lprnet'):
    """
    Setup logger for training and evaluation
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger


def get_device():
    """
    Get the device to use (CPU or GPU)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """
    Save checkpoint during training
    """
    torch.save(state, os.path.join(output_dir, filename))
    if is_best:
        torch.save(state, os.path.join(output_dir, 'best.pth'))


def load_checkpoint(model, optimizer, scheduler, path):
    """
    Load checkpoint for resuming training
    """
    if not os.path.exists(path):
        return 0, 0, 0
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['epoch'], checkpoint['best_acc'], checkpoint['best_loss']


def count_parameters(model):
    """
    Count the number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_prediction(image, true_text, pred_text, output_path=None):
    """
    Visualize the prediction result
    """
    plt.figure(figsize=(10, 5))
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        image = image.astype(np.uint8)
    
    plt.imshow(image)
    plt.title(f"True: {true_text}, Pred: {pred_text}")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def process_image(image_path, img_size=(94, 24)):
    """
    Process image for inference
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img = np.array(img) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)
    return img


def process_double_layer_plate(img):
    """
    Process double-layer license plates by merging top and bottom layers side by side
    """
    if isinstance(img, torch.Tensor):
        # Convert from tensor to numpy for processing
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        img = img.astype(np.uint8)
    
    h, w, c = img.shape
    img_upper = img[0:int(5/12*h), :]
    img_lower = img[int(1/3*h):, :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    
    # Convert back to proper format
    new_img = cv2.resize(new_img, (94, 24))
    new_img = new_img / 255.0
    new_img = (new_img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    new_img = new_img.transpose(2, 0, 1)
    new_img = torch.FloatTensor(new_img)
    new_img = new_img.unsqueeze(0)
    
    return new_img


def build_char_dict(config):
    """
    Build character dictionary from config
    """
    chars_list = (
        config['CHARS']['PROVINCES'] + 
        config['CHARS']['ALPHABETS'] + 
        config['CHARS']['DIGITS'] + 
        config['CHARS']['SEPARATOR']
    )
    chars_dict = {char: i+1 for i, char in enumerate(chars_list)}
    
    return chars_dict, chars_list 