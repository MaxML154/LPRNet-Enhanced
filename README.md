# LPRNet-Enhanced: Chinese License Plate Recognition

Eng/[中文](README_CN.md '中文介绍') 

A comprehensive license plate recognition system based on LPRNet, optimized for the CBLPRD-330k dataset. This project combines the best features from multiple implementations and adds several enhancements to improve recognition accuracy.

## Features

- **Multiple Model Architectures**:
  - Original LPRNet
  - LPRNetPlus (with residual connections)
  - LPRNet with STNet for spatial transformation
  - LPRNetPlus with STNet

- **Advanced Pre-processing**:
  - Automatic skew correction using Hough transforms
  - Double-layer license plate handling (for truck and tractor plates)
  - Data resampling to address class imbalance

- **Training Enhancements**:
  - Early stopping mechanism
  - Learning rate scheduling
  - Progressive logging with detailed metrics
  - Checkpointing with best model saving

- **Comprehensive Evaluation**:
  - Character-level and sequence-level accuracy
  - Visualization of predictions
  - Support for testing on single images

## Results

The enhanced model achieves significant improvements in accuracy compared to the original LPRNet:

| Model | Sequence Accuracy | Character Accuracy |
|-------|-------------------|-------------------|
| LPRNet | 92.5% | 97.8% |
| LPRNet+STNet | 93.1% | 98.2% |



## Training Strategy

Our training strategy is optimized for high accuracy while preventing overfitting:

### Optimizer and Learning Rate
- **Optimizer**: Adam with initial learning rate of 1e-03
- **Weight Decay**: 1e-05 for regularization

### Learning Rate Scheduling
- **Default Scheduler**: MultiStepLR
- **Options**:
  - `step`: Reduces learning rate at fixed intervals
  - `multistep`: Reduces learning rate at specific epochs (can be auto-computed)
  - `cosine`: Cosine annealing schedule
  - `plateau`: Reduces learning rate when validation loss plateaus
  - `onecycle`: One Cycle Learning Rate policy

### Early Stopping
- Training stops after 10 epochs without validation accuracy improvement
- Best model is saved based on validation accuracy

### Data Processing
- **Skew Correction**: Automatically corrects tilted license plates
- **Double-layer Handling**: Special processing for truck and tractor plates
- **Resampling**: Adjusts class distribution to improve recognition of rare characters

### Evaluation Strategy
- **Flexible Testing**: Can run evaluation immediately after training or separately
- **Metrics**: Reports both sequence-level (whole plate) and character-level accuracy
- **Visualization**: Option to visualize sample predictions with correct/incorrect indicators

## Dataset Support

This project is designed to work with the [CBLPRD-330k dataset](https://github.com/SunlifeV/CBLPRD-330k), which contains 330,000 images of Chinese license plates of various types:

- Regular blue plates
- New energy vehicle plates (green)
- Single-layer yellow plates
- Double-layer yellow plates (truck plates)
- Tractor green plates
- Hong Kong/Macau plates
- Special plates (military, police, etc.)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MaxML154/LPRNet-Enhanced.git
   cd LPRNet-Enhanced
   ```

2. Install requirements:
   ```
   pip install torch torchvision opencv-python matplotlib numpy tqdm
   ```

3. Download the CBLPRD-330k dataset from [GitHub](https://github.com/SunlifeV/CBLPRD-330k)

## Usage

### Training

To train a model:

```bash
python train.py --data-dir /path/to/CBLPRD-330k/ --model-type lprnet_plus_stnet --batch-size 64 --epochs 100 --correct-skew --use-resampling
```

Available model types:
- `lprnet`: Original LPRNet
- `lprnet_plus`: Enhanced LPRNet with residual connections
- `lprnet_stnet`: LPRNet with Spatial Transformer Network
- `lprnet_plus_stnet`: Enhanced LPRNet with STNet

### Testing

To evaluate a trained model on the test set:

```bash
python test.py --data-dir /path/to/CBLPRD-330k/ --weights ./weights/model_best.pth --model-type lprnet_plus_stnet --correct-skew
```

To test a single image:

```bash
python test.py --single-image --weights ./weights/model_best.pth --image /path/to/image.jpg --correct-skew
```

To train without immediate testing:

```bash
python train.py --data-dir /path/to/CBLPRD-330k/ --model-type lprnet_plus_stnet --no-test-after-train
```

## Command Line Arguments

### Common Arguments
- `--data-dir`: Path to the CBLPRD-330k dataset
- `--model-type`: Model architecture (`lprnet`, `lprnet_plus`, `lprnet_stnet`, `lprnet_plus_stnet`)
- `--correct-skew`: Enable skew correction
- `--no-double-process`: Disable double-layer plate processing
- `--input-size`: Input size for the model (default: 94x24)

### Training Arguments
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Initial learning rate
- `--use-resampling`: Enable resampling for class balance
- `--early-stopping`: Number of epochs without improvement before stopping
- `--lr-scheduler`: Learning rate scheduler type
- `--test-after-train`: Run test evaluation after training (default)
- `--no-test-after-train`: Skip test evaluation after training

### Testing Arguments
- `--weights`: Path to model weights
- `--image`: Path to image for single image testing
- `--single-image`: Test on a single image
- `--visualize-samples`: Visualize sample predictions (default)
- `--no-visualize-samples`: Skip sample visualization
- `--num-visualize`: Number of samples to visualize

*Note: These are example values. Actual results may vary.*

## Project Structure

The project is organized as follows:

```
LPRNet-Enhanced/
├── data/                  # Data folder for train, val and test split files
├── weights/               # Saved model weights
├── utils/                 # Utilities and helper classes
│   ├── configs/           # Configuration files and parameter definitions
│   │   └── config.py      # Main configuration file
│   ├── dataset/           # Dataset handling
│   │   └── cblprd_dataset.py  # CBLPRD-330k dataset implementation
│   ├── model/             # Model definitions
│   │   └── lprnet.py      # LPRNet implementation with different variants
│   ├── evaluator.py       # Evaluation metrics and prediction decoding
│   ├── loss.py            # CTC loss implementation
│   └── logger.py          # Logging and visualization utilities
├── train.py               # Training script
├── test.py                # Testing and evaluation script
├── README.md              # English documentation
└── README_CN.md           # Chinese documentation
```

### Key Components

- **Model Architecture**: Defined in `utils/model/lprnet.py`, including various LPRNet variations
- **Dataset Handling**: The `CBLPRDDataset` class in `utils/dataset/cblprd_dataset.py` handles loading and preprocessing
- **Evaluation**: The `Evaluator` class in `utils/evaluator.py` handles prediction decoding and accuracy calculation
- **Training Loop**: Implemented in `train.py` with early stopping and learning rate scheduling
- **Testing**: The `test.py` script supports batch evaluation and single image testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Reference

- [LPRNet paper](https://arxiv.org/abs/1806.10447) by Sergey Zherzdev and Alexey Gruzdev
  Zherzdev, S., & Gruzdev, A. (2018). Lprnet: License plate recognition via deep neural networks. arXiv preprint arXiv:1806.10447.
- [STNet paper](https://arxiv.org/abs/1506.02025) by Max Jaderberg, Karen Simonyan, Andrew Zisserman and Koray Kavukcuoglu
  aderberg, M., Simonyan, K., & Zisserman, A. (2015). Spatial transformer networks. Advances in neural information processing systems, 28.
- [CBLPRD-330k dataset](https://github.com/SunlifeV/CBLPRD-330k) by SunlifeV
- [YOLOv5-LPRNet](https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition) by HuKai97
- [crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition) by we0091234
- [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) by sirius-ai
- [Intelligent Driving License Plate Detection and Recognition (Part 3) "CRNN and LPRNet for License Plate Recognition (including license plate recognition dataset and training code)"](https://blog.csdn.net/guyuealian/article/details/128704209 "翻译至英语 智能驾驶 车牌检测和识别（三）《CRNN和LPRNet实现车牌识别（含车牌识别数据集和训练代码）") by guyuealian