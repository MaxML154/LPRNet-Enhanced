# LPRNet-Enhanced - Chinese License Plate Recognition

This project implements License Plate Recognition for Chinese license plates using LPRNet (License Plate Recognition Network) architecture. It's trained on the CBLPRD-330k dataset.

## Features

- License plate character recognition using LPRNet
- Support for different types of Chinese license plates
- Pre-trained model with CBLPRD-330k dataset
- ONNX model export for deployment
- Sample generation for testing without the actual dataset

## Installation

1. Clone this repository:
```bash
git clone https://github.com/maxml154/LPRNet-Enhanced.git
cd LPRNet-Enhanced
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the CBLPRD-330k dataset which contains 330,000 Chinese license plate images. The dataset is available at: https://github.com/SunlifeV/CBLPRD-330k

### Using the original dataset

To use the CBLPRD-330k dataset:
1. Download the dataset
2. Run the preparation script:
```bash
python prepare_data.py --dataset_dir /path/to/CBLPRD-330k --train_file /path/to/train.txt --val_file /path/to/val.txt
```

### Using generated samples

If you don't have access to the CBLPRD-330k dataset, you can generate synthetic samples for testing:
```bash
python generate_samples.py --num_samples 500
```

This will create:
- 500 synthetic license plate images in `data/CBLPRD-330k/`
- Train and validation data files in `data/train.txt` and `data/val.txt`
- A sample image in `images/sample_plate.jpg` for demo

## Training

To check if your setup is ready for training:
```bash
python check_setup.py
```

To train the model, run:
```bash
python train.py --config config/lprnet_config.yaml
```

## Evaluation

To evaluate the model, run:
```bash
python evaluate.py --weights weights/best.pth
```

## Inference

To run inference on a single image:
```bash
python demo.py --image path/to/image.jpg --weights path/to/weights.pth
```

For a quick demo using a generated sample image:
```bash
python run_demo.py --weights weights/best.pth
```

## Export to ONNX

To export the trained model to ONNX format:
```bash
python export.py --weights path/to/weights.pth --output path/to/output.onnx
```

## Special Cases

### Double-layer license plates

For double-layer license plates, use the `--double_layer` flag:
```bash
python demo.py --image path/to/image.jpg --weights path/to/weights.pth --double_layer
```

## Project Structure

```
LPRNet-Enhanced/
├── config/             # Configuration files
├── data/               # Dataset directory
│   ├── CBLPRD-330k/    # License plate images
│   ├── train.txt       # Training data list
│   └── val.txt         # Validation data list
├── images/             # Sample images for demo
├── logs/               # Training logs
├── models/             # Model definitions
├── output/             # Output directory for training
├── utils/              # Utility functions
├── weights/            # Model weights
├── check_setup.py      # Setup check script
├── demo.py             # Demo script
├── evaluate.py         # Evaluation script
├── export.py           # Model export script
├── generate_samples.py # Sample generation script
├── prepare_data.py     # Data preparation script
├── README.md           # This file
├── requirements.txt    # Dependencies
├── run_demo.py         # Quick demo script
└── train.py            # Training script
```

## License

This project is open-source under the MIT License. See LICENSE file for more details.

## Acknowledgments

- [CBLPRD-330k](https://github.com/SunlifeV/CBLPRD-330k) for the dataset
- [crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition) for reference implementation
- Original LPRNet implementation references
