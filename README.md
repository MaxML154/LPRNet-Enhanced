# LPRNet-Enhanced: Advanced Chinese License Plate Recognition

An enhanced License Plate Recognition system based on optimized LPRNet architecture with advanced data augmentation and improved recognition capabilities for Chinese license plates.

## Features

- **Multi-type License Plate Recognition**: Supports various Chinese license plate types including standard blue plates, green new energy plates, yellow plates, black plates, and double-row plates
- **License Plate Color Classification**: Additional classification branch identifies plate color (blue, green, yellow, black)
- **Advanced Data Augmentation**: Comprehensive augmentation pipeline including rotation, brightness/contrast adjustment, perspective transforms, and more
- **Skew Correction**: Automatic skew detection and correction using Hough transform to handle tilted license plates
- **Double-row Plate Optimization**: Special processing for double-row plates (yellow trailer plates and green tractor plates) with horizontal concatenation
- **Re-sampling Mechanism**: Weighted random sampling to balance training data distribution and improve accuracy on rare plate types
- **Flexible Training Modes**: Simple command line selection between basic, color classification, and full-featured training modes

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

However, if you think that's too much for your model, you can [click here](https://github.com/MaxML154/LPRNet-Enhanced/blob/main/data/README_dataset_reduction.md "Reduce the size of CBLPRD-330k") to learn how to reduce the size of dataset.

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

Evaluate the model using:

```bash
python test.py --weights weights/best.pth
```

## Demo

Run a simple demo:

```bash
python demo.py --image test_images/plate.jpg --weights weights/best.pth
```
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


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CBLPRD-330k dataset creators
- Original LPRNet implementation
- YOLOv5-LPRNet project by HuKai97 (https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition)
- crnn_plate_recognition by we0091234 (https://github.com/we0091234/crnn_plate_recognition)
- LPRNet_Pytorch by sirius-ai (https://github.com/sirius-ai/LPRNet_Pytorch)
