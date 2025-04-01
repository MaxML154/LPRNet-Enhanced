# Dataset Reduction Tool for CBLPRD-330k

This tool helps reduce the size of the CBLPRD-330k license plate dataset while maintaining the proportion of different plate types. It's useful for training the LPRNet model with a smaller dataset to reduce computational requirements.

## Features

- Reduces dataset size while maintaining the proportion of different plate types
- Creates new dataset files (train2.txt and val2.txt) without modifying the original files
- Customizable reduction ratio
- Detailed reporting of dataset statistics before and after reduction

## Usage

```bash
python reduce_dataset.py [OPTIONS]
```

### Options

- `--train_file`: Path to original training data file (default: data/train.txt)
- `--val_file`: Path to original validation data file (default: data/val.txt)
- `--output_train_file`: Path to output reduced training data file (default: data/train2.txt)
- `--output_val_file`: Path to output reduced validation data file (default: data/val2.txt)
- `--config`: Path to configuration file (default: config/lprnet_config.yaml)
- `--reduction_ratio`: Ratio to reduce dataset to (default: 0.08 = 8%)
- `--seed`: Random seed for reproducibility (default: 42)

### Example

Reduce the dataset to 8% of its original size:

```bash
python reduce_dataset.py --train_file data/train1.txt --val_file data/val1.txt
```

Reduce the dataset to 5% of its original size:

```bash
python reduce_dataset.py --train_file data/train1.txt --val_file data/val1.txt --reduction_ratio 0.05
```

## After Running the Tool

After running the tool, you will need to update the configuration file to use the reduced dataset files. The tool will suggest the changes to make, such as:

```yaml
DATASET:
  TRAIN_FILE: "data/train2.txt"
  VAL_FILE: "data/val2.txt"
```

These changes will allow the LPRNet model to train using the reduced dataset.

## How It Works

1. The tool reads the original dataset files (train.txt and val.txt)
2. It groups entries by plate type (e.g., "黑色车牌", "双层黄牌", etc.)
3. For each plate type, it calculates how many samples to keep based on the reduction ratio
4. It randomly samples entries from each plate type to maintain the original proportions
5. Finally, it writes the sampled entries to new files (train2.txt and val2.txt)

This approach ensures that the reduced dataset maintains the same distribution of plate types as the original dataset, which is important for training a model that performs well on all types of license plates. 
