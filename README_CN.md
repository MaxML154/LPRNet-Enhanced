# LPRNet-Enhanced: 先进的中国车牌识别系统

基于优化的LPRNet架构的增强型车牌识别系统，具有高级数据增强和改进的中国车牌识别能力。

## 特点

- **多类型车牌识别**：支持各种中国车牌类型，包括标准蓝牌、绿色新能源车牌、黄牌、黑牌和双层车牌
- **车牌颜色分类**：附加的分类分支可识别车牌颜色（蓝色、绿色、黄色、黑色）
- **高级数据增强**：全面的增强管道，包括旋转、亮度/对比度调整、透视变换等
- **倾斜校正**：使用霍夫变换自动检测和校正倾斜车牌
- **双层车牌优化**：特殊处理双层车牌（黄色挂车牌和绿色拖拉机牌），采用水平拼接技术
- **重采样机制**：加权随机采样平衡训练数据分布，提高稀有车牌类型的识别准确率
- **灵活的训练模式**：简单的命令行选择基础、颜色分类和全功能训练模式

## 安装

1. 克隆此仓库：
```bash
git clone https://github.com/maxml154/LPRNet-Enhanced.git
cd LPRNet-Enhanced
```

2. 安装所需软件包：
```bash
pip install -r requirements.txt
```

## 数据集

本项目使用CBLPRD-330k数据集，这是一个包含33万张中国车牌图像的大型数据集。数据集可在以下地址获取：https://github.com/SunlifeV/CBLPRD-330k

### 使用原始数据集

要使用CBLPRD-330k数据集：
1. 下载数据集
2. 运行准备脚本：
```bash
python prepare_data.py --dataset_dir /path/to/CBLPRD-330k --train_file /path/to/train.txt --val_file /path/to/val.txt
```

### 使用生成的样本

如果您无法访问CBLPRD-330k数据集，可以生成合成样本进行测试：
```bash
python generate_samples.py --num_samples 500
```

这将创建：
- 500张合成车牌图像在`data/CBLPRD-330k/`目录
- 训练和验证数据文件在`data/train.txt`和`data/val.txt`
- 一个示例图像在`images/sample_plate.jpg`用于演示

如果您认为数据集太大，可以[点击这里](https://github.com/MaxML154/LPRNet-Enhanced/blob/main/data/README_dataset_reduction.md "减小CBLPRD-330k的大小")了解如何减小数据集大小。

## 训练

检查您的设置是否已准备好训练：
```bash
python check_setup.py
```

运行以下命令训练模型：
```bash
python train.py --config config/lprnet_config.yaml
```

您还可以选择不同的训练模式：

### 基础训练

```bash
python train.py --mode basic --config config/lprnet_config.yaml
```

### 带颜色分类的训练

```bash
python train.py --mode color --config config/lprnet_config.yaml
```

### 全功能训练（带重采样）

```bash
python train.py --mode full --config config/lprnet_config.yaml
```

### 恢复训练

```bash
python train.py --mode full --resume weights/checkpoint_epoch50.pth
```

## 评估

使用以下命令评估模型：

```bash
python test.py --weights weights/best.pth
```

## 演示

运行简单演示：

```bash
python demo.py --image test_images/plate.jpg --weights weights/best.pth
```

## 推理

对单个图像进行推理：
```bash
python demo.py --image path/to/image.jpg --weights path/to/weights.pth
```

使用生成的样本图像进行快速演示：
```bash
python run_demo.py --weights weights/best.pth
```

## 导出到ONNX

将训练好的模型导出为ONNX格式：
```bash
python export.py --weights path/to/weights.pth --output path/to/output.onnx
```

## 特殊情况

### 双层车牌

对于双层车牌，使用`--double_layer`标志：
```bash
python demo.py --image path/to/image.jpg --weights path/to/weights.pth --double_layer
```

## 模型架构

该模型架构基于优化的LPRNet，具有以下增强功能：

1. **基于ResNet的骨干网络**：可选择使用预训练的ResNet-50进行特征提取
2. **双分支设计**：专门针对中文字符和字母数字字符的专用分支
3. **颜色分类头**：用于车牌颜色分类的附加分支
4. **CTC损失**：用于字符序列识别的连接时序分类

## 数据预处理

数据预处理包括：

1. **倾斜校正**：使用霍夫变换检测并校正车牌倾斜角度
2. **双层车牌处理**：上下行适当间隔的水平拼接
3. **颜色特定增强**：基于车牌类型和颜色的对比度增强
4. **特殊字符处理**：对带有特殊后缀的车牌进行特殊处理（港、澳、使、领等）

## 项目结构

```
LPRNet-Enhanced/
├── config/             # 配置文件
├── data/               # 数据集目录
│   ├── CBLPRD-330k/    # 车牌图像
│   ├── train.txt       # 训练数据列表
│   └── val.txt         # 验证数据列表
├── images/             # 示例图像
├── logs/               # 训练日志
├── models/             # 模型定义
├── output/             # 训练输出目录
├── utils/              # 工具函数
├── weights/            # 模型权重
├── check_setup.py      # 设置检查脚本
├── demo.py             # 演示脚本
├── evaluate.py         # 评估脚本
├── export.py           # 模型导出脚本
├── generate_samples.py # 样本生成脚本
├── prepare_data.py     # 数据准备脚本
├── README.md           # 英文说明文件
├── README_CN.md        # 本中文说明文件
├── requirements.txt    # 依赖项
├── run_demo.py         # 快速演示脚本
└── train.py            # 训练脚本
```


## 许可证

本项目根据MIT许可证开源 - 有关详细信息，请参阅LICENSE文件。

## 致谢

- CBLPRD-330k数据集创建者
- 原始LPRNet实现
- HuKai97的YOLOv5-LPRNet项目 (https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition)
- we0091234的crnn_plate_recognition项目 (https://github.com/we0091234/crnn_plate_recognition)
- sirius-ai的LPRNet_Pytorch项目 (https://github.com/sirius-ai/LPRNet_Pytorch) 
