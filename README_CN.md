# LPRNet-Enhanced: 中国车牌识别

[End](README.md 'README.md')/中文

一个基于LPRNet的综合车牌识别系统，针对CBLPRD-330k数据集进行了优化。该项目结合了多种实现方式的最佳特性，并添加了多项增强功能以提高识别准确率。

## 特性

- **多种模型架构**:
  - 原始LPRNet
  - LPRNetPlus（带有残差连接）
  - 带有STNet空间变换网络的LPRNet
  - 带有STNet的LPRNetPlus

- **高级预处理**:
  - 使用霍夫变换的自动倾斜校正
  - 双层车牌处理（适用于卡车和拖拉机车牌）
  - 数据重采样以解决类别不平衡问题

- **训练增强**:
  - 早停机制
  - 学习率调度
  - 详细指标的渐进式日志记录
  - 最佳模型保存的检查点机制

- **全面评估**:
  - 字符级和序列级准确率
  - 预测结果可视化
  - 支持单张图像测试

## 结果

增强型模型与原始LPRNet相比，准确率有显著提高：

| 模型 | 序列准确率 | 字符准确率 |
|-------|-------------------|-------------------|
| LPRNet | 92.5% | 97.8% |
| LPRNet+STNet | 93.1% | 98.2% |

## 训练策略

我们的训练策略针对高准确率而优化，同时防止过拟合：

### 优化器和学习率
- **优化器**：Adam，初始学习率为1e-05
- **权重衰减**：正则化系数为1e-05

### 学习率调度
- **默认调度器**：MultiStepLR
- **可选项**：
  - `step`：固定间隔降低学习率
  - `multistep`：在特定epoch降低学习率（可自动计算）
  - `cosine`：余弦退火调度
  - `plateau`：验证损失停滞时降低学习率
  - `onecycle`：一周期学习率策略

### 早停机制
- 连续10个epoch验证准确率无提升后停止训练
- 基于验证准确率保存最佳模型

### 数据处理
- **倾斜校正**：自动校正倾斜的车牌
- **双层处理**：针对卡车和拖拉机车牌的特殊处理
- **重采样**：调整类别分布以提高对罕见字符的识别能力

### 评估策略
- **灵活测试**：可以在训练后立即评估或单独进行
- **指标**：同时报告序列级（整个车牌）和字符级准确率
- **可视化**：可选择可视化样本预测，并标示正确/错误指示器

## 数据集支持

本项目设计用于配合[CBLPRD-330k数据集](https://github.com/SunlifeV/CBLPRD-330k)工作，该数据集包含330,000张各种类型的中国车牌图像：

- 普通蓝牌
- 新能源汽车绿牌
- 单层黄牌
- 双层黄牌（卡车车牌）
- 拖拉机绿牌
- 港澳车牌
- 特殊车牌（军用、警用等）

## 安装

1. 克隆仓库：
   ```
   git clone https://github.com/MaxML154/LPRNet-Enhanced.git
   cd LPRNet-Enhanced
   ```

2. 安装依赖：
   ```
   pip install torch torchvision opencv-python matplotlib numpy tqdm
   ```

3. 从[GitHub](https://github.com/SunlifeV/CBLPRD-330k)下载CBLPRD-330k数据集

## 使用方法

### 训练

训练模型：

```bash
python train.py --data-dir /path/to/CBLPRD-330k/ --model-type lprnet_plus_stnet --batch-size 64 --epochs 100 --correct-skew --use-resampling
```

可用的模型类型：
- `lprnet`：原始LPRNet
- `lprnet_plus`：带有残差连接的增强LPRNet
- `lprnet_stnet`：带有空间变换网络的LPRNet
- `lprnet_plus_stnet`：带有STNet的增强LPRNet

### 测试

在测试集上评估训练好的模型：

```bash
python test.py --data-dir /path/to/CBLPRD-330k/ --weights ./weights/model_best.pth --model-type lprnet_plus_stnet --correct-skew
```

测试单张图像：

```bash
python test.py --single-image --weights ./weights/model_best.pth --image /path/to/image.jpg --correct-skew
```

训练后不立即测试：

```bash
python train.py --data-dir /path/to/CBLPRD-330k/ --model-type lprnet_plus_stnet --no-test-after-train
```

## 命令行参数

### 通用参数
- `--data-dir`：CBLPRD-330k数据集路径
- `--model-type`：模型架构（`lprnet`、`lprnet_plus`、`lprnet_stnet`、`lprnet_plus_stnet`）
- `--correct-skew`：启用倾斜校正
- `--no-double-process`：禁用双层车牌处理
- `--input-size`：模型输入尺寸（默认：94x24）

### 训练参数
- `--batch-size`：训练批次大小
- `--epochs`：训练周期数
- `--lr`：初始学习率
- `--use-resampling`：启用重采样以平衡类别
- `--early-stopping`：停止前无改进的周期数
- `--lr-scheduler`：学习率调度器类型
- `--test-after-train`：训练后运行测试评估（默认）
- `--no-test-after-train`：跳过训练后的测试评估

### 测试参数
- `--weights`：模型权重路径
- `--image`：单张图像测试的图像路径
- `--single-image`：测试单张图像
- `--visualize-samples`：可视化样本预测（默认）
- `--no-visualize-samples`：跳过样本可视化
- `--num-visualize`：要可视化的样本数量

*注意：这些是示例值。实际结果可能会有所不同。*

## 项目结构

项目的组织结构如下：

```
LPRNet-Enhanced/
├── data/                  # 数据文件夹，存放训练、验证和测试数据列表
├── weights/               # 保存训练好的模型权重
├── utils/                 # 工具函数和辅助类
│   ├── configs/           # 配置文件和参数定义
│   │   └── config.py      # 主要配置文件
│   ├── dataset/           # 数据集处理相关
│   │   └── cblprd_dataset.py  # CBLPRD-330k数据集实现
│   ├── model/             # 模型定义
│   │   └── lprnet.py      # LPRNet模型实现与各种变体
│   ├── evaluator.py       # 评估指标和预测解码
│   ├── loss.py            # CTC损失函数实现
│   └── logger.py          # 日志记录和可视化工具
├── train.py               # 训练脚本
├── test.py                # 测试和评估脚本
├── README.md              # 英文说明文档
└── README_CN.md           # 中文说明文档
```

### 核心组件

- **模型架构**：在`utils/model/lprnet.py`中定义，包括多种LPRNet变体
- **数据集处理**：`utils/dataset/cblprd_dataset.py`中的`CBLPRDDataset`类负责加载和预处理
- **评估系统**：`utils/evaluator.py`中的`Evaluator`类处理预测解码和准确率计算
- **训练循环**：在`train.py`中实现，包含早停和学习率调度
- **测试系统**：`test.py`脚本支持批量评估和单张图像测试

## 许可证

该项目基于MIT许可证 - 有关详细信息，请参阅LICENSE文件。

## 参考

- [LPRNet论文](https://arxiv.org/abs/1806.10447)，作者 Sergey Zherzdev和Alexey Gruzdev
  Zherzdev, S., & Gruzdev, A. (2018). Lprnet: License plate recognition via deep neural networks. arXiv preprint arXiv:1806.10447.
- [STNet论文](https://arxiv.org/abs/1506.02025)，作者Max Jaderberg, Karen Simonyan, Andrew Zisserman和Koray Kavukcuoglu
  Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks[C]//Advances in Neural Information Processing Systems, 2015, 28.
- [CBLPRD-330k数据集](https://github.com/SunlifeV/CBLPRD-330k)，作者 SunlifeV
- [YOLOv5-LPRNet](https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition) 作者 HuKai97
- [crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition) 作者 we0091234
- [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) 作者 sirius-ai
- [智能驾驶 车牌检测和识别（三）《CRNN和LPRNet实现车牌识别（含车牌识别数据集和训练代码）》](https://blog.csdn.net/guyuealian/article/details/128704209)，作者AI吃大瓜