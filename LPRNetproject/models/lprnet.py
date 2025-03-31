import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallBasicBlock(nn.Module):
    """
    Small Basic Block from LPRNet
    """
    def __init__(self, in_channels, out_channels):
        super(SmallBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(out_channels//4)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        return x


class LPRNet(nn.Module):
    """
    LPRNet model for license plate recognition
    """
    def __init__(self, class_num, dropout_rate=0.5, input_size=(94, 24)):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.dropout_rate = dropout_rate
        
        # Input size is expected to be (batch_size, 3, height, width)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 1), padding=1),
            
            SmallBasicBlock(64, 128),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
            
            SmallBasicBlock(128, 256),
            SmallBasicBlock(256, 256),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
            
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, class_num, kernel_size=1, stride=1),
            nn.BatchNorm2d(class_num),
            nn.ReLU(inplace=True)
        )
        
        # Calculate output dimensions
        # Assuming input_size is (width, height)
        self.input_width, self.input_height = input_size
        self.output_height = self.input_height // 4  # Due to two max pooling with stride 2 in height
        self.output_width = self.input_width // 1   # No reduction in width due to stride 1
        
        # Global average pooling to get final features
        self.avgpool = nn.AvgPool2d((self.output_height, 1))
        
    def forward(self, x):
        # x shape: (batch_size, 3, height, width)
        features = self.backbone(x)
        # features shape: (batch_size, class_num, reduced_height, width)
        
        # Apply global average pooling (height-wise)
        features = self.avgpool(features)
        # features shape: (batch_size, class_num, 1, width)
        
        # 检查并确保正确移除高度维度
        if features.dim() == 4:
            # 获取实际形状
            batch_size, num_classes, height, width = features.shape
            
            # 首先尝试squeeze，如果高度维度确实为1
            if height == 1:
                features = features.squeeze(2)
            else:
                # 如果高度不为1，使用平均池化确保高度为1，再squeeze
                adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, width))
                features = adaptive_pool(features).squeeze(2)
        
        # features shape: (batch_size, class_num, width)
        
        # 再次验证维度，确保是3维
        if features.dim() != 3:
            raise ValueError(f"Expected 3D tensor before permute, got {features.dim()}D tensor with shape {features.shape}")
        
        # Permute to get the right shape for CTC loss
        logits = features.permute(0, 2, 1)
        # logits shape: (batch_size, width, class_num)
        
        return logits


class CTCLoss(nn.Module):
    """
    CTC Loss for LPRNet
    """
    def __init__(self, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
        
    def forward(self, logits, targets, logits_lengths, targets_lengths):
        """
        Args:
            logits: (batch_size, time_steps, class_num)
            targets: (batch_size, max_target_len)
            logits_lengths: (batch_size,)
            targets_lengths: (batch_size,)
        """
        # Permute and make logits compatible with CTCLoss
        log_probs = F.log_softmax(logits, dim=2)
        
        # Create flatten targets
        batch_size = targets.size(0)
        # 将targets展平成1D张量，只保留有效部分
        targets_flat = []
        for i in range(batch_size):
            valid_length = targets_lengths[i].item()
            targets_flat.extend(targets[i, :valid_length].tolist())
        
        # 转换为张量
        targets_flat = torch.tensor(targets_flat, dtype=torch.int32, device=targets.device)
        
        # 打印调试信息
        # print(f"logits shape: {logits.shape}")
        # print(f"log_probs shape: {log_probs.shape}")
        # print(f"targets shape: {targets.shape}")
        # print(f"targets_flat shape: {targets_flat.shape}")
        # print(f"logits_lengths: {logits_lengths}")
        # print(f"targets_lengths: {targets_lengths}")
        # print(f"targets_lengths sum: {targets_lengths.sum().item()}")
        
        # 验证targets_flat长度
        expected_flat_length = targets_lengths.sum().item()
        if targets_flat.size(0) != expected_flat_length:
            print(f"WARNING: targets_flat size {targets_flat.size(0)} != sum of targets_lengths {expected_flat_length}")
            # 确保长度正确
            targets_flat = targets_flat[:expected_flat_length]
            
        # 确保targets是int类型
        targets_flat = targets_flat.int()
            
        # Forward pass through CTC loss
        try:
            loss = self.ctc_loss(log_probs.transpose(0, 1), targets_flat, logits_lengths, targets_lengths)
        except Exception as e:
            print(f"CTC loss error: {e}")
            print(f"log_probs shape: {log_probs.shape}, transpose: {log_probs.transpose(0, 1).shape}")
            print(f"targets_flat shape: {targets_flat.shape}, dtype: {targets_flat.dtype}")
            print(f"logits_lengths: {logits_lengths}")
            print(f"targets_lengths: {targets_lengths}, sum: {targets_lengths.sum().item()}")
            # 尝试使用更安全的方式
            padded_targets = torch.zeros(batch_size, targets.size(1), device=targets.device, dtype=torch.int)
            for i in range(batch_size):
                valid_length = min(targets_lengths[i].item(), targets.size(1))
                padded_targets[i, :valid_length] = targets[i, :valid_length]
            
            # 再次尝试计算损失
            loss = self.ctc_loss(log_probs.transpose(0, 1), targets.view(-1), logits_lengths, targets_lengths)
        
        return loss


def build_lprnet(config):
    """
    Build LPRNet model from config
    """
    model = LPRNet(
        class_num=config['MODEL']['CLASS_NUM'],
        dropout_rate=0.5,
        input_size=config['MODEL']['INPUT_SIZE']
    )
    return model


def decode_ctc(logits, chars_list, blank=0):
    """
    Decode CTC output to license plate text
    """
    # Apply softmax and get predicted indices
    probs = F.softmax(logits, dim=2)
    pred_indices = torch.argmax(probs, dim=2)
    
    batch_size = logits.size(0)
    results = []
    
    for i in range(batch_size):
        indices = pred_indices[i].tolist()
        # Merge repeated characters
        merged = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != blank:
                merged.append(idx)
            prev = idx
        
        # Convert indices to characters
        chars = []
        for idx in merged:
            if idx > 0 and idx <= len(chars_list):
                chars.append(chars_list[idx - 1])
        
        results.append(''.join(chars))
    
    return results 