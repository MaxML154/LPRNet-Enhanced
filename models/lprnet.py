import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class SmallBasicBlock(nn.Module):


    def __init__(self, in_channels, out_channels):
        super(SmallBasicBlock, self).__init__()
        self.shortcut = None

        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        
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
        
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        
        out = self.conv4(out)
        out = self.bn4(out)

        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out += identity
        out = self.relu_out(out)
        
        return out


class OptimizedLPRNet(nn.Module):
    
    def __init__(self, class_num, dropout_rate=0.5, use_resnet=True, input_size=(94, 24)):
        super(OptimizedLPRNet, self).__init__()
        self.class_num = class_num
        self.input_width, self.input_height = input_size

        
        if use_resnet:
            resnet = models.resnet50(pretrained=True)
            
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1.weight.data = resnet.conv1.weight.data

            
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  # output: 256 channels

            
            self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 24))

            
            self.layer2 = nn.Sequential(
                SmallBasicBlock(256, 256),
                SmallBasicBlock(256, 512)
            )
            
        else:
            
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=(1, 1), padding=1),
                
                SmallBasicBlock(96, 128),
                nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
                
                SmallBasicBlock(128, 256),
                SmallBasicBlock(256, 256),
                SmallBasicBlock(256, 384),
                nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
                
                SmallBasicBlock(384, 512),
            )

        
        self.use_resnet = use_resnet

        
        self.chinese_branch = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(512, 256, kernel_size=(1, 4), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.latin_branch = nn.Sequential(
            nn.Dropout(dropout_rate * 0.8),  
            nn.Conv2d(512, 256, kernel_size=(1, 4), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        
        self.dropout = nn.Dropout(dropout_rate * 0.5)  
        self.classifier = nn.Conv2d(512, class_num, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(class_num)
        self.relu = nn.ReLU(inplace=True)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, None)) 

        
        self._initialize_weights()
        
    def _initialize_weights(self):
        
        if self.use_resnet:
            return
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        
        batch_size = x.size(0)

        
        if self.use_resnet:
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)

            
            # print(f"After layer1: {x.shape}")

            
            x = self.adaptive_pool(x)
            # print(f"After adaptive_pool: {x.shape}")
            
            features = self.layer2(x)
            # print(f"After layer2: {features.shape}")
        else:
            features = self.backbone(x)
            # print(f"Backbone features: {features.shape}")

        
        chinese_features = self.chinese_branch(features)
        latin_features = self.latin_branch(features)

        
        combined = torch.cat([chinese_features, latin_features], dim=1)
        combined = self.dropout(combined)

        
        logits = self.classifier(combined)
        logits = self.bn(logits)
        logits = self.relu(logits)

        
        logits = self.avgpool(logits)

        
        if logits.dim() == 4:
            _, num_classes, height, width = logits.shape

            
            if height == 1:
                logits = logits.squeeze(2)
            else:
                
                logits = logits.reshape(batch_size, num_classes, width)

        
        if logits.dim() != 3:
            
            if logits.dim() == 4:
                
                num_elements = logits.numel()
                expected_shape = batch_size * self.class_num * (num_elements // (batch_size * self.class_num))
                logits = logits.reshape(batch_size, self.class_num, -1)
            else:
                raise ValueError(f"Expected 3D tensor before permute, got {logits.dim()}D tensor")

        
        logits = logits.permute(0, 2, 1)  # (batch_size, width, class_num)
        
        return logits


class CTCLoss(nn.Module):


    def __init__(self, blank=0, reduction='mean', zero_infinity=True):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        
    def forward(self, logits, targets, logits_lengths, targets_lengths):
        """
        Args:
            logits: (batch_size, time_steps, class_num)
            targets: (batch_size, max_target_len)
            logits_lengths: (batch_size,)
            targets_lengths: (batch_size,)
        """
        
        batch_size = targets.size(0)
        max_target_len = targets.size(1)
        for i in range(batch_size):
            if targets_lengths[i] > max_target_len:
                targets_lengths[i] = max_target_len
        
        log_probs = F.log_softmax(logits, dim=2)
        
        targets_flat = []
        for i in range(batch_size):
            valid_length = min(targets_lengths[i].item(), max_target_len)
            targets_flat.extend(targets[i, :valid_length].tolist())
        
        targets_flat = torch.tensor(targets_flat, dtype=torch.int32, device=targets.device)
        
        expected_length = targets_lengths.sum().item()
        if targets_flat.size(0) != expected_length:
            targets_flat = targets_flat[:expected_length]
        
        targets_flat = targets_flat.int()
        
        try:
            
            loss = self.ctc_loss(log_probs.transpose(0, 1), targets_flat, logits_lengths, targets_lengths)

            
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)

            
            return loss + 0.0001 * l2_reg
            
        except Exception as e:
            print(f"CTC loss error: {e}")
            print(f"Shapes - logits: {logits.shape}, targets: {targets.shape}")
            print(f"Lengths - logits: {logits_lengths}, targets: {targets_lengths}")

            
            return torch.tensor(10.0, device=logits.device, requires_grad=True)


def build_lprnet(config):


    model = OptimizedLPRNet(
        class_num=config['MODEL']['CLASS_NUM'],
        dropout_rate=0.5,
        use_resnet=True,  
        input_size=config['MODEL']['INPUT_SIZE']
    )
    return model


def decode_ctc(logits, chars_list, blank=0, confidence_threshold=0.1):
    
    probs = F.softmax(logits, dim=2)
    
    batch_size = logits.size(0)
    results = []
    
    for i in range(batch_size):
        
        max_probs, pred_indices = torch.max(probs[i], dim=1)

        
        indices = []
        for t, (idx, prob) in enumerate(zip(pred_indices.tolist(), max_probs.tolist())):
            if prob > confidence_threshold and idx != blank:
                indices.append(idx)

        
        merged = []
        prev = -1
        
        for idx in indices:
            
            if idx != prev:
                merged.append(idx)
            prev = idx

        
        chars = []
        for idx in merged:
            if idx > 0 and idx < len(chars_list) + 1:
                try:
                    chars.append(chars_list[idx - 1])
                except IndexError:
                    
                    print(f"Warning: Invalid index {idx-1} for chars_list with length {len(chars_list)}")

        
        plate_text = ''.join(chars)

        
        if not plate_text and confidence_threshold > 0.05:
            
            indices = []
            for t, (idx, prob) in enumerate(zip(pred_indices.tolist(), max_probs.tolist())):
                if prob > confidence_threshold/2 and idx != blank:
                    indices.append(idx)

            
            merged = []
            prev = -1
            for idx in indices:
                if idx != prev:
                    merged.append(idx)
                prev = idx
            
            chars = []
            for idx in merged:
                if idx > 0 and idx < len(chars_list) + 1:
                    try:
                        chars.append(chars_list[idx - 1])
                    except IndexError:
                        pass
            
            plate_text = ''.join(chars)
        
        results.append(plate_text)
    
    return results 
