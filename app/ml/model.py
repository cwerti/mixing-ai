import torch
import torch.nn as nn
from app.dataset.dataset_schema import CHAIN_ORDER, PARAMETER_VECTOR_LENGTH

class ResidualBlock(nn.Module):
    """
    Остаточный блок (Residual Block) для стабильного градиентного спуска
    в глубоких сверточных сетях (по аналогии с ResNet).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Соединение в обход (Shortcut / Skip Connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MixingAIModel(nn.Module):
    """
    Улучшенная многозадачная сверточная нейросеть (ResNet CNN) для Mixing-AI.
    Обеспечивает точный анализ Мел-спектрограмм благодаря остаточным блокам.
    """

    def __init__(self, num_classes: int = len(CHAIN_ORDER), num_params: int = PARAMETER_VECTOR_LENGTH):
        """
        Инициализирует слои нейросети.
        """
        super().__init__()
        
        # Общий кодировщик признаков (ResNet Backbone)
        self.in_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = ResidualBlock(64, 64, stride=2)   # 128 -> 64 частотных бинов
        self.layer2 = ResidualBlock(64, 128, stride=2)   # 64 -> 32
        self.layer3 = ResidualBlock(128, 256, stride=2)  # 32 -> 16
        self.layer4 = ResidualBlock(256, 512, stride=2) # 16 -> 8
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Схлопывание в вектор (batch, 512, 1, 1)
        
        # Классификационная ветка (Выбор плагинов)
        self.class_fc1 = nn.Linear(512, 128)
        self.class_fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        # Регрессионная ветка (Определение параметров ручек)
        self.reg_fc1 = nn.Linear(512 + num_classes, 256)
        self.reg_fc2 = nn.Linear(256, 128)
        self.reg_fc3 = nn.Linear(128, num_params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, chain_onehot: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход нейросети.
        """
        # Сверточные остаточные слои
        out = self.relu(self.in_bn(self.in_conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Глобальный пулинг
        out = self.pool(out).squeeze(-1).squeeze(-1) # Выход: (batch, 512)
        
        # Классификация
        class_out = self.dropout(self.relu(self.class_fc1(out)))
        logits = self.class_fc2(class_out)
        
        # Инференс: автоматический выбор маски
        if chain_onehot is None:
            probs = torch.sigmoid(logits)
            chain_onehot = (probs > 0.5).float()
            
        # Регрессия параметров
        reg_in = torch.cat([out, chain_onehot], dim=1) # Форма: (batch, 260)
        reg_out = self.relu(self.reg_fc1(reg_in))
        reg_out = self.relu(self.reg_fc2(reg_out))
        params = self.sigmoid(self.reg_fc3(reg_out))
        
        return logits, params


def compute_multitask_loss(logits: torch.Tensor, pred_params: torch.Tensor, true_chain: torch.Tensor, true_params: torch.Tensor, param_mask: torch.Tensor, alpha: float = 1.0, beta: float = 15.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Вычисляет комбинированную функцию потерь (Multi-Task Loss).
    """
    bce_loss = nn.BCEWithLogitsLoss()
    loss_class = bce_loss(logits, true_chain)
    
    diff = (pred_params - true_params) ** 2
    masked_diff = diff * param_mask
    loss_reg = masked_diff.sum() / (param_mask.sum() + 1e-8)
    
    total_loss = alpha * loss_class + beta * loss_reg
    return total_loss, loss_class, loss_reg
