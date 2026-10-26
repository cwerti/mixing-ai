import torch
import torch.nn as nn
import torch.nn.functional as F
from app.dataset.dataset_schema import CHAIN_ORDER, PARAMETER_VECTOR_LENGTH, NUM_DSP_DESCRIPTORS


class ResidualBlock(nn.Module):
    """
    Остаточный сверточный блок со Squeeze-and-Excitation (Channel Attention)
    для выделения значимых спектральных и Mid/Side полос.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int] = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Squeeze-and-Excitation (SE) для взвешивания каналов
        se_hidden = max(16, out_channels // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, se_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(se_hidden, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.se(out)
        out = out + self.shortcut(x)
        return self.act(out)


class TemporalStatsPool2d(nn.Module):
    """
    Многомасштабный статистический пулинг по времени (Mean + Std + Max).
    В отличие от простого AdaptiveAvgPool2d((1, 1)), сохраняет:
      1. Средний спектральный баланс и Mid/Side картину (Mean).
      2. Динамический разброс, модуляцию хоруса и повторы дилея во времени (Std).
      3. Транзиентные пики и сибилянты (Max).
    При этом сохраняет 4 укрупненные частотные зоны (Low, Low-Mid, High-Mid, Air).
    """
    def __init__(self, freq_bins: int = 4):
        super().__init__()
        self.freq_bins = freq_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F_in, T) -> приводим частотную ось к freq_bins (4 полосы)
        x_band = F.adaptive_avg_pool2d(x, (self.freq_bins, x.shape[-1]))  # (B, C, 4, T)
        mean_t = x_band.mean(dim=-1)                                      # (B, C, 4)
        std_t = torch.sqrt(x_band.var(dim=-1, unbiased=False) + 1e-6)     # (B, C, 4)
        max_t = x_band.amax(dim=-1)                                       # (B, C, 4)
        # Объединяем и усредняем по 4 частотным зонам с проекцией
        stats = torch.cat([mean_t, std_t, max_t], dim=1)                  # (B, 3*C, 4)
        return stats.flatten(start_dim=1)                                 # (B, 12*C)


class MixingAIModel(nn.Module):
    """
    Двухвходовая гибридная нейросеть Mixing-AI v2 (Dual-Input ResNet-SE + Physical DSP Descriptor MLP).
    Принимает:
      - x_spec: (B, 6, 128, T) [src_mid, src_side, ref_mid, ref_side, delta_mid, delta_side]
        (также обратно совместима с 1-канальным входом (B, 1, 128, T) для старых тестов)
      - x_desc: (B, 54) [desc_src (18), desc_ref (18), delta_desc (18)] (опционально)
    """

    def __init__(
        self,
        num_classes: int = len(CHAIN_ORDER),
        num_params: int = PARAMETER_VECTOR_LENGTH,
        in_channels: int = 6,
        desc_dim: int = NUM_DSP_DESCRIPTORS * 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.desc_dim = desc_dim
        self.num_classes = num_classes
        self.num_params = num_params

        # 1. Спектральный кодировщик (ResNet-SE)
        self.in_conv = nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm2d(48)
        self.act = nn.GELU()

        self.layer1 = ResidualBlock(48, 64, stride=2)     # 128 -> 64
        self.layer2 = ResidualBlock(64, 128, stride=2)    # 64 -> 32
        self.layer3 = ResidualBlock(128, 192, stride=2)   # 32 -> 16
        self.layer4 = ResidualBlock(192, 256, stride=2)   # 16 -> 8

        self.stats_pool = TemporalStatsPool2d(freq_bins=4)  # Выход: 256 * 3 * 4 = 3072
        self.spec_proj = nn.Sequential(
            nn.Linear(256 * 3 * 4, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # 2. Кодировщик явных физических DSP-дескрипторов (54 -> 192)
        self.desc_mlp = nn.Sequential(
            nn.Linear(desc_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.GELU()
        )

        # 3. Общее слияние признаков (384 + 192 = 576 -> 384)
        fusion_dim = 384 + 192
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # 4. Классификационная голова (выбор активных плагинов)
        self.class_head = nn.Sequential(
            nn.Linear(384 + 192, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, num_classes)
        )

        # 5. Регрессионная голова (точная настройка 24 параметров)
        # Skip-connection от x_desc (54) и desc_emb (192) позволяет напрямую читать лаг дилея/Хааса, Crest Factor и центы тюна!
        reg_in_dim = 384 + 192 + desc_dim + num_classes
        self.reg_head = nn.Sequential(
            nn.Linear(reg_in_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_params),
            nn.Sigmoid()
        )

    def _adapt_input_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Обеспечивает обратную совместимость, если подан 1-канальный или 2-канальный тензор."""
        c = x.shape[1]
        if c == self.in_channels:
            return x
        if c == 1 and self.in_channels == 6:
            # Нормализуем, если поданы сырые дБ [-80..0]
            if x.min() < -1.5:
                x = torch.clamp((x + 40.0) / 40.0, -1.0, 1.0)
            zeros = torch.zeros_like(x)
            neg_ones = torch.full_like(x, -1.0)
            return torch.cat([x, neg_ones, x, neg_ones, zeros, zeros], dim=1)
        if c < self.in_channels:
            pad = torch.zeros(x.shape[0], self.in_channels - c, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        return x[:, :self.in_channels]

    def forward(
        self,
        x_spec: torch.Tensor,
        chain_onehot: torch.Tensor = None,
        x_desc: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход нейросети.
        """
        x_spec = self._adapt_input_channels(x_spec)
        b = x_spec.shape[0]

        if x_desc is None:
            x_desc = torch.zeros(b, self.desc_dim, device=x_spec.device, dtype=x_spec.dtype)

        # Сверточная ветка со статистическим пулингом по времени
        out = self.act(self.in_bn(self.in_conv(x_spec)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        spec_stats = self.stats_pool(out)
        spec_emb = self.spec_proj(spec_stats)

        # Дескрипторная ветка
        desc_emb = self.desc_mlp(x_desc)

        # Слияние
        fused = self.fusion(torch.cat([spec_emb, desc_emb], dim=1))

        # Классификация активных плагинов
        logits = self.class_head(torch.cat([fused, desc_emb], dim=1))

        # Во время обучения используем мягкую смесь вероятностей и ground-truth маски,
        # а на инференсе — непрерывные вероятности sigmoid(logits) вместо жесткого порога,
        # чтобы градиенты и уверенность классификатора плавно передавались в регрессор.
        probs = torch.sigmoid(logits)
        if chain_onehot is None:
            cond_chain = probs
        else:
            cond_chain = 0.5 * chain_onehot + 0.5 * probs.detach()

        # Регрессия параметров
        reg_in = torch.cat([fused, desc_emb, x_desc, cond_chain], dim=1)
        params = self.reg_head(reg_in)

        return logits, params


def compute_multitask_loss(
    logits: torch.Tensor,
    pred_params: torch.Tensor,
    true_chain: torch.Tensor,
    true_params: torch.Tensor,
    param_mask: torch.Tensor,
    pos_weight: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 12.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Комбинированная функция потерь:
      - Классификация: взвешенный BCEWithLogitsLoss (с pos_weight для баланса редких и частых плагинов).
      - Регрессия: Masked SmoothL1 + MSE Loss (устойчива к выбросам и дает сильный градиент даже при малых ошибках,
        предотвращая коллапс предсказаний к константе 0.5!).
    """
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_class = bce_loss(logits, true_chain)

    # Комбинация L1 (MAE) и L2 (MSE) на активных параметрах: L1 дает постоянный градиент +-1 вблизи 0.5!
    l1_diff = torch.abs(pred_params - true_params)
    l2_diff = (pred_params - true_params) ** 2
    combined_reg = (0.65 * l1_diff + 0.35 * l2_diff) * param_mask
    loss_reg = combined_reg.sum() / (param_mask.sum() + 1e-8)

    total_loss = alpha * loss_class + beta * loss_reg
    return total_loss, loss_class, loss_reg
