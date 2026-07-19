import pytest
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.dataset.dataset_schema import CHAIN_ORDER, PARAMETER_VECTOR_LENGTH
from app.ml.model import MixingAIModel, compute_multitask_loss

def test_model_forward_pass():
    model = MixingAIModel()
    num_classes = len(CHAIN_ORDER)
    
    # Генерируем тестовый батч из 2 сэмплов (3 канала, 128 мел-фильтров, 700 фреймов)
    x = torch.randn(2, 3, 128, 700)
    
    # 1. Тест инференса (без передачи маски)
    logits, params = model(x, chain_onehot=None)
    
    assert logits.shape == (2, num_classes)
    assert params.shape == (2, PARAMETER_VECTOR_LENGTH)
    # Значения параметров должны лежать в [0..1] из-за Sigmoid на конце
    assert torch.all(params >= 0.0) and torch.all(params <= 1.0)
    
    # 2. Тест обучения (с передачей маски цепочки)
    true_chain = torch.zeros(2, num_classes, dtype=torch.float32)
    true_chain[0, 0:2] = 1.0  # EQ, Compressor
    true_chain[1, [0, 2, 3]] = 1.0  # EQ, Deesser, Distortion
    
    logits, params = model(x, true_chain)
    
    assert logits.shape == (2, num_classes)
    assert params.shape == (2, PARAMETER_VECTOR_LENGTH)

def test_multitask_loss_computation():
    num_classes = len(CHAIN_ORDER)
    
    # Создаем фиктивные предсказания и таргеты для батча из 2 сэмплов
    logits = torch.randn(2, num_classes, requires_grad=True)
    pred_params = torch.rand(2, PARAMETER_VECTOR_LENGTH, requires_grad=True)
    
    true_chain = torch.zeros(2, num_classes, dtype=torch.float32)
    true_chain[0, 0:2] = 1.0
    true_chain[1, [0, 2]] = 1.0
    
    true_params = torch.rand(2, PARAMETER_VECTOR_LENGTH)
    
    # Маска активных параметров (1 для Comp в сэмпле 0, 1 для Reverb в сэмпле 1)
    param_mask = torch.zeros(2, PARAMETER_VECTOR_LENGTH)
    # Comp (0-3) активно для сэмпла 0
    param_mask[0, 0:4] = 1.0
    # Reverb (11-14) активно для сэмпла 1
    param_mask[1, 11:15] = 1.0
    
    # Расчет потерь
    total_loss, loss_class, loss_reg = compute_multitask_loss(
        logits, pred_params, true_chain, true_params, param_mask
    )
    
    assert total_loss.ndim == 0  # Скаляр
    assert total_loss.item() > 0.0
    
    # Проверка возможности прохода градиентов (backpropagation)
    total_loss.backward()
    assert logits.grad is not None
    assert pred_params.grad is not None
