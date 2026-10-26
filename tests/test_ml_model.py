import pytest
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.dataset.dataset_schema import CHAIN_ORDER, PARAMETER_VECTOR_LENGTH, NUM_DSP_DESCRIPTORS
from app.ml.model import MixingAIModel, compute_multitask_loss


def test_model_forward_pass():
    model = MixingAIModel()
    num_classes = len(CHAIN_ORDER)

    # 1. Тест нового 6-канального спектрального входа и 54-мерного дескрипторного входа
    x_spec = torch.randn(2, 6, 128, 520)
    x_desc = torch.randn(2, NUM_DSP_DESCRIPTORS * 3)

    logits, params = model(x_spec, chain_onehot=None, x_desc=x_desc)
    assert logits.shape == (2, num_classes)
    assert params.shape == (2, PARAMETER_VECTOR_LENGTH)
    assert torch.all(params >= 0.0) and torch.all(params <= 1.0)

    # 2. Тест обучения с передачей маски цепочки и обратной совместимости с 1-канальным входом
    x_legacy = torch.randn(2, 1, 128, 520)
    true_chain = torch.zeros(2, num_classes, dtype=torch.float32)
    true_chain[0, 0:2] = 1.0
    true_chain[1, [0, 2, 3]] = 1.0

    logits2, params2 = model(x_legacy, true_chain)
    assert logits2.shape == (2, num_classes)
    assert params2.shape == (2, PARAMETER_VECTOR_LENGTH)


def test_multitask_loss_computation():
    num_classes = len(CHAIN_ORDER)

    logits = torch.randn(2, num_classes, requires_grad=True)
    pred_params = torch.rand(2, PARAMETER_VECTOR_LENGTH, requires_grad=True)

    true_chain = torch.zeros(2, num_classes, dtype=torch.float32)
    true_chain[0, 0:2] = 1.0
    true_chain[1, [0, 2]] = 1.0

    true_params = torch.rand(2, PARAMETER_VECTOR_LENGTH)
    param_mask = torch.zeros(2, PARAMETER_VECTOR_LENGTH)
    param_mask[0, 0:4] = 1.0
    param_mask[1, 11:15] = 1.0

    total_loss, loss_class, loss_reg = compute_multitask_loss(
        logits, pred_params, true_chain, true_params, param_mask
    )

    assert total_loss.ndim == 0
    assert total_loss.item() > 0.0

    total_loss.backward()
    assert logits.grad is not None
    assert pred_params.grad is not None
