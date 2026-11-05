import torch

from app.dataset.dataset_schema import CHAIN_ORDER, PARAMETER_VECTOR_LENGTH
from app.ml.tone_training import EQ_PARAMS, TONE_NAMES, render, spectral_loss, tone_mask

SR = 44100


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(SR) / SR
    audio = (torch.randn(2, 1, SR, generator=g) * (0.05 + 0.4 * ((t > 0.3) & (t < 0.6)))).repeat(1, 2, 1)
    eq = torch.full((2, len(EQ_PARAMS)), 0.5)
    on = torch.zeros(2, len(CHAIN_ORDER))
    on[:, [CHAIN_ORDER.index(n) for n in TONE_NAMES]] = 1.0
    return audio, eq, on


def test_spectral_loss_is_zero_for_true_chain_and_grows_with_wrong_params():
    audio, eq, on = _batch()
    true = torch.full((2, PARAMETER_VECTOR_LENGTH), 0.3)
    target = render(audio, eq, true, on, SR)
    assert spectral_loss(render(audio, eq, true, on, SR), target).max() < 1e-5
    wrong = true.clone()
    wrong[:, tone_mask() == 1] = 0.9
    assert spectral_loss(render(audio, eq, wrong, on, SR), target).min() > 0.01


def test_sound_gradient_reaches_tone_params_and_gates():
    audio, eq, on = _batch(1)
    target = render(audio, eq, torch.full((2, PARAMETER_VECTOR_LENGTH), 0.3), on, SR)
    pred = torch.full((2, PARAMETER_VECTOR_LENGTH), 0.6, requires_grad=True)
    gates = torch.full((2, len(CHAIN_ORDER)), 0.7, requires_grad=True)
    spectral_loss(render(audio, eq, pred, gates, SR), target).mean().backward()
    tone = tone_mask() == 1
    assert pred.grad[:, tone].abs().sum() > 0
    assert pred.grad[:, ~tone].abs().sum() == 0  # ручки вне тона учатся только регрессией
    assert all(gates.grad[:, CHAIN_ORDER.index(n)].abs().sum() > 0 for n in TONE_NAMES)
