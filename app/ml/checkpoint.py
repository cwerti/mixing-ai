"""Загрузка чекпоинта MixingAIModel: строго, с понятной ошибкой вместо молча случайных слоев."""
from pathlib import Path
from typing import Any

import torch

from app.ml.model import MixingAIModel


class CheckpointMismatchError(RuntimeError):
    """Веса чекпоинта не подходят к текущей архитектуре MixingAIModel."""


def _describe(keys: list[str], limit: int = 10) -> str:
    shown = ", ".join(keys[:limit])
    return shown + (f" и еще {len(keys) - limit}" if len(keys) > limit else "")


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[MixingAIModel, dict[str, Any]]:
    """
    Загружает MixingAIModel из чекпоинта (словарь с model_state_dict) в режиме eval.

    Любое расхождение с архитектурой (отсутствующие, лишние или другого размера слои) — CheckpointMismatchError:
    пропуск таких слоев оставил бы случайно инициализированные головы и бессмысленные предсказания.
    """
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CheckpointMismatchError(f"{path}: это не чекпоинт MixingAIModel (нет model_state_dict).")
    state_dict = checkpoint["model_state_dict"]
    model = MixingAIModel().to(device)
    expected = model.state_dict()

    missing = [k for k in expected if k not in state_dict]
    unexpected = [k for k in state_dict if k not in expected]
    mismatched = [f"{k} {tuple(state_dict[k].shape)}->{tuple(v.shape)}" for k, v in expected.items()
                  if k in state_dict and state_dict[k].shape != v.shape]
    if missing or unexpected or mismatched:
        problems = [f"{title}: {_describe(keys)}" for title, keys in
                    (("нет в чекпоинте", missing), ("лишние", unexpected), ("другой размер", mismatched)) if keys]
        raise CheckpointMismatchError(
            f"Чекпоинт {path} не подходит к текущей архитектуре модели — " + "; ".join(problems))

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint
