"""Регрессии ревью P1: проверка входного аудио, строгая загрузка чекпоинта, отсутствие val-выборки."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT))

from app.audio.audio_processor import AudioProcessor
from app.audio.input_check import AudioInputError, check_audio, load_checked_audio
from app.ml.checkpoint import CheckpointMismatchError, load_model
from app.ml.dataset import MixingAIDataset, has_split
from app.ml.features import model_inputs, prepare_model_audio
from app.ml.model import MixingAIModel
from app.ml.pipeline import MixingAIMLPipeline
from synthetic_audio import synthetic_vocal

SR = 22050


def _write(path: Path, y: np.ndarray, sr: int = SR) -> Path:
    sf.write(path, y, sr)
    return path


# --- 1. Входное аудио -------------------------------------------------------------------------------------------

def test_check_audio_rejects_short_silent_and_non_finite():
    good = synthetic_vocal(SR, seconds=2.0, seed=1)
    check_audio(good, SR)

    with pytest.raises(AudioInputError, match="слишком короткая"):
        check_audio(good[: SR // 5], SR)
    with pytest.raises(AudioInputError, match="беззвучная"):
        check_audio(np.zeros(2 * SR, dtype=np.float32), SR)
    broken = good.copy()
    broken[100] = np.nan
    with pytest.raises(AudioInputError, match="NaN"):
        check_audio(broken, SR)


def test_load_checked_audio_reports_unreadable_and_missing(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF\x00\x00\x00\x00WAVEnot really audio")
    with pytest.raises(AudioInputError, match="не удалось прочитать"):
        load_checked_audio(corrupt, SR, "Исходный вокал")
    with pytest.raises(AudioInputError, match="не найден"):
        load_checked_audio(tmp_path / "missing.wav", SR)

    y = load_checked_audio(_write(tmp_path / "ok.wav", synthetic_vocal(SR, seconds=2.0, seed=2)), SR)
    assert y.shape[-1] == 2 * SR


def test_model_inputs_survive_very_short_input():
    """0.2 с раньше падало в pyloudnorm: «Audio must have length greater than the block size»."""
    processor = AudioProcessor(sr=SR)
    short = synthetic_vocal(SR, seconds=0.2, seed=3)
    assert prepare_model_audio(short, SR).shape[0] == 2
    x_spec, x_desc = model_inputs(short, synthetic_vocal(SR, seconds=3.0, seed=4), processor)
    assert np.all(np.isfinite(x_spec)) and np.all(np.isfinite(x_desc))


def test_pipeline_run_rejects_short_source_before_inference(tmp_path):
    pipeline = MixingAIMLPipeline.__new__(MixingAIMLPipeline)  # без загрузки весов: до модели дело не доходит
    pipeline.processor = AudioProcessor(sr=SR)
    src = _write(tmp_path / "src.wav", synthetic_vocal(SR, seconds=0.3, seed=5))
    ref = _write(tmp_path / "ref.wav", synthetic_vocal(SR, seconds=3.0, seed=6))
    with pytest.raises(AudioInputError, match="Исходный вокал"):
        pipeline.run(src, ref)


@pytest.mark.parametrize("bad", ["source", "ref"])
def test_process_vocal_cli_exits_1_with_message(tmp_path, bad):
    ok = synthetic_vocal(SR, seconds=3.0, seed=7)
    silent = np.zeros(3 * SR, dtype=np.float32)
    src = _write(tmp_path / "src.wav", silent if bad == "source" else ok)
    ref = _write(tmp_path / "ref.wav", silent if bad == "ref" else ok)
    model = tmp_path / "model.pth"
    model.write_bytes(b"")  # до весов дело не доходит
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "process_vocal.py"), "--source", str(src), "--ref", str(ref),
         "--model-path", str(model), "--output-fst", str(tmp_path / "out.fst"),
         "--output-wav", str(tmp_path / "out.wav")],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    label = "Исходный вокал" if bad == "source" else "Референс"
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert f"Ошибка: {label}" in proc.stdout and "беззвучная" in proc.stdout
    assert "Traceback" not in proc.stderr


# --- 2. Чекпоинт -------------------------------------------------------------------------------------------------

def test_load_model_rejects_mismatched_checkpoint(tmp_path):
    state = MixingAIModel().state_dict()
    name = next(k for k, v in state.items() if v.ndim == 2)
    state[name] = torch.zeros(state[name].shape[0] + 1, state[name].shape[1])
    dropped = next(k for k in state if k != name)
    del state[dropped]
    state["extra.weight"] = torch.zeros(1)
    path = tmp_path / "bad.pth"
    torch.save({"model_state_dict": state}, path)

    with pytest.raises(CheckpointMismatchError) as err:
        load_model(path)
    message = str(err.value)
    assert name in message and dropped in message and "extra.weight" in message


def test_load_model_roundtrip(tmp_path):
    path = tmp_path / "good.pth"
    torch.save({"model_state_dict": MixingAIModel().state_dict(), "epoch": 3}, path)
    model, checkpoint = load_model(path)
    assert not model.training and checkpoint["epoch"] == 3


@pytest.mark.parametrize("version", ["v4", "v5", "v6"])
def test_shipped_checkpoints_load_strictly(version):
    path = ROOT / "data" / "models" / version / "best_model.pth"
    if not path.exists():
        pytest.skip(f"нет {path}")
    load_model(path)


# --- 3. Датасет без val ------------------------------------------------------------------------------------------

def test_dataset_without_val_split_raises(tmp_path):
    rows = [{"id": f"s{i}", "split": "train"} for i in range(3)]
    (tmp_path / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert has_split(tmp_path, "train") and not has_split(tmp_path, "val")
    assert len(MixingAIDataset(tmp_path, split="train")) == 3
    assert len(MixingAIDataset(tmp_path, split="all")) == 3
    with pytest.raises(ValueError, match='split="val"'):
        MixingAIDataset(tmp_path, split="val")
