"""
Спектрограммы набора для прослушивания: каждый wav папки — строка, столбцы Mid и Side, общая шкала дБ.

    python scripts/spectrogram_compare.py data/processed/listen_blind_sm/whiptails

Пишет spectrograms.png в ту же папку и печатает по файлу: уровень хвоста в паузах (дБ относительно слов),
доля энергии выше 8 кГц, Side/Mid. Паузы и окна слов берутся по огибающей референса (файл с «reference» в имени,
иначе первый), чтобы сравнение шло по одним и тем же кадрам.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

HOP = 512
N_MELS = 128


def mel_db(y: np.ndarray, sr: int) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=HOP, n_mels=N_MELS, fmax=sr / 2)
    return librosa.power_to_db(m, ref=1.0, amin=1e-10)


def compare(folder: Path) -> Path:
    """Спектрограммы и сводка по всем wav папки; возвращает путь к png."""
    files = sorted(folder.glob("*.wav"))
    audio = {}
    for f in files:
        y, sr = sf.read(f, dtype="float32", always_2d=True)
        y = y.T if y.shape[1] > 1 else np.vstack([y[:, 0], y[:, 0]])
        audio[f.stem] = (0.5 * (y[0] + y[1]), 0.5 * (y[0] - y[1]))
    ref_name = next((n for n in audio if "reference" in n), next(iter(audio)))
    ref_mid = audio[ref_name][0]
    rms = librosa.feature.rms(y=ref_mid, frame_length=2048, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    words = rms_db > rms_db.max() - 20
    gaps = rms_db < rms_db.max() - 35

    fig, axes = plt.subplots(len(audio), 2, figsize=(16, 2.6 * len(audio)), squeeze=False)
    print(f"{'файл':<36}{'хвост в паузах':>16}{'>8 кГц':>9}{'Side/Mid':>10}")
    for row, (name, (mid, side)) in enumerate(audio.items()):
        mm, ms = mel_db(mid, sr), mel_db(side, sr)
        top = mm.max()
        for col, (m, label) in enumerate(((mm, "Mid"), (ms, "Side"))):
            ax = axes[row][col]
            librosa.display.specshow(m, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel", fmax=sr / 2, ax=ax,
                                     vmin=top - 80, vmax=top, cmap="magma")
            ax.set_title(f"{name} — {label}", fontsize=9)
        n = min(len(words), mm.shape[1])
        e = librosa.db_to_power(mm[:, :n]).sum(axis=0)
        tail = 10 * np.log10(e[gaps[:n]].mean() / e[words[:n]].mean()) if gaps[:n].any() else float("nan")
        freqs = librosa.mel_frequencies(n_mels=N_MELS, fmax=sr / 2)
        p = librosa.db_to_power(mm)
        hi = p[freqs > 8000].sum() / p.sum()
        sm = 10 * np.log10((side ** 2).sum() / max((mid ** 2).sum(), 1e-12) + 1e-12)
        print(f"{name:<36}{tail:>13.1f} дБ{100 * hi:>8.1f}%{sm:>8.1f} дБ")
    fig.tight_layout()
    out = folder / "spectrograms.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(f"[+] {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    compare(parser.parse_args().folder)


if __name__ == "__main__":
    main()
