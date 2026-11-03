"""
Адаптер «целевой эффект → режим и ручки конкретного плагина» (здесь — Valhalla Supermassive).

Цель задается отпечатком (fingerprint + карта отклика) и сравнивается в пространстве выученного эмбеддинга
(app/eval/fingerprint_embedding.py) или ручным fingerprint_distance. Поиск: суррогат (MLP «ручки → эмбеддинг»,
обучен на уже снятых эффектах плагина) ранжирует режимы и дает старт, затем CMA-ES по непрерывным ручкам на
честных рендерах лучших режимов.

Быстрый отпечаток: полный fingerprint() делает 5–6 рендеров, и на каждый у Supermassive нужна свежая загрузка
(reset=True его не сбрасывает), а сам рендер при большой плотности идет примерно в реальном времени. Здесь все
тестовые сигналы склеены в один вход (~22 с вместо ~40 с) и рендерятся одним свежим экземпляром за один вызов.
Хвост предыдущего участка заходит в следующий; насколько это сдвигает отпечаток — scripts/adapter_search.py
--check-fast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.eval.effect_fingerprint import SHIFT_S, SINE_HZ, _as_2ch, impulse_features, sine_features
from app.eval.fingerprint_embedding import response_map

SM_PATH = "C:/Program Files/Common Files/VST3/ValhallaSupermassive.vst3"
SM_MODES = ("Gemini", "Hydra", "Centaurus", "Sagittarius", "Great Annihilator", "Andromeda", "Lyra", "Capricorn",
            "Triangulum", "Large Magellanic Cloud", "Cirrus Major", "Cirrus Minor", "Cassiopeia", "Orion",
            "Aquarius", "Pisces", "Scorpio", "Libra", "Leo", "Virgo", "Pleiades", "Sirius")


@dataclass(frozen=True)
class Knob:
    name: str
    lo: float
    hi: float
    log: bool = False

    def to_unit(self, v: float) -> float:
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        if self.log:
            return float(math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo))))
        return float(self.lo + u * (self.hi - self.lo))


# Непрерывные ручки Supermassive в тех же границах, что и случайные настройки набора эмбеддинга
SM_KNOBS = (
    Knob("mix", 20.0, 80.0), Knob("delay_ms", 0.0, 1000.0), Knob("feedback", 0.0, 95.0),
    Knob("density", 0.0, 100.0), Knob("width", 0.0, 100.0), Knob("lowcut", 10.0, 500.0, log=True),
    Knob("highcut", 2000.0, 20000.0, log=True), Knob("modrate", 0.05, 5.0, log=True),
    Knob("moddepth", 0.0, 100.0), Knob("delaywarp", 0.0, 100.0),
)
# Значения по умолчанию плагина (22 режима «по умолчанию» в наборе сняты именно так)
SM_DEFAULTS = {"mix": 50.0, "delay_ms": 300.0, "feedback": 50.0, "density": 0.0, "width": 100.0, "lowcut": 10.0,
               "highcut": 20000.0, "modrate": 0.5, "moddepth": 50.0, "delaywarp": 0.0}


def knobs_to_unit(params: dict) -> np.ndarray:
    return np.array([np.clip(k.to_unit(float(params.get(k.name, SM_DEFAULTS[k.name]))), 0.0, 1.0)
                     for k in SM_KNOBS])


def unit_to_params(mode: str, u: np.ndarray) -> dict:
    out = {"mode": mode, "delaysync": "Msec"}
    out.update({k.name: k.from_unit(v) for k, v in zip(SM_KNOBS, u)})
    return out


PRESET_META_KEYS = ("raw", "preset")


def load_vpreset(path) -> dict:
    """
    Пресет Valhalla (.vpreset, XML с нормализованными 0..1 значениями) → {"preset": имя, "raw": {параметр: 0..1}}.
    Нормализованные значения выставляются как есть (raw_value): так точно воспроизводятся и дискретные ручки
    (режим, синхронизация, нота), без перевода в единицы плагина.
    """
    import xml.etree.ElementTree as ET
    attrs = ET.parse(path).getroot().attrib
    raw = {k.lower(): float(v) for k, v in attrs.items() if k not in ("pluginVersion", "presetName")
           and not k.lower().startswith("reserved")}
    return {"preset": attrs.get("presetName", str(path)), "raw": raw}


def sm_render(params: dict, x: np.ndarray, sr: int) -> np.ndarray:
    """
    Рендер свежим экземпляром Supermassive (детерминированно, в отличие от повторного process). params — значения
    в единицах плагина; ключ "raw" — нормализованные значения пресета (выставляются первыми, остальные ключи поверх).
    """
    import pedalboard
    plugin = pedalboard.load_plugin(SM_PATH)
    for key, value in params.get("raw", {}).items():
        plugin.parameters[key].raw_value = value
    for key, value in params.items():
        if key not in PRESET_META_KEYS:
            setattr(plugin, key, value)
    return plugin.process(x.astype(np.float32), sr, reset=True)


# ------------------------------------------------------------------------------------------------ быстрый отпечаток

IR_SECONDS = 8.0
IR2_SECONDS = 1.1   # от сдвинутого импульса нужна только первая секунда (повторяемость)
GAP_SECONDS = 1.0   # тишина перед синусами: хвост импульсов уходит под синус
SINE_SECONDS = 4.0   # как в fingerprint(): при 2.5 с повторы дилея от нарастания синуса попадают в окно анализа
SINE_LEVELS = (-36.0, -18.0, -6.0)
IMPULSE_AMP = 0.5


def probe_signal(sr: int, sine_seconds: float = SINE_SECONDS) -> tuple[np.ndarray, dict]:
    """
    Склеенный тестовый сигнал и позиции участков (в отсчетах): импульс, через IR_SECONDS + SHIFT_S — второй
    импульс (его отклик берется только за 1 с, хвост первого к этому моменту на −20 дБ и ниже), пауза, синусы
    по возрастанию уровня (хвост тихого синуса под громким на 18 дБ ниже). Supermassive при большой плотности
    рендерит примерно в реальном времени, поэтому длина сигнала — главная цена точки поиска.
    """
    n_ir, n_sine = int(IR_SECONDS * sr), int(sine_seconds * sr)
    shift = int(SHIFT_S * sr)
    pos = {"ir": 0, "ir2": n_ir + shift}
    start = pos["ir2"] + int((IR2_SECONDS + GAP_SECONDS) * sr)
    t = np.arange(n_sine) / sr
    fade = 0.5 - 0.5 * np.cos(np.pi * np.minimum(1.0, t / 1.0))
    x = np.zeros((2, start + n_sine * len(SINE_LEVELS)), dtype=np.float32)
    x[:, pos["ir"]] = IMPULSE_AMP
    x[:, pos["ir2"]] = IMPULSE_AMP
    for k, level in enumerate(SINE_LEVELS):
        amp = 10.0 ** (level / 20.0) * np.sqrt(2.0)
        a = start + k * n_sine
        x[:, a:a + n_sine] = amp * np.sin(2 * np.pi * SINE_HZ * t) * fade
        pos[f"sine{level:g}"] = a
    pos["n_ir"], pos["n_ir2"], pos["n_sine"] = n_ir, int(IR2_SECONDS * sr), n_sine
    return x, pos


def fast_fingerprint(render, sr: int = 48000, sine_seconds: float = SINE_SECONDS) -> tuple[dict, np.ndarray]:
    """(отпечаток, карта отклика) по одному рендеру склеенного сигнала. render(x, sr) -> y."""
    x, pos = probe_signal(sr, sine_seconds)
    y = _as_2ch(render(x, sr))
    if y.shape[1] < x.shape[1]:
        y = np.pad(y, ((0, 0), (0, x.shape[1] - y.shape[1])))
    n_ir, n_sine = pos["n_ir"], pos["n_sine"]
    h = y[:, pos["ir"]:pos["ir"] + n_ir] / IMPULSE_AMP
    h2 = y[:, pos["ir2"]:pos["ir2"] + pos["n_ir2"]] / IMPULSE_AMP
    f = impulse_features(h, sr)
    f["ir_seconds"] = IR_SECONDS
    win = slice(0, int(1.0 * sr))
    a, b = h[:, win].ravel(), h2[:, win].ravel()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    f["ir_repeatability"] = float(np.dot(a, b) / denom) if denom > 0 else 1.0
    outputs = {lv: y[:, pos[f"sine{lv:g}"]:pos[f"sine{lv:g}"] + n_sine] for lv in SINE_LEVELS}
    f.update(sine_features(outputs, sr, {lv: lv for lv in SINE_LEVELS}))
    return f, response_map(h, sr)


# ------------------------------------------------------------------------------------------------ суррогат

def surrogate_input(mode: str, u: np.ndarray) -> np.ndarray:
    onehot = np.zeros(len(SM_MODES), dtype=np.float32)
    onehot[SM_MODES.index(mode)] = 1.0
    return np.concatenate([onehot, np.asarray(u, dtype=np.float32)])


def train_surrogate(x: np.ndarray, y: np.ndarray, seed: int = 0, steps: int = 3000):
    """MLP «режим + ручки (0..1) → эмбеддинг». Данных мало (~170 точек), поэтому сеть маленькая с weight decay."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                          nn.Linear(128, y.shape[1]))
    xt, yt = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(steps):
        loss = ((model(xt) - yt) ** 2).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    def predict(inp: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(np.atleast_2d(inp).astype(np.float32))).numpy()
    return predict
