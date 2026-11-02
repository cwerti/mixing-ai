"""
Отпечатки эффектов (app/eval/effect_fingerprint.py): наш DSP и VST3 через pedalboard — в одну таблицу и jsonl.

Примеры:
    python scripts/effect_fingerprint.py --builtin
    python scripts/effect_fingerprint.py --vst "C:/Program Files/Common Files/VST3/ValhallaSupermassive.vst3" \
        --sweep mode --set mix=0.5
    python scripts/effect_fingerprint.py --compare data/runs/fingerprint/*.jsonl

Для каждого эффекта печатается ближайший сосед по расстоянию отпечатков: если разные режимы одного ревера
оказываются ближе друг к другу, чем к ревера с другим характером, язык отпечатков их не различает.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.dsp_engine import DSPEngine  # noqa: E402
from app.dataset.dataset_schema import ChainConfig, PluginConfig  # noqa: E402
from app.eval.effect_fingerprint import fingerprint, fingerprint_distance  # noqa: E402

OUT_DIR = Path("data/runs/fingerprint")
TABLE_KEYS = ("drr_db", "predelay_ms", "peak_ms", "rt60_s", "rt60_500", "rt60_4000", "tail_db_8000", "ned_mid",
              "echo_period_l_ms", "echo_period_r_ms", "echo_strength", "side_mid_db", "iacc_late",
              "ir_repeatability", "mod_spread_db", "pitch_up_db", "pitch_down_db", "compression_db",
              "saturation_db")

# Эффекты нашей палитры в том виде, в каком их рендерит оптимизатор (нормализованные параметры схемы)
BUILTIN: dict[str, tuple[str, dict]] = {
    "dsp_reverb_freeverb": ("reverb", {"room_size": 0.7, "damping": 0.5, "wet_level": 0.3, "dry_level": 0.8}),
    "dsp_reverb_2s_dark": ("reverb", {"decay_s": 0.086, "damping": 1.0, "wet_level": 0.3, "dry_level": 0.8,
                                      "high_cut_hz": 0.1}),
    "dsp_reverb_2s_bright": ("reverb", {"decay_s": 0.086, "damping": 0.0, "wet_level": 0.3, "dry_level": 0.8,
                                        "high_cut_hz": 1.0}),
    "dsp_reverb_6s_predelay": ("reverb", {"decay_s": 0.29, "damping": 0.5, "wet_level": 0.3, "dry_level": 0.8,
                                          "predelay_ms": 0.5}),
    "dsp_delay_320_spread": ("delay", {"delay_seconds": 0.571, "feedback": 0.5, "mix": 0.8, "stereo_spread": 0.2,
                                       "high_cut_hz": 0.15, "low_cut_hz": 0.2}),
    "dsp_delay_slap_120": ("delay", {"delay_seconds": 0.095, "feedback": 0.0, "mix": 0.8, "stereo_spread": 0.0,
                                     "high_cut_hz": 1.0, "low_cut_hz": 0.0}),
    "dsp_chorus": ("chorus", {"rate_hz": 0.3, "depth": 0.6, "feedback": 0.2, "mix": 0.6}),
    "dsp_distortion": ("distortion", {"drive_db": 0.6}),
    "dsp_compressor": ("compressor", {"threshold_db": 0.3, "ratio": 0.6, "attack_ms": 0.3, "release_ms": 0.3}),
}


def dsp_effect(plugin: str, params: dict):
    engine = DSPEngine()
    chain = ChainConfig(plugins=[PluginConfig(name=plugin, params=params)])
    return lambda x, sr: engine.apply_chain(x, sr, chain, final_limiter=False)


def _parse_value(text: str):
    try:
        return float(text)
    except ValueError:
        return {"true": True, "false": False}.get(text.lower(), text)


def vst_effects(path: str, settings: list[str], sweep: str | None, sweep_steps: int):
    """(имя, эффект, параметры) для VST3: фиксированные --set и перебор одного параметра --sweep."""
    import pedalboard
    plugin = pedalboard.load_plugin(path)
    base = {k: _parse_value(v) for k, v in (s.split("=", 1) for s in settings)}
    for key, value in base.items():
        setattr(plugin, key, value)
    values: list = [None]
    if sweep:
        param = plugin.parameters[sweep]
        valid = getattr(param, "valid_values", None)
        if valid and len(valid) <= 64:
            values = list(dict.fromkeys(valid))  # pedalboard повторяет первое значение в конце списка
        else:
            lo, hi = param.range[0], param.range[1]
            values = list(np.linspace(lo, hi, sweep_steps))
    stem = Path(path).stem
    for value in values:
        if value is not None:
            setattr(plugin, sweep, value)
        name = stem if value is None else f"{stem}:{sweep}={value}"
        params = {k: (v if isinstance(v, (bool, int, float, str)) else str(v))
                  for k, v in ((k, getattr(plugin, k)) for k in plugin.parameters)}

        # Свежий экземпляр на каждый рендер: reset=True не сбрасывает внутреннее состояние части плагинов
        # (Supermassive: фаза LFO и буферы — два рендера подряд расходятся на ~2 дБ log-mel), а свежая
        # загрузка дает результат бит в бит
        def render(x, sr, value=value):
            fresh = pedalboard.load_plugin(path)
            for key, val in base.items():
                setattr(fresh, key, val)
            if value is not None:
                setattr(fresh, sweep, value)
            return fresh.process(x, sr, reset=True)
        yield name, render, params


def print_table(rows: list[dict]) -> None:
    width = 13
    names = [r["name"] for r in rows]
    print("признак".ljust(18) + "".join(n[-width + 1:].rjust(width) for n in names))
    for key in TABLE_KEYS:
        cells = []
        for r in rows:
            v = r["fingerprint"].get(key)
            cells.append(f"{v:{width}.2f}" if isinstance(v, (int, float)) and v is not None else " " * (width - 1) + "-")
        print(key.ljust(18) + "".join(cells))


def print_neighbors(rows: list[dict]) -> None:
    if len(rows) < 2:
        return
    print("\nближайший сосед (RMS по масштабированным признакам; < 1 — одна «заметная» разница на признак):")
    for r in rows:
        dists = [(fingerprint_distance(r["fingerprint"], o["fingerprint"]), o["name"]) for o in rows if o is not r]
        d, other = min(dists)
        print(f"  {r['name']:<48} -> {other:<48} {d:5.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--builtin", action="store_true", help="эффекты нашей палитры (DSPEngine)")
    parser.add_argument("--vst", action="append", default=[], help="путь к .vst3 (можно несколько)")
    parser.add_argument("--set", action="append", default=[], help="параметр VST: имя=значение")
    parser.add_argument("--sweep", help="перебрать значения параметра VST (дискретный — все, иначе --sweep-steps)")
    parser.add_argument("--sweep-steps", type=int, default=5)
    parser.add_argument("--compare", nargs="*", default=[], help="jsonl прошлых прогонов — добавить к таблице")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--ir-seconds", type=float, default=6.0)
    parser.add_argument("--tag", default=None, help="имя jsonl в data/runs/fingerprint/")
    parser.add_argument("--repeat", type=int, default=1, help="снять каждый отпечаток N раз (проверка шума)")
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.compare:
        rows += [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    jobs = []
    if args.builtin:
        jobs += [(name, dsp_effect(plugin, params), {"plugin": plugin, **params})
                 for name, (plugin, params) in BUILTIN.items()]
    for path in args.vst:
        jobs += list(vst_effects(path, args.set, args.sweep, args.sweep_steps))

    new_rows = []
    for name, effect, params in jobs:
        for k in range(args.repeat):
            fp = fingerprint(effect, sr=args.sr, ir_seconds=args.ir_seconds)
            row_name = name if args.repeat == 1 else f"{name}#{k}"
            new_rows.append({"name": row_name, "params": params, "fingerprint": fp})
            print(f"[+] {row_name}", flush=True)
    if new_rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = args.tag or ("builtin" if args.builtin and not args.vst else Path(args.vst[0]).stem if args.vst else "run")
        out = OUT_DIR / f"{tag}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows), encoding="utf-8")
        print(f"[+] {out}")
    rows += new_rows
    print_table(rows)
    print_neighbors(rows)


if __name__ == "__main__":
    main()
