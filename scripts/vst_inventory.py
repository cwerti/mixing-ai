"""
Опись VST3-плагинов, уже установленных на машине, для доменной рандомизации (app/audio/effect_variants.py).

Ничего не устанавливает и не скачивает. Каждый плагин грузится в отдельном процессе с таймаутом
(падение или зависание плагина не роняет опись): pedalboard.load_plugin, список параметров, два оффлайн-рендера
шума (проверка детерминизма и конечности) и скорость рендера (x реального времени). Категория — по имени
плагина (reverb, delay, compressor, saturation, eq, modulation, pitch, multiband).

Результат: data/raw/vst_inventory.json (его читает effect_variants: в рандомизацию идут только плагины со
статусом ok и известной категорией).

Пример:
    python scripts/vst_inventory.py
    python scripts/vst_inventory.py --extra-dir "D:/Plugins/VST3" --timeout 90
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

OUT = Path("data/raw/vst_inventory.json")
SR = 44100

# Где на Windows лежат VST3 (и куда их ставит FL Studio). VST2 (.dll) pedalboard на Windows не грузит.
DEFAULT_DIRS = [
    "C:/Program Files/Common Files/VST3",
    "C:/Program Files (x86)/Common Files/VST3",
    "C:/Program Files/VSTPlugins",
    "C:/Program Files/Steinberg/VSTPlugins",
    "C:/Program Files/Image-Line",
    os.path.expandvars("%LOCALAPPDATA%/Programs/Common/VST3"),
]

# Категория по имени плагина (регистр не важен). Порядок важен: multiband раньше compressor.
CATEGORY_PATTERNS = [
    ("multiband", r"ott|multiband|multi-band|maximus|xpander"),
    ("reverb", r"reverb|verb|supermassive|room|hall|plate|space|valhalla(?!.*delay)|lux|ambien"),
    ("delay", r"delay|echo|tape ?echo"),
    ("pitch", r"tune|pitch|graillon|melodyne|pitcher"),
    ("modulation", r"chorus|flanger|phaser|ensemble|dimension|vibrato|tremolo"),
    ("saturation", r"satur|distort|drive|tube|tape|clip|crush|decapitator|soundgoodizer|overdrive|fuzz|waveshap"),
    ("compressor", r"compress|limit|leveler|opto|1176|la-?2a|vca|glue|dynamic"),
    ("eq", r"\beq\b|equali[sz]er|pro-?q|nova|parametric|filter"),
]


def categorize(name: str) -> str | None:
    lower = name.lower()
    for category, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, lower):
            return category
    return None


def find_plugins(dirs: list[str]) -> list[Path]:
    found = []
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        # .vst3 — каталог-бандл или одиночный файл; во вложенные бандлы не спускаемся
        for p in root.rglob("*.vst3"):
            if not any(parent.suffix.lower() == ".vst3" for parent in p.parents):
                found.append(p)
    return sorted(set(found))


def probe(path: str) -> dict:
    """Загрузка и рендер одного плагина (выполняется в дочернем процессе)."""
    import numpy as np
    import pedalboard
    started = time.perf_counter()
    plugin = pedalboard.load_plugin(path)
    load_s = time.perf_counter() - started
    info = {"name": getattr(plugin, "name", Path(path).stem), "load_s": load_s,
            "is_effect": bool(getattr(plugin, "is_effect", True)),
            "params": {k: str(v) for k, v in plugin.parameters.items()}}
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal((2, 4 * SR))).astype(np.float32)
    started = time.perf_counter()
    y1 = plugin.process(x, SR, reset=True)
    render_s = time.perf_counter() - started
    y2 = plugin.process(x, SR, reset=True)
    info.update({"render_x_realtime": (x.shape[1] / SR) / max(render_s, 1e-6),
                 "finite": bool(np.isfinite(y1).all()),
                 "deterministic": bool(y1.shape == y2.shape and np.allclose(y1, y2, atol=1e-5)),
                 "out_shape": list(y1.shape),
                 "changes_signal": bool(y1.shape != x.shape or np.abs(y1 - x).mean() > 1e-6)})
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Опись установленных VST3 (pedalboard, в подпроцессах)")
    parser.add_argument("--extra-dir", nargs="*", default=[])
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--probe", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        print("PROBE_JSON " + json.dumps(probe(args.probe)))
        return

    dirs = DEFAULT_DIRS + args.extra_dir
    plugins = find_plugins(dirs)
    print(f"[*] Каталоги: {[d for d in dirs if Path(d).exists()] or 'ни одного нет'}; VST3 найдено: {len(plugins)}")
    rows = []
    for path in plugins:
        row = {"path": str(path), "category": categorize(path.stem)}
        try:
            proc = subprocess.run([sys.executable, __file__, "--probe", str(path)], capture_output=True, text=True,
                                  timeout=args.timeout, errors="replace")
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_JSON ")), None)
            if line is None:
                row.update(status="error", error=(proc.stderr or proc.stdout)[-500:], returncode=proc.returncode)
            else:
                info = json.loads(line[len("PROBE_JSON "):])
                ok = info["finite"] and info["deterministic"] and info["is_effect"] and info["changes_signal"]
                row.update(status="ok" if ok else "unusable", **info)
                row["category"] = row["category"] or categorize(info["name"])
        except subprocess.TimeoutExpired:
            row.update(status="timeout")
        rows.append(row)
        print(f"    {row['status']:8s} {row['category'] or '-':11s} {path.name}"
              + (f" ({row.get('render_x_realtime', 0):.0f}x RT, {len(row.get('params', {}))} параметров)"
                 if row["status"] == "ok" else ""))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"), "dirs": dirs, "plugins": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    usable = sum(r["status"] == "ok" and r["category"] for r in rows)
    print(f"[+] {out}: {len(rows)} плагинов, пригодны для рандомизации: {usable}")


if __name__ == "__main__":
    main()
