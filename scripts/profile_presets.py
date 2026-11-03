"""
Отпечатки заводских пресетов Supermassive: один раз на плагин, дальше выбор идет по кэшу.

    python scripts/profile_presets.py --workers 18

Каждый .vpreset из папки пресетов рендерится свежим экземпляром на склеенном тестовом сигнале (fast_fingerprint).
Результат — data/runs/fingerprint/presets/presets.jsonl: имя, категория (папка), нормализованные значения,
значения в единицах плагина, отпечаток; карты отклика — presets_rmap.npz (вход эмбеддинга).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.plugin_adapter import fast_fingerprint, load_vpreset, sm_render  # noqa: E402

PRESETS = Path("C:/ProgramData/Valhalla DSP, LLC/ValhallaSupermassive/Presets/Factory")
OUT = Path("data/runs/fingerprint/presets")


def plugin_values(raw: dict) -> dict:
    """Значения в единицах плагина (для отчета и экспорта в DAW)."""
    import pedalboard
    from app.eval.plugin_adapter import SM_PATH
    plugin = pedalboard.load_plugin(SM_PATH)
    for key, value in raw.items():
        plugin.parameters[key].raw_value = value
    # pedalboard отдает подклассы float/str (обертки), которые не пиклятся между процессами — приводим к базовым
    return {k: (bool(v) if isinstance(v, bool) else float(v) if isinstance(v, (int, float)) else str(v))
            for k, v in ((k, getattr(plugin, k)) for k in raw)}


def job(path: str) -> dict:
    preset = load_vpreset(path)
    fp, rmap = fast_fingerprint(lambda x, sr: sm_render(preset, x, sr))
    category = Path(path).relative_to(PRESETS).parts[0]
    return {"name": preset["preset"], "category": category, "path": path, "raw": preset["raw"],
            "values": plugin_values(preset["raw"]), "fingerprint": fp, "rmap": rmap}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(str(p) for p in PRESETS.rglob("*.vpreset"))
    if args.limit:
        paths = paths[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(job, p) for p in paths]
        for k, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            print(f"[{k}/{len(paths)}] {rows[-1]['category']}/{rows[-1]['name']} ({time.time() - t0:.0f} с)", flush=True)
    rows.sort(key=lambda r: r["path"])
    np.savez_compressed(OUT / "presets_rmap.npz", rmap=np.stack([r.pop("rmap") for r in rows]))
    (OUT / "presets.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=float) + "\n" for r in rows),
                                       encoding="utf-8")
    print(f"[+] {OUT / 'presets.jsonl'}: {len(rows)} пресетов")


if __name__ == "__main__":
    main()
