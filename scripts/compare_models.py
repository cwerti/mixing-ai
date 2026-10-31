"""
Сравнение чекпоинтов MixingAIModel по голове выбора плагинов и регрессии (протокол docs/research/calibration.md).

Подкоманды:
  val    — логиты и параметры на val-сплите датасетов (без аугментаций) для каждой модели;
  pairs  — логиты на парах в стиле бенчмарка (val-песни, исходник и референс из разных песен, 30 с, референс
           отрендерен известной цепочкой), на главном окне и на 4 фиксированных окнах референса;
  report — метрики (F1/FPR/ECE/MAE по плагинам), Platt по плагину (подгонка на val своего датасета),
           results.json и markdown-таблицы; --save-platt сохраняет Platt рядом с чекпоинтом.

Наборы пар:
  bench_balanced  — цепочка _generate_balanced_random_config (prior датасета v3, ~50 % на эффект);
  bench_realistic — _generate_realistic_config + даблы/гармонии в 35 % референсов (prior датасета v4);
  preset          — как разобранные лид-пресеты: никогда chorus/distortion/exciter/multiband;
  preset_layers   — то же плюс даблы/гармонии в каждом референсе (проверка «дабл ≠ chorus»).

Пример:
  python scripts/compare_models.py val --models v4=data/models/v4/best_model.pth v5=data/models/v5/best_model.pth \
      --datasets v4=data/processed/dataset_v4 v5=data/processed/dataset_v5
  python scripts/compare_models.py pairs --models ... --set preset_layers --n 200 --seed 4
  python scripts/compare_models.py report --platt-fit v4=v3 v5=v4 --save-platt v5
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.dataset.dataset_schema import (  # noqa: E402
    MODEL_PLUGINS, PARAM_LAYOUT, PARAM_OFFSETS, ChainConfig, PluginConfig, remap_param_vector,
)

SR = 44100
SECONDS = 30.0
N_WINDOWS = 4
PLUGINS = [p for p in MODEL_PLUGINS if p != "eq"]  # EQ включен всегда
COLS = [MODEL_PLUGINS.index(p) for p in PLUGINS]
SETS = ("bench_balanced", "bench_realistic", "preset", "preset_layers")
CLIP = 30.0


def parse_pairs(items: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items)


# ---------------------------------------------------------------- модели
def load_model(path: str, device):
    import torch
    from app.ml.model import MixingAIModel

    model = MixingAIModel().to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck.get("param_ranges")


def run_models(models: dict, xs: np.ndarray, xd: np.ndarray, device, batch: int = 64) -> dict[str, tuple]:
    """Логиты и параметры (в текущих диапазонах DATAGEN_PLUGINS) для каждой модели."""
    import torch

    out = {}
    for name, (model, ranges) in models.items():
        lg_all, pr_all = [], []
        with torch.no_grad():
            for i in range(0, len(xs), batch):
                lg, pr = model(torch.from_numpy(xs[i:i + batch]).to(device), None,
                               torch.from_numpy(xd[i:i + batch]).to(device))
                lg_all.append(lg.cpu().numpy())
                pr_all.append(np.stack([remap_param_vector(v, ranges) for v in pr.cpu().numpy()]))
        out[name] = (np.vstack(lg_all), np.vstack(pr_all))
    return out


# ---------------------------------------------------------------- val
def cmd_val(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    from app.ml.dataset import MixingAIDataset

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {n: load_model(p, device) for n, p in parse_pairs(args.models).items()}
    for ds_name, ds_dir in parse_pairs(args.datasets).items():
        ds = MixingAIDataset(ds_dir, split="val", augment=False)
        legacy_labels = not json.loads((Path(ds_dir) / "stats.json").read_text(encoding="utf-8")).get("prior")
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=args.num_workers)
        acc = {n: ([], []) for n in models}
        ys, ps, masks = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for xs, xd, yc, yp, mask in loader:
                res = run_models(models, xs.numpy(), xd.numpy(), device, batch=128)
                for n, (lg, pr) in res.items():
                    acc[n][0].append(lg)
                    acc[n][1].append(pr)
                ys.append(yc.numpy())
                # Метки датасетов до v4 нормализованы в прежних диапазонах: переводим их в текущие
                yp = yp.numpy()
                ps.append(np.stack([remap_param_vector(v, None) for v in yp]) if legacy_labels else yp)
                masks.append(mask.numpy())
        src_files = np.array([s.get("src_file", "") for s in ds.samples])
        for n, (lg, pr) in acc.items():
            np.savez(out_dir / f"val_{ds_name}__{n}.npz", logits=np.vstack(lg), params=np.vstack(pr),
                     y=np.vstack(ys), y_params=np.vstack(ps), mask=np.vstack(masks), src_files=src_files)
        print(f"[+] val {ds_name}: {len(ds)} сэмплов, {time.time() - t0:.0f} с", flush=True)


# ---------------------------------------------------------------- pairs
def preset_chain(rng: random.Random) -> ChainConfig:
    """Цепочка как у разобранных лид-пресетов: EQ, динамика, пространство; без chorus/distortion/exciter/OTT."""
    from app.dataset.dataset_schema import CHAIN_ORDER

    u = rng.uniform
    chain = ChainConfig(plugins=[PluginConfig("eq", {**{f"band{i}_freq_hz": u(0.25, 0.75) for i in range(1, 8)},
                                                    **{f"band{i}_gain_db": u(0.42, 0.58) for i in range(1, 8)}})])
    cand = {
        "pitch_corrector": (0.5, {"speed": u(0.3, 0.9)}),
        "compressor": (0.9, {"threshold_db": u(0.2, 0.7), "ratio": u(0.2, 0.6), "attack_ms": u(0.2, 0.6),
                             "release_ms": u(0.2, 0.6)}),
        "deesser": (0.7, {"threshold_db": u(0.3, 0.7), "ratio": u(0.2, 0.6)}),
        "stereo_enhancer": (0.3, {"delay_ms": u(0.2, 0.6), "width": u(0.2, 0.6)}),
        "reverb": (0.85, {"room_size": u(0.3, 0.7), "damping": u(0.3, 0.7), "wet_level": u(0.1, 0.45),
                          "dry_level": u(0.6, 0.9)}),
        "delay": (0.6, {"delay_seconds": u(0.2, 0.6), "feedback": u(0.2, 0.5), "mix": u(0.1, 0.4)}),
    }
    for name in CHAIN_ORDER:
        if name in cand and rng.random() < cand[name][0]:
            chain.plugins.append(PluginConfig(name, cand[name][1]))
    return chain


def val_lead_groups(dry_dir: str) -> dict[str, list[Path]]:
    """Лид-вокалы val-песен (тот же split, что у dataset_v3/v4), без песен реального теста."""
    from app.dataset.dataset_generator import DatasetGenerator, _is_audio_file, is_dry_source, is_test_song
    from app.eval.benchmark import _song_groups

    files = [f for ext in ("*.wav", "*.flac") for f in Path(dry_dir).rglob(ext) if _is_audio_file(f)]
    _, val = DatasetGenerator._split_files_by_group(files, val_ratio=0.15)
    val = [f for f in val if is_dry_source(f) and not is_test_song(f)]
    lead = [f for f in val if "lead" in f.stem.lower()]
    return _song_groups(lead or val)


def _pair_worker(task):
    """Рендер одной пары и входы модели на главном окне и на N_WINDOWS окнах референса."""
    import warnings

    warnings.filterwarnings("ignore")
    from app.audio.audio_processor import AudioProcessor
    from app.dataset.dataset_generator import (
        _generate_balanced_random_config, _generate_realistic_config, sample_reference_layers,
        with_reference_layers,
    )
    from app.eval.benchmark import load_stereo, make_case
    from app.ml.features import model_inputs, window_seconds

    set_name, i, seed, groups = task
    rng = random.Random(seed * 100003 + i)
    keys = sorted(groups)
    s, r = rng.sample(keys, 2)
    src_file, ref_file = rng.choice(groups[s]), rng.choice(groups[r])
    layers = []
    if set_name == "bench_balanced":
        chain = _generate_balanced_random_config(rng)
    elif set_name == "bench_realistic":
        chain = _generate_realistic_config(rng)
        layers = sample_reference_layers(rng) if rng.random() < 0.35 else []
    else:
        chain = preset_chain(rng)
        layers = sample_reference_layers(rng) if set_name == "preset_layers" else []
    render = with_reference_layers(chain, layers) if layers else chain
    case = make_case(f"{set_name}{i:03d}", load_stereo(src_file, SR, SECONDS), load_stereo(ref_file, SR, SECONDS),
                     render, SR)
    proc = AudioProcessor(sr=SR)
    win = int(window_seconds(SR) * SR)
    n = case.ref.shape[1]
    refs = [case.ref] + [case.ref[:, st:st + win] for st in np.linspace(0, max(0, n - win), N_WINDOWS).astype(int)]
    inputs = [model_inputs(case.src, ref, proc) for ref in refs]
    on = {p.name for p in chain.plugins}
    return (np.stack([x for x, _ in inputs]), np.stack([d for _, d in inputs]),
            [1.0 if c in on else 0.0 for c in MODEL_PLUGINS], f"{src_file.stem}__{ref_file.stem}", len(layers))


def cmd_pairs(args) -> None:
    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {n: load_model(p, device) for n, p in parse_pairs(args.models).items()}
    groups = val_lead_groups(args.dry_dir)
    print(f"[*] {args.set}: {args.n} пар из {len(groups)} val-песен", flush=True)
    tasks = [(args.set, i, args.seed, groups) for i in range(args.n)]
    lg = {n: ([], []) for n in models}
    ys, names, n_layers = [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (xs, xd, y, name, nl) in enumerate(ex.map(_pair_worker, tasks)):
            for m, (logits, _) in run_models(models, xs, xd, device).items():
                lg[m][0].append(logits[0])
                lg[m][1].append(logits[1:])
            ys.append(y)
            names.append(name)
            n_layers.append(nl)
            if k % 50 == 0:
                print(f"    {k}/{args.n} {time.time() - t0:.0f} с", flush=True)
    for m, (main, wins) in lg.items():
        np.savez(out_dir / f"pairs_{args.set}__{m}.npz", logits=np.array(main), logits_win=np.array(wins),
                 y=np.array(ys), names=np.array(names), n_layers=np.array(n_layers))
    print(f"[+] {args.set}: {args.n} пар за {time.time() - t0:.0f} с", flush=True)


# ---------------------------------------------------------------- report
def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -CLIP, CLIP)))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * np.mean(idx == b)
                     for b in range(bins) if np.any(idx == b)))


def cls_metrics(p, y, thr=0.5) -> dict:
    from sklearn.metrics import roc_auc_score

    pred = p > thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    neg = int(np.sum(y == 0))
    nan = float("nan")
    return dict(base=float(y.mean()), n=int(len(y)),
                prec=tp / (tp + fp) if tp + fp else nan, rec=tp / (tp + fn) if tp + fn else nan,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else nan,
                fpr=fp / neg if neg else nan, fpr_keep=float(np.mean(p[y == 0] > 0.9)) if neg else nan,
                auc=float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else nan, ece=ece(p, y))


def fit_platt(z: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float]]:
    from sklearn.linear_model import LogisticRegression

    out = {}
    for i, name in enumerate(PLUGINS):
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(np.clip(z[:, i:i + 1], -CLIP, CLIP), y[:, i])
        out[name] = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    return out


def apply_platt(z: np.ndarray, platt: dict) -> np.ndarray:
    a = np.array([platt[p][0] for p in PLUGINS])
    b = np.array([platt[p][1] for p in PLUGINS])
    return sig(np.clip(z, -CLIP, CLIP) * a + b)


def table(z: np.ndarray, y: np.ndarray, platt: dict | None = None) -> dict:
    p = apply_platt(z, platt) if platt else sig(z)
    return {name: cls_metrics(p[:, i], y[:, i]) for i, name in enumerate(PLUGINS)}


def param_mae(params, y_params, mask) -> dict:
    """MAE параметров по плагину на сэмплах, где плагин действительно включен (как Val MAE в train_ml.py)."""
    out = {}
    for name, layout in PARAM_LAYOUT.items():
        sl = slice(PARAM_OFFSETS[name], PARAM_OFFSETS[name] + len(layout))
        on = mask[:, sl].sum(1) > 0
        if on.any():
            err = np.abs(params[on, sl] - y_params[on, sl])
            out[name] = dict(mae=float(err.mean()), per_param=dict(zip(layout, err.mean(0).round(4).tolist())),
                             n=int(on.sum()))
    allm = mask == 1
    out["_all"] = dict(mae=float(np.abs(params - y_params)[allm].mean()))
    return out


def song_halves(src_files: np.ndarray, seed: int) -> np.ndarray:
    from app.dataset.dataset_generator import DatasetGenerator

    keys = np.array([DatasetGenerator.song_key(Path(f.replace("\\", "/"))) for f in src_files])
    uniq = sorted(set(keys))
    random.Random(seed).shuffle(uniq)
    return np.isin(keys, uniq[: len(uniq) // 2])


def cmd_report(args) -> None:
    out_dir = Path(args.out)
    fit_on = parse_pairs(args.platt_fit)  # модель -> имя датасета, на val которого подгоняется Platt
    results = {"val": {}, "pairs": {}, "platt": {}, "platt_halves": {}}

    def load(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    for f in sorted(out_dir.glob("val_*__*.npz")):
        ds, model = f.stem[4:].split("__")
        d = load(f)
        z, y = d["logits"][:, COLS], d["y"][:, COLS]
        results["val"].setdefault(ds, {})[model] = dict(raw=table(z, y),
                                                       mae=param_mae(d["params"], d["y_params"], d["mask"]))
        if fit_on.get(model) == ds:
            results["platt"][model] = fit_platt(z, y)
            # Проверка переносимости: подгонка на половине val-песен, оценка на другой половине (10 разбиений)
            fpr_b = []
            for seed in range(10):
                a = song_halves(d["src_files"], seed)
                pl = fit_platt(z[a], y[a])
                fpr_b.append({k: v for k, v in table(z[~a], y[~a], pl).items()})
            results["platt_halves"][model] = {
                name: {m: float(np.nanmean([r[name][m] for r in fpr_b])) for m in ("f1", "fpr", "rec", "ece")}
                for name in PLUGINS}

    for model, platt in results["platt"].items():
        for ds, by_model in results["val"].items():
            if model in by_model:
                f = out_dir / f"val_{ds}__{model}.npz"
                d = load(f)
                by_model[model]["platt"] = table(d["logits"][:, COLS], d["y"][:, COLS], platt)

    for f in sorted(out_dir.glob("pairs_*__*.npz")):
        set_name, model = f.stem[6:].split("__")
        d = load(f)
        z, zw, y = d["logits"][:, COLS], d["logits_win"][..., COLS].mean(1), d["y"][:, COLS]
        entry = dict(raw=table(z, y), win4=table(zw, y))
        if model in results["platt"]:
            entry["platt"] = table(z, y, results["platt"][model])
            entry["platt_win4"] = table(zw, y, results["platt"][model])
        results["pairs"].setdefault(set_name, {})[model] = entry

    (out_dir / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(markdown(results))

    for model in args.save_platt or []:
        ck = Path(parse_pairs(args.models)[model])
        path = ck.with_suffix(".platt.json")
        payload = {
            "formula": "p = sigmoid(a * logit + b), по плагину; logit — выход class_head",
            "fit": f"val датасета {fit_on[model]} ({out_dir})",
            "eq": "всегда включен, не калибруется",
            "platt": {k: {"a": round(a, 4), "b": round(b, 4)} for k, (a, b) in results["platt"][model].items()},
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Platt сохранен: {path}")


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def markdown(results: dict) -> str:
    lines = []
    for ds, by_model in results["val"].items():
        models = sorted(by_model)
        lines.append(f"\n### val {ds}: F1 / FPR / AUROC / ECE (сырые, порог 0.5)\n")
        lines.append("| Плагин | База | " + " | ".join(models) + " |")
        lines.append("|---|---|" + "---|" * len(models))
        for p in PLUGINS:
            base = by_model[models[0]]["raw"][p]["base"]
            cells = [" / ".join(fmt(by_model[m]["raw"][p][k]) for k in ("f1", "fpr", "auc", "ece")) for m in models]
            lines.append(f"| {p} | {base:.2f} | " + " | ".join(cells) + " |")
        f1 = [np.nanmean([by_model[m]["raw"][p]["f1"] for p in PLUGINS]) for m in models]
        lines.append("| **macro F1** | | " + " | ".join(f"{x:.3f}" for x in f1) + " |")
        lines.append(f"\n### val {ds}: MAE параметров по плагину (активные, текущие диапазоны)\n")
        lines.append("| Плагин | " + " | ".join(models) + " |")
        lines.append("|---|" + "---|" * len(models))
        for p in list(PARAM_LAYOUT) + ["_all"]:
            lines.append(f"| {p} | " + " | ".join(fmt(by_model[m]["mae"].get(p, {}).get("mae"), 3)
                                                  for m in models) + " |")
    for set_name, by_model in results["pairs"].items():
        models = sorted(by_model)
        lines.append(f"\n### {set_name}: FPR на 0.5 (raw / 4 окна / Platt / Platt+4 окна), F1 raw\n")
        lines.append("| Плагин | База | " + " | ".join(models) + " |")
        lines.append("|---|---|" + "---|" * len(models))
        for p in PLUGINS:
            base = by_model[models[0]]["raw"][p]["base"]
            cells = []
            for m in models:
                e = by_model[m]
                fprs = " / ".join(fmt(e[k][p]["fpr"]) for k in ("raw", "win4", "platt", "platt_win4") if k in e)
                cells.append(f"{fprs}; F1 {fmt(e['raw'][p]['f1'])}")
            lines.append(f"| {p} | {base:.2f} | " + " | ".join(cells) + " |")
    for model, platt in results["platt"].items():
        lines.append(f"\n### Platt {model}\n")
        lines.append("| Плагин | a | b | порог сырой вероятности |")
        lines.append("|---|---|---|---|")
        for p, (a, b) in platt.items():
            thr = sig(np.array([-b / a]))[0] if a > 0 else float("nan")
            lines.append(f"| {p} | {a:.3f} | {b:+.3f} | {fmt(thr)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение чекпоинтов MixingAIModel (выбор плагинов и параметры)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", nargs="+", default=["v4=data/models/v4/best_model.pth",
                                                        "v5=data/models/v5/best_model.pth"])
    common.add_argument("--out", default="data/runs/model_v5")
    p_val = sub.add_parser("val", parents=[common])
    p_val.add_argument("--datasets", nargs="+", default=["v4=data/processed/dataset_v4", "v5=data/processed/dataset_v5"])
    p_val.add_argument("--num-workers", type=int, default=4)
    p_pairs = sub.add_parser("pairs", parents=[common])
    p_pairs.add_argument("--set", choices=SETS, required=True)
    p_pairs.add_argument("--n", type=int, default=200)
    p_pairs.add_argument("--seed", type=int, default=1)
    p_pairs.add_argument("--workers", type=int, default=12)
    p_pairs.add_argument("--dry-dir", default="data/raw/dry_vocals")
    p_rep = sub.add_parser("report", parents=[common])
    p_rep.add_argument("--platt-fit", nargs="*", default=["v4=v3", "v5=v4"],
                       help="модель=датасет: на val какого датасета подгонять Platt этой модели")
    p_rep.add_argument("--save-platt", nargs="*", help="Сохранить Platt этих моделей рядом с чекпоинтом")
    args = parser.parse_args()
    {"val": cmd_val, "pairs": cmd_pairs, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
