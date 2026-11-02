"""
Выученный эмбеддинг отпечатка (app/eval/fingerprint_embedding.py): генерация набора эффектов, обучение, сравнение.

    python scripts/train_fingerprint_embedding.py generate --dsp 300 --sm 250 --workers 16
    python scripts/train_fingerprint_embedding.py train

generate: случайные эффекты нашего DSP (ревер, дилей, хорус и их сочетания через DSPEngine.apply_chain) и
Valhalla Supermassive (случайные режим и ручки) + 22 режима Supermassive по умолчанию (тест «sm_default», как в
docs/research/effect_fingerprint.md). Для каждого: отпечаток, карта отклика на импульс, log-mel мокрых рендеров
склейки трех сухих вокалов по 6 с (один рендер). Supermassive — свежий экземпляр на каждый рендер (reset=True его не сбрасывает).

train: цель пары — log-mel расстояние мокрых рендеров одного и того же сухого (симметричное, среднее по сухим и
каналам). Сравниваются на отложенных эффектах: ручное fingerprint_distance, взвешенная L2 по признакам (NNLS) и
сиамский MLP. Сплиты: случайные 30 % эффектов; целиком отложенные режимы Supermassive; 22 режима по умолчанию.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("data/runs/fingerprint/embedding")
SM_PATH = "C:/Program Files/Common Files/VST3/ValhallaSupermassive.vst3"
DRY_FILES = ("data/processed/real_pairs_v10/whiptails_vox/dry.wav",
             "data/processed/archetypes/pair2_mckay/dry.wav",
             "data/processed/archetypes/pair5_wessel/dry.wav")
VOCAL_SECONDS = 6.0
MEL_POOL = 4  # кадров log-mel в одном (hop 512 × 4 ≈ 46 мс): хвосты видны, а матрица пар считается быстро
SM_MODES = ("Gemini", "Hydra", "Centaurus", "Sagittarius", "Great Annihilator", "Andromeda", "Lyra", "Capricorn",
            "Triangulum", "Large Magellanic Cloud", "Cirrus Major", "Cirrus Minor", "Cassiopeia", "Orion",
            "Aquarius", "Pisces", "Scorpio", "Libra", "Leo", "Virgo", "Pleiades", "Sirius")


# ------------------------------------------------------------------------------------------------- эффекты

def random_dsp_spec(rng: random.Random) -> dict:
    """Случайная цепочка из ревера/дилея/хоруса с нормализованными параметрами схемы."""
    kind = rng.choice(["reverb", "reverb", "delay", "chorus", "reverb+delay", "chorus+reverb"])
    plugins = []
    for name in kind.split("+"):
        if name == "reverb":
            p = {"wet_level": rng.uniform(0.1, 0.8), "dry_level": rng.uniform(0.5, 1.0), "damping": rng.random(),
                 "predelay_ms": rng.random() * rng.choice([0.0, 0.3, 1.0]), "low_cut_hz": rng.random() * 0.4,
                 "high_cut_hz": rng.uniform(0.05, 1.0)}
            if rng.random() < 0.75:  # свертка с RT60, иначе Freeverb с room_size
                p["decay_s"] = rng.random() ** 2
            else:
                p["room_size"] = rng.random()
        elif name == "delay":
            p = {"delay_seconds": rng.random(), "feedback": rng.random(), "mix": rng.uniform(0.2, 1.0),
                 "stereo_spread": rng.random() * rng.choice([0.0, 1.0]), "low_cut_hz": rng.random() * 0.5,
                 "high_cut_hz": rng.random()}
        else:
            p = {"rate_hz": rng.random(), "depth": rng.random(), "feedback": rng.random(), "mix": rng.random()}
        plugins.append({"name": name, "params": p})
    return {"kind": "dsp", "plugins": plugins}


def random_sm_spec(rng: random.Random) -> dict:
    params = {
        "mode": rng.choice(SM_MODES), "mix": rng.uniform(20.0, 80.0), "delaysync": "Msec",
        "delay_ms": rng.choice([0.0, rng.uniform(0.0, 120.0), rng.uniform(80.0, 1000.0)]),
        "feedback": rng.uniform(0.0, 95.0), "density": rng.uniform(0.0, 100.0), "width": rng.uniform(0.0, 100.0),
        "lowcut": float(np.exp(rng.uniform(np.log(10.0), np.log(500.0)))),
        "highcut": float(np.exp(rng.uniform(np.log(2000.0), np.log(20000.0)))),
        "modrate": float(np.exp(rng.uniform(np.log(0.05), np.log(5.0)))), "moddepth": rng.uniform(0.0, 100.0),
        "delaywarp": rng.uniform(0.0, 100.0),
    }
    return {"kind": "vst", "path": SM_PATH, "params": params}


def build_effect(spec: dict):
    if spec["kind"] == "dsp":
        from app.audio.dsp_engine import DSPEngine
        from app.dataset.dataset_schema import ChainConfig, PluginConfig
        engine = DSPEngine()
        chain = ChainConfig(plugins=[PluginConfig(name=p["name"], params=p["params"]) for p in spec["plugins"]])
        return lambda x, sr: engine.apply_chain(x, sr, chain, final_limiter=False)

    import pedalboard

    def render(x, sr):
        plugin = pedalboard.load_plugin(spec["path"])  # свежий экземпляр: reset не сбрасывает состояние
        for key, value in spec["params"].items():
            setattr(plugin, key, value)
        return plugin.process(x, sr, reset=True)
    return render


# ------------------------------------------------------------------------------------------------- генерация

_DRYS: list[tuple[np.ndarray, int]] | None = None


def _loudest_window(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    if y.shape[1] <= n:
        return y
    hop = sr // 4
    e = [float(np.sum(y[:, i:i + n] ** 2)) for i in range(0, y.shape[1] - n, hop)]
    start = int(np.argmax(e)) * hop
    return y[:, start:start + n]


def _drys() -> list[tuple[np.ndarray, int]]:
    global _DRYS
    if _DRYS is None:
        import soundfile as sf
        _DRYS = []
        for path in DRY_FILES:
            y, sr = sf.read(path, dtype="float32", always_2d=True)
            y = y.T if y.shape[1] == 2 else np.vstack([y.T, y.T])
            _DRYS.append((_loudest_window(y, sr, VOCAL_SECONDS), sr))
    return _DRYS


def vocal_mels(effect) -> np.ndarray:
    """
    log-mel (1, каналы, 80, кадры/MEL_POOL) мокрого рендера склейки трех сухих в дБ, float16. Один рендер
    вместо трех: загрузка Supermassive (~2.6 с) дороже самого рендера. Хвост одного куска заходит в следующий —
    это часть того, что эффект делает с этим входом, и одинаково для всех эффектов.
    """
    import librosa
    from app.eval.effect_fingerprint import _render
    drys = _drys()
    sr = drys[0][1]
    wet = _render(effect, np.concatenate([y for y, _ in drys], axis=1), sr)
    chans = []
    for ch in wet:
        m = librosa.feature.melspectrogram(y=ch.astype(np.float32), sr=sr, n_fft=2048, hop_length=512, n_mels=80)
        k = m.shape[1] // MEL_POOL
        m = m[:, :k * MEL_POOL].reshape(80, k, MEL_POOL).mean(axis=2)
        chans.append(librosa.power_to_db(m, ref=1.0, amin=1e-10))
    return np.stack(chans)[None].astype(np.float16)


def process_spec(idx: int, spec: dict, group: str) -> str:
    from app.eval.effect_fingerprint import fingerprint
    from app.eval.fingerprint_embedding import impulse_response, response_map
    path = OUT / "effects" / f"{idx:05d}.npz"
    if path.exists():
        return path.name
    effect = build_effect(spec)
    t = time.time()
    fp = fingerprint(effect, sr=48000, ir_seconds=8.0)
    rmap = response_map(impulse_response(effect, sr=48000, seconds=8.0), 48000)
    mels = vocal_mels(effect)
    np.savez_compressed(path, rmap=rmap, mels=mels,
                        meta=json.dumps({"idx": idx, "group": group, "spec": spec, "fingerprint": fp,
                                         "seconds": time.time() - t}, ensure_ascii=False, default=float))
    return path.name


def generate(args) -> None:
    (OUT / "effects").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    jobs = [(i, {"kind": "vst", "path": SM_PATH, "params": {"mode": m}}, "sm_default")
            for i, m in enumerate(SM_MODES)]
    base = len(jobs)
    jobs += [(base + i, random_dsp_spec(rng), "dsp") for i in range(args.dsp)]
    base = len(jobs)
    jobs += [(base + i, random_sm_spec(rng), "sm") for i in range(args.sm)]
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_spec, i, s, g): i for i, s, g in jobs}
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as exc:  # битый эффект не должен ронять весь набор
                print(f"[-] {futures[fut]}: {exc!r}", flush=True)
            if done % 25 == 0 or done == len(jobs):
                print(f"[+] {done}/{len(jobs)} за {time.time() - t0:.0f} с", flush=True)


# ------------------------------------------------------------------------------------------------- обучение

def load_set():
    from app.eval.fingerprint_embedding import embedding_input
    from app.eval.effect_fingerprint import fingerprint_vector
    metas, inputs, feats, mels = [], [], [], []
    for path in sorted((OUT / "effects").glob("*.npz")):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        metas.append(meta)
        inputs.append(embedding_input(meta["fingerprint"], z["rmap"]))
        feats.append(fingerprint_vector(meta["fingerprint"]))
        mels.append(z["mels"].astype(np.float32))
    return metas, np.stack(inputs), np.stack(feats), np.stack(mels)


def pair_targets(mels: np.ndarray) -> np.ndarray:
    """Симметричная log-mel ошибка (дБ) для всех пар, как logmel_db в scripts/real_pairs.py (пол −80 от максимума)."""
    n = len(mels)
    floor = mels.max(axis=(3, 4), keepdims=True) - 80.0  # (n, сухие, каналы, 1, 1)
    d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a = np.maximum(mels[i][None], floor)          # i, обрезанный по полу j
        b = np.maximum(mels, floor)                   # j, обрезанный по своему полу
        fwd = np.abs(a - b).mean(axis=(1, 2, 3, 4))
        a2 = np.maximum(mels[i][None], floor[i][None])
        b2 = np.maximum(mels, floor[i][None])
        bwd = np.abs(b2 - a2).mean(axis=(1, 2, 3, 4))
        d[i] = 0.5 * (fwd + bwd)
    return 0.5 * (d + d.T)


def cached_pair_targets(mels: np.ndarray) -> np.ndarray:
    """pair_targets с кэшем на диске: матрица 472×472 считается ~1 мин, а обучений с ней много."""
    cache = OUT / "pair_targets.npy"
    if cache.exists():
        d = np.load(cache)
        if d.shape == (len(mels), len(mels)):
            return d
    d = pair_targets(mels)
    np.save(cache, d)
    return d


def _pairs(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.triu_indices(len(idx), k=1)
    return idx[i], idx[j]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def nn_rank(pred: np.ndarray, true: np.ndarray, idx: np.ndarray) -> float:
    """Средний ранг по звуку ближайшего по предсказанию соседа (0 — совпал с ближайшим по звуку)."""
    p, t = pred[np.ix_(idx, idx)].copy(), true[np.ix_(idx, idx)].copy()
    np.fill_diagonal(p, np.inf)
    np.fill_diagonal(t, np.inf)
    return float(np.mean([(t[k] < t[k, np.argmin(p[k])]).sum() for k in range(len(idx))]))


def fit_weighted_l2(feats: np.ndarray, d: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Baseline: d² ≈ Σ w_k Δ_k², w ≥ 0 (NNLS) по парам обучающих эффектов."""
    from scipy.optimize import nnls
    i, j = _pairs(train)
    rng = np.random.default_rng(0)
    if len(i) > 40000:
        sel = rng.choice(len(i), 40000, replace=False)
        i, j = i[sel], j[sel]
    w, _ = nnls((feats[i] - feats[j]) ** 2, d[i, j] ** 2)
    return w


def train_embedder(inputs: np.ndarray, d: np.ndarray, train: np.ndarray, seed: int, epochs: int = 300):
    import torch
    from app.eval.fingerprint_embedding import FingerprintEmbedder, embed, input_stats
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = input_stats(inputs[train])
    x = torch.from_numpy(((inputs - mean) / std).astype(np.float32))
    target = torch.from_numpy(d)
    model = FingerprintEmbedder(inputs.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    i_all, j_all = _pairs(train)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(i_all))
        for start in range(0, len(order), 4096):
            sel = order[start:start + 4096]
            i, j = torch.from_numpy(i_all[sel]), torch.from_numpy(j_all[sel])
            ei, ej = model(x[i]), model(x[j])
            pred = torch.sqrt(((ei - ej) ** 2).sum(dim=1) + 1e-6)
            loss = torch.nn.functional.smooth_l1_loss(pred, target[i, j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, mean, std, embed(model, inputs, mean, std)


def train_embedder_rank(inputs: np.ndarray, d: np.ndarray, train: np.ndarray, seed: int, steps: int = 3000,
                        rank_weight: float = 1.0, tau: float = 1.0):
    """
    Регрессия расстояния + listwise-ранжирование (ListNet по строкам матрицы): для каждого якоря распределение
    softmax(−d/τ) по остальным обучающим эффектам должно совпадать с softmax(−d̂/τ). При τ ≈ 1 дБ вес почти весь
    у ближайших соседей — именно их порядок нужен адаптеру («какой режим ближе всех»), а регрессия держит
    глобальный масштаб. Полный батч: обучающих эффектов ~300, матрица 300×300 дешевле выборки пар.
    """
    import torch
    from app.eval.fingerprint_embedding import FingerprintEmbedder, embed, input_stats
    torch.manual_seed(seed)
    mean, std = input_stats(inputs[train])
    x_all = torch.from_numpy(((inputs - mean) / std).astype(np.float32))
    x = x_all[torch.from_numpy(train)]
    t = torch.from_numpy(d[np.ix_(train, train)])
    n = len(train)
    eye = torch.eye(n, dtype=torch.bool)
    iu = torch.triu_indices(n, n, offset=1)
    target = torch.softmax((-t / tau).masked_fill(eye, -1e9), dim=1)
    model = FingerprintEmbedder(inputs.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        model.train()
        e = model(x)
        dist = torch.sqrt(((e[:, None, :] - e[None, :, :]) ** 2).sum(dim=2) + 1e-6)
        reg = torch.nn.functional.smooth_l1_loss(dist[iu[0], iu[1]], t[iu[0], iu[1]])
        logp = torch.log_softmax((-dist / tau).masked_fill(eye, -1e9), dim=1)
        rank = -(target * logp).sum(dim=1).mean() + (target * torch.log(target + 1e-12)).sum(dim=1).mean()
        loss = reg + rank_weight * rank
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model, mean, std, embed(model, inputs, mean, std)


def train(args) -> None:
    from app.eval.effect_fingerprint import fingerprint_distance
    from app.eval.fingerprint_embedding import save_embedder
    metas, inputs, feats, mels = load_set()
    groups = np.array([m["group"] for m in metas])
    modes = np.array([m["spec"]["params"].get("mode", "") if m["spec"]["kind"] == "vst" else "" for m in metas])
    n = len(metas)
    print(f"эффектов: {n} (dsp {np.sum(groups == 'dsp')}, sm {np.sum(groups == 'sm')}, "
          f"sm_default {np.sum(groups == 'sm_default')})")
    d = cached_pair_targets(mels)
    # fingerprint_distance = RMS разности fingerprint_vector — то же самое векторно
    hand = np.sqrt(((feats[:, None, :] - feats[None, :, :]) ** 2).mean(axis=2)).astype(np.float32)
    assert np.isclose(hand[0, 1], fingerprint_distance(metas[0]["fingerprint"], metas[1]["fingerprint"]), atol=1e-4)

    rng = np.random.default_rng(args.seed)
    not_default = np.nonzero(groups != "sm_default")[0]
    held_modes = set(rng.choice(SM_MODES, 6, replace=False).tolist())
    splits = {}
    perm = rng.permutation(not_default)
    k = int(0.3 * len(perm))
    splits["random30"] = (np.sort(perm[k:]), np.sort(perm[:k]))
    in_held = np.isin(modes, list(held_modes))
    splits["held_modes"] = (np.nonzero(~in_held & (groups != "sm_default"))[0],
                            np.nonzero(in_held & (groups == "sm"))[0])
    splits["sm_default"] = (not_default, np.nonzero(groups == "sm_default")[0])

    report = {"n": n, "held_modes": sorted(held_modes), "splits": {}}
    for name, (tr, te) in splits.items():
        if len(te) < 3:
            print(f"\n[{name}] пропущен: в тесте {len(te)} эффектов")
            continue
        pi, pj = _pairs(te)
        true = d[pi, pj]
        row = {"train": int(len(tr)), "test": int(len(te)), "pairs": int(len(pi))}
        row["hand"] = {"spearman": spearman(hand[pi, pj], true), "nn_rank": nn_rank(hand, d, te)}
        w = fit_weighted_l2(feats, d, tr)
        wl2 = np.sqrt(((feats[:, None, :] - feats[None, :, :]) ** 2 * w).sum(axis=2))
        row["weighted_l2"] = {"spearman": spearman(wl2[pi, pj], true), "nn_rank": nn_rank(wl2, d, te)}
        sp, rk = [], []
        for seed in range(args.seeds):
            if args.loss == "rank":
                model, mean, std, e = train_embedder_rank(inputs, d, tr, seed, steps=args.steps,
                                                          rank_weight=args.rank_weight, tau=args.tau)
            else:
                model, mean, std, e = train_embedder(inputs, d, tr, seed, epochs=args.epochs)
            emb = np.sqrt(((e[:, None, :] - e[None, :, :]) ** 2).sum(axis=2))
            sp.append(spearman(emb[pi, pj], true))
            rk.append(nn_rank(emb, d, te))
            if name == "sm_default" and seed == 0:
                save_embedder(OUT / args.model_name, model, mean, std,
                              {"train": "all except sm_default", "loss": args.loss, "epochs": args.epochs,
                               "steps": args.steps, "rank_weight": args.rank_weight, "tau": args.tau})
        row["embedding"] = {"spearman": float(np.mean(sp)), "spearman_std": float(np.std(sp)),
                            "nn_rank": float(np.mean(rk)), "per_seed": sp}
        report["splits"][name] = row
        print(f"\n[{name}] обучение {len(tr)}, тест {len(te)} эффектов, {len(pi)} пар")
        for key in ("hand", "weighted_l2", "embedding"):
            extra = f" ± {row[key]['spearman_std']:.3f}" if key == "embedding" else ""
            print(f"  {key:<12} Spearman {row[key]['spearman']:.3f}{extra}   ранг соседа {row[key]['nn_rank']:.2f}")
    report["loss"] = {"loss": args.loss, "rank_weight": args.rank_weight, "tau": args.tau, "steps": args.steps}
    (OUT / args.report_name).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[+] {OUT / args.report_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--dsp", type=int, default=300)
    g.add_argument("--sm", type=int, default=250)
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--seeds", type=int, default=3)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--loss", choices=("regress", "rank"), default="regress",
                   help="regress — только регрессия расстояния по парам; rank — + listwise по ближайшим")
    t.add_argument("--steps", type=int, default=3000, help="шаги полного батча для --loss rank")
    t.add_argument("--rank-weight", type=float, default=1.0)
    t.add_argument("--tau", type=float, default=1.0, help="температура listwise, дБ log-mel")
    t.add_argument("--model-name", default="embedder.pt")
    t.add_argument("--report-name", default="report.json")
    args = parser.parse_args()
    generate(args) if args.cmd == "generate" else train(args)


if __name__ == "__main__":
    main()
