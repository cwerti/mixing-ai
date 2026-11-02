"""
Проверка адаптера (app/eval/plugin_adapter.py): по целевому эффекту найти режим и ручки Valhalla Supermassive.

    python scripts/adapter_search.py --check-fast            # быстрый отпечаток против полного
    python scripts/adapter_search.py --targets-sm 10 --targets-dsp 5 --workers 9

Цели берутся из набора эмбеддинга (data/runs/fingerprint/embedding/effects): случайные настройки Supermassive
(«тот же плагин») и эффекты нашего DSP с ревером/дилеем («референс сделан другим плагином»). Эмбеддинг и
суррогат обучаются на всех эффектах, кроме целей.

Поиск для каждой цели и каждой цели поиска (эмбеддинг / ручное fingerprint_distance):
1. суррогат «режим + ручки → вектор» оценивает все 22 режима (CMA-ES по суррогату), берутся TOP лучших;
2. на каждом из них CMA-ES по честным рендерам (быстрый отпечаток, свежий экземпляр на рендер), старт —
   оптимум суррогата.

Оценка — log-mel ошибка мокрого рендера найденной настройки против целевого на тех же сухих вокалах (как цель
обучения эмбеддинга). Сравнение: случайные настройки Supermassive, ближайший по эмбеддингу эффект Supermassive
из набора (без рендера) и лучший эффект Supermassive из набора по самому log-mel (потолок «выбрать из готовых»).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_fingerprint_embedding as tfe  # noqa: E402
from app.eval.effect_fingerprint import fingerprint_vector  # noqa: E402
from app.eval.plugin_adapter import (SM_MODES, fast_fingerprint, knobs_to_unit, sm_render,  # noqa: E402
                                     surrogate_input, train_surrogate, unit_to_params)

OUT = tfe.OUT / "adapter"


# ------------------------------------------------------------------------------------------------ рабочие процессы

def _fast_job(params: dict) -> tuple[dict, np.ndarray]:
    return fast_fingerprint(lambda x, sr: sm_render(params, x, sr))


def _mels_job(params: dict) -> np.ndarray:
    return tfe.vocal_mels(lambda x, sr: sm_render(params, x, sr))


def mel_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(tfe.pair_targets(np.stack([a, b]))[0, 1])


# ------------------------------------------------------------------------------------------------ векторы цели

class Space:
    """Пространство сравнения: эмбеддинг (выученный) или вектор ручных признаков (RMS = fingerprint_distance)."""

    def __init__(self, kind: str, model=None, mean=None, std=None):
        self.kind, self.model, self.mean, self.std = kind, model, mean, std

    def vec(self, fp: dict, rmap: np.ndarray) -> np.ndarray:
        if self.kind == "hand":
            return fingerprint_vector(fp)
        from app.eval.fingerprint_embedding import embed, embedding_input
        return embed(self.model, embedding_input(fp, rmap)[None], self.mean, self.std)[0]

    def dist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = np.atleast_2d(a) - np.atleast_2d(b)
        return np.sqrt((diff ** 2).mean(axis=1)) if self.kind == "hand" else np.sqrt((diff ** 2).sum(axis=1))


def surrogate_modes(predict, target: np.ndarray, space: Space, seed: int, evals: int = 400) -> list:
    """Для каждого режима — минимум расстояния по суррогату (CMA-ES): [(расстояние, режим, ручки 0..1)]."""
    import cma
    out = []
    for mode in SM_MODES:
        es = cma.CMAEvolutionStrategy(np.full(10, 0.5), 0.3, {"bounds": [0, 1], "seed": seed + 1, "verbose": -9,
                                                              "maxfevals": evals, "popsize": 16})
        while not es.stop():
            xs = es.ask()
            pred = predict(np.stack([surrogate_input(mode, x) for x in xs]))
            es.tell(xs, space.dist(pred, target).tolist())
        out.append((float(es.result.fbest), mode, np.clip(es.result.xbest, 0, 1)))
    return sorted(out, key=lambda r: r[0])


def real_search(pool, target: np.ndarray, space: Space, starts: list, generations: int, popsize: int,
                seed: int) -> dict:
    """CMA-ES по честным рендерам на каждом стартовом режиме; рендеры поколения идут параллельно."""
    import cma
    best = {"dist": np.inf}
    history = []
    for _, mode, x0 in starts:
        es = cma.CMAEvolutionStrategy(x0, 0.15, {"bounds": [0, 1], "seed": seed + 7, "verbose": -9,
                                                 "popsize": popsize})
        for _ in range(generations):
            xs = es.ask()
            params = [unit_to_params(mode, x) for x in xs]
            fps = list(pool.map(_fast_job, params))
            ds = [float(space.dist(space.vec(fp, rmap), target)[0]) for fp, rmap in fps]
            es.tell(xs, ds)
            history += [{"mode": mode, "dist": d} for d in ds]
            k = int(np.argmin(ds))
            if ds[k] < best["dist"]:
                best = {"dist": ds[k], "mode": mode, "params": params[k]}
    best["evals"] = len(history)
    best["history"] = history
    return best


# ------------------------------------------------------------------------------------------------ проверка быстрого отпечатка

def check_fast(args) -> None:
    from app.eval.fingerprint_embedding import embed, embedding_input, load_embedder
    metas, inputs, feats, mels = tfe.load_set()
    sm = [i for i, m in enumerate(metas) if m["group"] in ("sm", "sm_default")]
    rng = np.random.default_rng(0)
    pick = rng.choice(sm, args.check_n, replace=False)
    params = [metas[i]["spec"]["params"] for i in pick]
    cache = OUT / f"check_fast_points_{args.check_n}.npz"
    if cache.exists():  # рендеры дорогие, а модель эмбеддинга для сравнения меняется
        z = np.load(cache)
        res = list(zip(json.loads(str(z["fps"])), z["rmaps"]))
        wall = float(z["wall"])
    else:
        t = time.time()
        with ProcessPoolExecutor(args.workers) as pool:
            res = list(pool.map(_fast_job, params))
        wall = time.time() - t
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(cache, fps=json.dumps([f for f, _ in res], default=float), rmaps=np.stack([r for _, r in res]),
                 wall=wall)
    model, mean, std = load_embedder(Path(args.embedder))
    e_full = embed(model, inputs, mean, std)
    e_fast = embed(model, np.stack([embedding_input(f, r) for f, r in res]), mean, std)
    self_d = np.linalg.norm(e_fast - e_full[pick], axis=1)
    cross = np.linalg.norm(e_full[pick][:, None] - e_full[sm][None], axis=2)
    cross[cross < 1e-9] = np.inf
    hand_self = np.sqrt(((np.stack([fingerprint_vector(f) for f, _ in res]) - feats[pick]) ** 2).mean(axis=1))
    report = {"n": int(args.check_n), "workers": args.workers, "seconds_per_point": wall / args.check_n,
              "emb_fast_vs_full_median": float(np.median(self_d)), "emb_fast_vs_full_max": float(self_d.max()),
              "emb_nearest_other_median": float(np.median(cross.min(axis=1))),
              "hand_fast_vs_full_median": float(np.median(hand_self)), "hand_fast_vs_full_max": float(hand_self.max())}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check_fast.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ основной прогон

def choose_targets(metas: list, n_sm: int, n_dsp: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    sm = [i for i, m in enumerate(metas) if m["group"] == "sm"]
    dsp = [i for i, m in enumerate(metas) if m["group"] == "dsp"
           and any(p["name"] in ("reverb", "delay") for p in m["spec"]["plugins"])]
    return rng.sample(sm, n_sm) + rng.sample(dsp, n_dsp)


def run(args) -> None:
    metas, inputs, feats, mels = tfe.load_set()
    d = tfe.cached_pair_targets(mels)
    n = len(metas)
    targets = choose_targets(metas, args.targets_sm, args.targets_dsp, args.seed)
    train_idx = np.array([i for i in range(n) if i not in set(targets)])
    print(f"целей {len(targets)}, обучение эмбеддинга и суррогата на {len(train_idx)} эффектах", flush=True)

    model, mean, std, emb = tfe.train_embedder_rank(inputs, d, train_idx, seed=args.seed, steps=args.steps,
                                                    rank_weight=args.rank_weight, tau=args.tau)
    from app.eval.fingerprint_embedding import save_embedder
    OUT.mkdir(parents=True, exist_ok=True)
    save_embedder(OUT / "embedder.pt", model, mean, std, {"targets": targets, "loss": "rank"})
    spaces = {"emb": (Space("emb", model, mean, std), emb), "hand": (Space("hand"), feats)}

    sm_pool = [i for i in train_idx if metas[i]["group"] in ("sm", "sm_default")]
    sur_x = np.stack([surrogate_input(metas[i]["spec"]["params"].get("mode", "Gemini"),
                                      knobs_to_unit(metas[i]["spec"]["params"])) for i in sm_pool])
    predictors = {}
    for key, (space, vecs) in spaces.items():
        predictors[key] = train_surrogate(sur_x, vecs[sm_pool], seed=args.seed)
        pred = predictors[key](sur_x)
        print(f"суррогат {key}: средняя ошибка на обучении {space.dist(pred, vecs[sm_pool]).mean():.3f}", flush=True)

    rows = []
    out_path = OUT / f"results_seed{args.seed}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows.append(r)
            done.add((r["target"], r["objective"]))
    rng = random.Random(args.seed + 100)
    with ProcessPoolExecutor(args.workers) as pool:
        for t in targets:
            meta = metas[t]
            true_mode = meta["spec"]["params"].get("mode") if meta["group"] == "sm" else None
            # Базовые линии без поиска: выбор из готовых эффектов Supermassive набора
            oracle = min(sm_pool, key=lambda j: d[t, j])
            for key, (space, vecs) in spaces.items():
                if (t, key) in done:
                    continue
                t0 = time.time()
                target_vec = vecs[t]
                nearest = min(sm_pool, key=lambda j: float(space.dist(vecs[j], target_vec)[0]))
                ranked = surrogate_modes(predictors[key], target_vec, space, args.seed)
                best = real_search(pool, target_vec, space, ranked[:args.top_modes], args.generations,
                                   args.popsize, args.seed)
                found_mels = pool.submit(_mels_job, best["params"]).result()
                # Базовая линия «только суррогат»: лучший режим и ручки по суррогату без честных рендеров
                _, s_mode, s_x = ranked[0]
                surrogate_mels = pool.submit(_mels_job, unit_to_params(s_mode, s_x)).result()
                row = {"target": t, "group": meta["group"], "objective": key, "true_mode": true_mode,
                       "found_mode": best["mode"], "found_params": best["params"], "search_dist": best["dist"],
                       "evals": best["evals"], "surrogate_top": [m for _, m, _ in ranked[:args.top_modes]],
                       "true_mode_rank_surrogate": ([m for _, m, _ in ranked].index(true_mode) if true_mode else None),
                       "logmel_found": mel_distance(found_mels, mels[t]),
                       "logmel_surrogate_only": mel_distance(surrogate_mels, mels[t]), "surrogate_mode": s_mode,
                       "logmel_nearest_pool": float(d[t, nearest]), "logmel_oracle_pool": float(d[t, oracle]),
                       "seconds": time.time() - t0}
                if true_mode and best["mode"] != true_mode:
                    # Эквивалентен ли найденный режим по звуку: целевые ручки, но найденный режим
                    alt = dict(meta["spec"]["params"], mode=best["mode"])
                    row["logmel_found_mode_target_knobs"] = mel_distance(
                        pool.submit(_mels_job, alt).result(), mels[t])
                rows.append(row)
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                print(f"[{len(rows)}] цель {t} ({meta['group']}, {true_mode or 'dsp'}) {key}: режим {best['mode']}, "
                      f"log-mel {row['logmel_found']:.2f} дБ (ближайший из набора {row['logmel_nearest_pool']:.2f}, "
                      f"лучший из набора {row['logmel_oracle_pool']:.2f}), {best['evals']} рендеров, "
                      f"{row['seconds']:.0f} с", flush=True)
            if not any(r["target"] == t and r["objective"] == "random" for r in rows):
                specs = [tfe.random_sm_spec(rng)["params"] for _ in range(args.random)]
                rand_mels = list(pool.map(_mels_job, specs))
                ds = [mel_distance(m, mels[t]) for m in rand_mels]
                row = {"target": t, "group": meta["group"], "objective": "random", "true_mode": true_mode,
                       "logmel_found": float(np.mean(ds)), "logmel_random_best": float(np.min(ds))}
                rows.append(row)
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summarize(rows, OUT / f"summary_seed{args.seed}.json")


def summarize(rows: list[dict], path: Path) -> None:
    out = {}
    for group in ("sm", "dsp"):
        for obj in ("emb", "hand", "random"):
            rs = [r for r in rows if r["group"] == group and r["objective"] == obj]
            if not rs:
                continue
            s = {"n": len(rs), "logmel_found_mean": float(np.mean([r["logmel_found"] for r in rs])),
                 "logmel_found_median": float(np.median([r["logmel_found"] for r in rs]))}
            if obj != "random":
                s["logmel_surrogate_only_mean"] = float(np.mean([r["logmel_surrogate_only"] for r in rs]))
                s["logmel_nearest_pool_mean"] = float(np.mean([r["logmel_nearest_pool"] for r in rs]))
                s["logmel_oracle_pool_mean"] = float(np.mean([r["logmel_oracle_pool"] for r in rs]))
                if group == "sm":
                    s["mode_exact"] = int(sum(r["found_mode"] == r["true_mode"] for r in rs))
                    s["true_mode_in_surrogate_top"] = int(sum(r["true_mode"] in r["surrogate_top"] for r in rs))
                    alt = [r["logmel_found_mode_target_knobs"] for r in rs if "logmel_found_mode_target_knobs" in r]
                    s["wrong_mode_equiv_logmel_mean"] = float(np.mean(alt)) if alt else None
            else:
                s["logmel_random_best_mean"] = float(np.mean([r["logmel_random_best"] for r in rs]))
            out[f"{group}/{obj}"] = s
    print(json.dumps(out, ensure_ascii=False, indent=1))
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-fast", action="store_true")
    parser.add_argument("--check-n", type=int, default=18)
    parser.add_argument("--embedder", default=str(tfe.OUT / "embedder.pt"), help="модель для --check-fast")
    parser.add_argument("--targets-sm", type=int, default=10)
    parser.add_argument("--targets-dsp", type=int, default=5)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--top-modes", type=int, default=3)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--popsize", type=int, default=9)
    parser.add_argument("--random", type=int, default=5, help="случайных настроек на цель (базовая линия)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    check_fast(args) if args.check_fast else run(args)


if __name__ == "__main__":
    main()
