"""
Сеть слепой оценки отпечатка (app/eval/blind_fingerprint_net.py): набор, обучение, сравнение с ручным оценщиком.

    python scripts/train_blind_fingerprint.py drys
    python scripts/train_blind_fingerprint.py generate --kind dsp --workers 6
    python scripts/train_blind_fingerprint.py generate --kind sm --workers 8
    python scripts/train_blind_fingerprint.py train
    python scripts/train_blind_fingerprint.py real

drys: 8-секундные фрагменты сухих вокалов Cambridge (data/raw/dry_vocals/cambridge), до 3 дорожек на песню; песни
делятся на A (обучение) и B (отложенные вокалы, ~20 %).

generate: эффекты — 472 из data/runs/fingerprint/embedding (300 нашего DSP, 150 случайных Supermassive, 22 режима
по умолчанию; отпечаток уже снят) + новые случайные цепочки DSP. Каждый эффект рендерится на нескольких сухих с
аугментацией до эффекта (уровень, полки EQ, иногда компрессор): обучающие эффекты — K сухих из A и 1 из B,
отложенные — 2 из A и 2 из B. Supermassive — свежий экземпляр на каждый рендер. Для отложенных рендеров сразу
считается ручной оценщик (blind_fingerprint) — база сравнения.

Сплиты: отложенные эффекты (20 % DSP и 20 % случайных Supermassive), целиком отложенные режимы Supermassive
(HELD_MODES), отложенные вокалы (B).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.blind_fingerprint_net import SECONDS, SR  # noqa: E402

OUT = Path("data/runs/fingerprint/blind_net")
EMB = Path("data/runs/fingerprint/embedding/effects")
CAMBRIDGE = Path("data/raw/dry_vocals/cambridge")
HELD_MODES = ("Hydra", "Cirrus Major", "Orion", "Pleiades")
K_TRAIN = {"dsp": 6, "sm": 3}


def _song(path: Path) -> str:
    parts = path.stem.split("_")
    return "_".join(parts[:-2]) if len(parts) > 2 else path.stem


def _is_b(key: str, frac: float = 0.2) -> bool:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


# ------------------------------------------------------------------------------------------------- сухие

def make_drys(args) -> None:
    """Фрагменты по 8 с: окно с наибольшей долей активных кадров (слова с паузами, а не сплошная тишина)."""
    import librosa
    import soundfile as sf
    files = sorted(p for p in CAMBRIDGE.glob("*.wav") if "vox" in p.name.lower() or "vocal" in p.name.lower())
    by_song: dict[str, list[Path]] = {}
    for p in files:
        by_song.setdefault(_song(p), []).append(p)
    n = int(SECONDS * SR)
    frags, songs, names = [], [], []
    for song, paths in sorted(by_song.items()):
        paths = sorted(paths, key=lambda p: (0 if "lead" in p.name.lower() else 1, p.name))[:3]
        for p in paths:
            y, sr = sf.read(p, dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            if len(y) < n:
                continue
            hop = SR // 2
            frame = SR // 20
            e = np.array([np.sum(y[i:i + frame] ** 2) for i in range(0, len(y) - frame, frame)])
            if e.max() <= 0:
                continue
            active = (10 * np.log10(e / e.max() + 1e-12) > -35.0).astype(float)
            per_win = n // frame
            csum = np.concatenate([[0], np.cumsum(active)])
            starts = np.arange(0, len(active) - per_win, hop // frame)
            if not len(starts):
                continue
            scores = csum[starts + per_win] - csum[starts]
            # Лучшее окно, но не сплошное пение: пауз должно быть хотя бы 10 %
            ok = scores < 0.9 * per_win
            best = starts[np.argmax(np.where(ok, scores, scores * 0.5))]
            seg = y[best * frame:best * frame + n]
            if np.sqrt(np.mean(seg ** 2)) < 1e-3:
                continue
            frags.append(seg / (np.abs(seg).max() + 1e-9) * 0.7)
            songs.append(song)
            names.append(p.name)
    split = np.array(["B" if _is_b(s) else "A" for s in songs])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "drys.npz", audio=np.stack(frags).astype(np.float32), song=np.array(songs), name=np.array(names),
             split=split)
    print(f"[+] сухих {len(frags)}: A {int((split == 'A').sum())}, B {int((split == 'B').sum())}, "
          f"песен {len(set(songs))}")


_DRYS = None


def _drys():
    global _DRYS
    if _DRYS is None:
        z = np.load(OUT / "drys.npz")
        _DRYS = (z["audio"], z["split"])
    return _DRYS


def augment(y: np.ndarray, rng: random.Random) -> np.ndarray:
    """Обработка до эффекта: уровень, полки EQ, иногда компрессор — сухие в миксах не стерильны."""
    from pedalboard import Compressor, Gain, HighShelfFilter, LowShelfFilter, Pedalboard
    fx = [LowShelfFilter(cutoff_frequency_hz=rng.uniform(120, 300), gain_db=rng.uniform(-4, 4)),
          HighShelfFilter(cutoff_frequency_hz=rng.uniform(3000, 8000), gain_db=rng.uniform(-4, 5))]
    if rng.random() < 0.4:
        fx.append(Compressor(threshold_db=rng.uniform(-24, -10), ratio=rng.uniform(2, 6), attack_ms=rng.uniform(2, 20),
                             release_ms=rng.uniform(50, 200)))
    fx.append(Gain(gain_db=rng.uniform(-12, 0)))
    return Pedalboard(fx)(y, SR)


# ------------------------------------------------------------------------------------------------- генерация

def effect_list(n_new_dsp: int, seed: int) -> list[dict]:
    """Эффекты: из набора эмбеддинга (с отпечатком) + новые случайные DSP (отпечаток снимается в генерации)."""
    from scripts.train_fingerprint_embedding import random_dsp_spec
    out = []
    for path in sorted(EMB.glob("*.npz")):
        meta = json.loads(str(np.load(path)["meta"]))
        out.append({"id": f"emb{meta['idx']:05d}", "group": meta["group"], "spec": meta["spec"],
                    "fingerprint": meta["fingerprint"]})
    rng = random.Random(seed)
    for i in range(n_new_dsp):
        out.append({"id": f"new{i:05d}", "group": "dsp", "spec": random_dsp_spec(rng), "fingerprint": None})
    for e in out:
        mode = e["spec"].get("params", {}).get("mode")
        if mode in HELD_MODES:
            e["split"] = "held_mode"
        elif e["group"] == "sm_default":
            e["split"] = "train"  # 22 режима по умолчанию — обучение; их отложенная часть уже в held_mode
        else:
            e["split"] = "test" if _is_b("effect:" + e["id"]) else "train"
    return out


def process_effect(e: dict) -> str:
    from scripts.train_fingerprint_embedding import build_effect
    from app.eval.blind_fingerprint import blind_fingerprint
    from app.eval.blind_fingerprint_net import wet_features
    from app.eval.effect_fingerprint import _render, fingerprint
    path = OUT / "renders" / f"{e['id']}.npz"
    if path.exists():
        return "skip"
    t0 = time.time()
    effect = build_effect(e["spec"])
    fp = e["fingerprint"] or fingerprint(effect, sr=48000, ir_seconds=8.0)
    audio, split = _drys()
    rng = random.Random(e["id"])
    a_idx, b_idx = np.nonzero(split == "A")[0].tolist(), np.nonzero(split == "B")[0].tolist()
    kind = "sm" if e["spec"]["kind"] == "vst" else "dsp"
    if e["split"] == "train":
        picks = [(i, "A") for i in rng.sample(a_idx, K_TRAIN[kind])] + [(rng.choice(b_idx), "B")]
    else:
        picks = [(i, "A") for i in rng.sample(a_idx, 2)] + [(i, "B") for i in rng.sample(b_idx, 2)]
    mels, acfs, rows = [], [], []
    for dry_i, dry_split in picks:
        y = augment(audio[dry_i], rng)
        wet = _render(effect, np.stack([y, y]), SR)
        if not np.isfinite(wet).all() or np.abs(wet).max() < 1e-6:
            continue
        mel, acf = wet_features(wet, SR)
        mels.append(mel.astype(np.float16))
        acfs.append(acf)
        row = {"dry": int(dry_i), "dry_split": dry_split}
        if e["split"] != "train" or dry_split == "B":
            try:
                row["blind"] = blind_fingerprint(wet, SR)
            except Exception as exc:  # ручной оценщик на странном входе — без базы, но рендер оставляем
                row["blind_error"] = repr(exc)
        rows.append(row)
    if not mels:
        raise ValueError("нет годных рендеров")
    # Запись через временный файл: два прогона generate могут дойти до одного эффекта одновременно
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, mels=np.stack(mels), acf=np.stack(acfs),
                        meta=json.dumps({"id": e["id"], "group": e["group"], "split": e["split"], "spec": e["spec"],
                                         "fingerprint": fp, "renders": rows, "seconds": time.time() - t0},
                                        ensure_ascii=False, default=float))
    if path.exists():
        tmp.unlink()
    else:
        os.replace(tmp, path)
    return f"{time.time() - t0:.1f}s"


def generate(args) -> None:
    (OUT / "renders").mkdir(parents=True, exist_ok=True)
    effects = effect_list(args.new_dsp, args.seed)
    want = "vst" if args.kind == "sm" else "dsp"
    jobs = [e for e in effects if e["spec"]["kind"] == want and not (OUT / "renders" / f"{e['id']}.npz").exists()]
    if args.limit:
        jobs = jobs[:args.limit]
    random.Random(os.getpid()).shuffle(jobs)  # параллельные прогоны идут в разном порядке и реже пересекаются
    print(f"[+] {args.kind}: заданий {len(jobs)}, процессов {args.workers}", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_effect, e): e["id"] for e in jobs}
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as exc:  # битый эффект не должен ронять весь набор
                print(f"[-] {futures[fut]}: {exc!r}", flush=True)
            if done % 25 == 0 or done == len(jobs):
                print(f"[+] {done}/{len(jobs)} за {time.time() - t0:.0f} с", flush=True)


# ------------------------------------------------------------------------------------------------- обучение

def load_renders(lag_sim: bool = False):
    from app.eval.blind_fingerprint_net import echo_targets, lag_similarity, target_vector
    mels, acfs, vecs, masks, has, periods, info = [], [], [], [], [], [], []
    for path in sorted((OUT / "renders").glob("*.npz")):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        vec, mask = target_vector(meta["fingerprint"])
        h, per = echo_targets(meta["fingerprint"])
        for k, row in enumerate(meta["renders"]):
            mels.append(z["mels"][k])
            acf = z["acf"][k][:2]
            if lag_sim:  # сходство по лагам считается по сохраненным log-mel, перерендер не нужен
                acf = np.concatenate([acf, lag_similarity(z["mels"][k])])
                acf[:, :3] = 0.0
            acfs.append(acf)
            vecs.append(vec)
            masks.append(mask)
            has.append(h)
            periods.append(per)
            mode = meta["spec"].get("params", {}).get("mode")
            info.append({"id": meta["id"], "group": meta["group"], "split": meta["split"], "mode": mode,
                         "dry_split": row["dry_split"], "blind": row.get("blind"), "fp": meta["fingerprint"]})
    return (np.stack(mels), np.stack(acfs), np.stack(vecs), np.stack(masks), np.array(has, np.float32),
            np.stack(periods), info)


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


def train_model(data, train_idx, val_idx, seed: int, epochs: int, device: str):
    import torch
    from app.eval.blind_fingerprint_net import BlindFingerprintNet
    mels, acfs, vecs, masks, has, periods, _ = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BlindFingerprintNet(n_acf=data[1].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = epochs * max(1, len(train_idx) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)

    def batch(idx, aug):
        m = torch.from_numpy(mels[idx].astype(np.float32)).to(device)
        if aug:  # сдвиг по времени: эффект не должен зависеть от того, где в окне слова
            shift = np.random.randint(0, m.shape[-1])
            m = torch.roll(m, shift, dims=-1)
        return (m, torch.from_numpy(acfs[idx]).to(device), torch.from_numpy(vecs[idx]).to(device),
                torch.from_numpy(masks[idx]).to(device), torch.from_numpy(has[idx]).to(device),
                torch.from_numpy(periods[idx]).to(device))

    def loss_fn(out, tgt):
        reg, per, echo = out
        _, _, v, mk, h, p = tgt
        l_reg = (torch.nn.functional.smooth_l1_loss(reg, v, reduction="none") * mk).sum() / mk.sum().clamp(min=1)
        l_echo = torch.nn.functional.binary_cross_entropy_with_logits(echo, h)
        # Период — кросс-энтропия с мягкой целью (гаусс σ = 1 лаг вокруг дробного истинного лага)
        lags = torch.arange(per.shape[-1], device=per.device, dtype=per.dtype)
        target = torch.softmax(-0.5 * (lags[None, None] - p[..., None]) ** 2, dim=-1)
        ce = -(target * torch.log_softmax(per, dim=-1)).sum(-1).mean(1)
        l_per = (ce * h).sum() / h.sum().clamp(min=1)
        return l_reg + 0.5 * l_echo + 0.5 * l_per

    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        for i in range(0, len(perm) - 63, 64):
            tgt = batch(perm[i:i + 64], aug=True)
            loss = loss_fn(model(tgt[0], tgt[1]), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(val_idx), 128):
                tgt = batch(val_idx[i:i + 128], aug=False)
                vl.append(float(loss_fn(model(tgt[0], tgt[1]), tgt)) * len(tgt[0]))
            vloss = sum(vl) / max(len(val_idx), 1)
        if vloss < best:
            best, best_state = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    seed {seed} эпоха {ep}: val {vloss:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model.eval()


def predict_all(model, data, idx, device):
    import torch
    from app.eval.blind_fingerprint_net import LAG_S, lag_from_logits, vector_to_dict
    mels, acfs = data[0], data[1]
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 128):
            j = idx[i:i + 128]
            reg, per, echo = model(torch.from_numpy(mels[j].astype(np.float32)).to(device),
                                   torch.from_numpy(acfs[j]).to(device))
            for r, p, e in zip(reg.cpu().numpy(), per.cpu().numpy(), torch.sigmoid(echo).cpu().numpy()):
                d = vector_to_dict(r)
                d["echo_period_l_ms"], d["echo_period_r_ms"] = (lag_from_logits(p[c]) * LAG_S * 1000.0 for c in (0, 1))
                d["echo_prob"] = float(e)
                out.append(d)
    return out


EVAL_KEYS = ("side_mid_db", "iacc_late", "iacc_early", "rt60_s", "rt60_125", "rt60_1000", "rt60_4000", "rt60_8000",
             "tail_db_125", "tail_db_1000", "tail_db_8000", "drr_db", "predelay_ms", "echo_strength",
             "mod_spread_db", "ir_repeatability", "pitch_up_db", "saturation_db")


def evaluate(preds: list[dict], infos: list[dict]) -> dict:
    """Spearman по признакам (сеть и ручной оценщик на тех же рендерах), попадание периода ±10 мс."""
    res = {}
    for key in EVAL_KEYS:
        truth = [i["fp"].get(key, np.nan) for i in infos]
        net = [p.get(key, np.nan) for p in preds]
        hand = [(i["blind"] or {}).get(key, np.nan) for i in infos]
        res[key] = {"net": _spearman(net, truth), "hand": _spearman(hand, truth),
                    "hand_cover": float(np.mean([np.isfinite(h) for h in np.asarray(hand, float)]))}
    hits_net, hits_hand, n = 0, 0, 0
    for p, i in zip(preds, infos):
        fp = i["fp"]
        if float(fp.get("echo_strength", 0) or 0) <= 0.5:
            continue
        for c in ("l", "r"):
            true = float(fp.get(f"echo_period_{c}_ms", 0) or 0)
            if true <= 0:
                continue
            n += 1
            hits_net += abs(p[f"echo_period_{c}_ms"] - true) <= 10.0
            h = (i["blind"] or {}).get(f"echo_period_{c}_ms")
            hits_hand += h is not None and np.isfinite(h) and abs(h - true) <= 10.0
    res["period_hit"] = {"net": hits_net / max(n, 1), "hand": hits_hand / max(n, 1), "n": n}
    has = np.array([float(i["fp"].get("echo_strength", 0) or 0) > 0.5 for i in infos])
    prob = np.array([p["echo_prob"] for p in preds])
    res["echo_detect_acc"] = float(np.mean((prob > 0.5) == has)) if len(has) else float("nan")
    return res


def train(args) -> None:
    import torch
    from app.eval.blind_fingerprint_net import save_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_renders(args.lag_sim)
    info = data[-1]
    split = np.array([i["split"] for i in info])
    dry = np.array([i["dry_split"] for i in info])
    train_all = np.nonzero((split == "train") & (dry == "A"))[0]
    rng = np.random.default_rng(0)
    # Валидация для ранней остановки — 10 % обучающих эффектов целиком (не рендеров)
    ids = sorted({info[k]["id"] for k in train_all})
    val_ids = set(rng.choice(ids, size=max(1, len(ids) // 10), replace=False).tolist())
    val_idx = np.array([k for k in train_all if info[k]["id"] in val_ids])
    train_idx = np.array([k for k in train_all if info[k]["id"] not in val_ids])
    groups = {
        "эффекты отложены, вокалы A": np.nonzero((split == "test") & (dry == "A"))[0],
        "эффекты отложены, вокалы B": np.nonzero((split == "test") & (dry == "B"))[0],
        "эффекты из обучения, вокалы B": np.nonzero((split == "train") & (dry == "B"))[0],
        "режимы SM отложены": np.nonzero(split == "held_mode")[0],
        "отложенные, только DSP": np.nonzero((split == "test") & np.array([i["group"] == "dsp" for i in info]))[0],
        "отложенные, только SM": np.nonzero(((split == "test") | (split == "held_mode"))
                                            & np.array([i["group"] != "dsp" for i in info]))[0],
    }
    print(f"[+] рендеров {len(info)}: обучение {len(train_idx)}, валидация {len(val_idx)}, "
          + ", ".join(f"{k} {len(v)}" for k, v in groups.items()), flush=True)
    report = {"device": device, "n": {k: int(len(v)) for k, v in groups.items()}, "seeds": {}}
    for seed in range(args.seeds):
        model = train_model(data, train_idx, val_idx, seed, args.epochs, device)
        report["seeds"][seed] = {}
        for name, idx in groups.items():
            if not len(idx):
                continue
            preds = predict_all(model, data, idx, device)
            report["seeds"][seed][name] = evaluate(preds, [info[k] for k in idx])
        if seed == 0:
            save_model(OUT / "blind_net.pt", model.cpu(), {"epochs": args.epochs, "train_renders": int(len(train_idx))})
            model.to(device)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print_report(report)


def print_report(report: dict) -> None:
    seeds = list(report["seeds"].values())
    for name in seeds[0]:
        print(f"\n== {name} (рендеров {report['n'][name]})")
        print(f"{'признак':<18}{'сеть':>14}{'ручной':>10}{'покрытие':>10}")
        for key in EVAL_KEYS:
            vals = [s[name][key]["net"] for s in seeds]
            r0 = seeds[0][name][key]
            print(f"{key:<18}{np.nanmean(vals):>8.2f} ±{np.nanstd(vals):4.2f}"
                  f"{r0['hand']:>10.2f}{r0['hand_cover']:>10.2f}")
        ph = [s[name]["period_hit"]["net"] for s in seeds]
        r0 = seeds[0][name]
        print(f"{'период ±10 мс':<18}{np.mean(ph):>8.0%} ±{np.std(ph):4.2f}{r0['period_hit']['hand']:>10.0%}"
              f"   (n={r0['period_hit']['n']})")
        print(f"{'повторы есть/нет':<18}{np.mean([s[name]['echo_detect_acc'] for s in seeds]):>8.0%}")


# ------------------------------------------------------------------------------------------------- реальные пары

def real(args) -> None:
    """Правдоподобие на реальных мокрых вокалах: предсказание сети и ручного оценщика рядом (истины нет)."""
    import soundfile as sf
    from app.eval.blind_fingerprint import blind_fingerprint
    from app.eval.blind_fingerprint_net import load_model, predict
    model = load_model(OUT / "blind_net.pt")
    keys = ("rt60_s", "rt60_1000", "rt60_8000", "tail_db_8000", "drr_db", "side_mid_db", "iacc_late",
            "echo_period_l_ms", "echo_period_r_ms", "echo_strength", "mod_spread_db", "pitch_up_db")
    rows = {}
    for d in sorted(Path("data/processed/real_pairs").iterdir()):
        path = d / "wet_true.wav"
        if not path.exists():
            continue
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.T
        n = int(SECONDS * sr)
        # Несколько окон по 8 с, медиана — устойчивее к одному неудачному куску
        starts = list(range(0, max(1, y.shape[1] - n), max(n // 2, 1)))[:8]
        preds = [predict(model, y[:, s:s + n], sr) for s in starts]
        net = {k: float(np.median([p[k] for p in preds])) for k in keys if k in preds[0]}
        net["echo_prob"] = float(np.median([p["echo_prob"] for p in preds]))
        try:
            hand = blind_fingerprint(y[:, :int(30 * sr)], sr)
        except Exception as exc:
            hand = {"error": repr(exc)}
        rows[d.name] = {"net": net, "hand": {k: hand.get(k) for k in keys}, "windows": len(starts)}
    (OUT / "real.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"{'пара':<22}" + "".join(f"{k[:12]:>13}" for k in keys) + f"{'echo_prob':>11}")
    for name, r in rows.items():
        for src in ("net", "hand"):
            vals = r[src]
            cells = "".join(f"{vals.get(k):>13.2f}" if isinstance(vals.get(k), (int, float)) and vals.get(k) is not None
                            else f"{'-':>13}" for k in keys)
            extra = f"{r['net']['echo_prob']:>11.2f}" if src == "net" else ""
            print(f"{(name if src == 'net' else '  ручной'):<22}{cells}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("drys")
    g = sub.add_parser("generate")
    g.add_argument("--kind", choices=("dsp", "sm"), required=True)
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--new-dsp", type=int, default=500)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--seeds", type=int, default=3)
    t.add_argument("--lag-sim", action="store_true", help="добавить сходство спектра по лагам (4 канала подсказки)")
    sub.add_parser("real")
    args = parser.parse_args()
    {"drys": make_drys, "generate": generate, "train": train, "real": real}[args.cmd](args)


if __name__ == "__main__":
    main()
