"""
Обучение кодировщика стиля обработки (app/eval/style_embedding.py) контрастным методом.

Банк данных строится блоками: 4 фрагмента сухого вокала (два из разных мест одного файла, два из других
песен) x 4 реалистичные цепочки (app/eval/style_chains.py) = 16 рендеров. Метка — цепочка: одинаковая
цепочка на разном содержании — позитив, разные цепочки (в т.ч. на том же фрагменте) — негативы
(Supervised Contrastive, Khosla et al. 2020). Вспомогательная цель — какие типы плагинов есть в цепочке.

Песни реальных тестовых пар исключены (style_chains.TEST_SONG_PREFIXES).

Пример:
    python scripts/train_style_encoder.py bank --blocks 2000 --split train
    python scripts/train_style_encoder.py bank --blocks 200 --split val
    python scripts/train_style_encoder.py train --minutes 60
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import numpy as np  # noqa: E402

BANK_DIR = Path("data/runs/style_encoder")
OUT_DIR = Path("data/models/style_encoder")
CONTENTS, CHAINS = 4, 4
EXCERPT_S, WARMUP_S = 4.0, 1.0
SR = 44100


def _read_excerpts(path: Path, rng: random.Random, n: int, seconds: float) -> list[np.ndarray]:
    """n фрагментов из громких мест файла, начала отстоят друг от друга минимум на seconds + 1 с."""
    import librosa
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.T
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    mono = y.mean(axis=0)
    length = int(seconds * SR)
    if len(mono) < length * n + SR:
        return []
    hop = SR // 4
    frame_power = (mono[: len(mono) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1)
    energy = np.convolve(frame_power, np.ones(int(seconds * 4)), mode="valid")
    good = np.flatnonzero(energy >= np.quantile(energy, 0.5))
    starts: list[int] = []
    for _ in range(50):
        if len(starts) == n:
            break
        s = int(rng.choice(good)) * hop
        if s + length <= y.shape[1] and all(abs(s - o) >= length + SR for o in starts):
            starts.append(s)
    if len(starts) < n:
        return []
    out = []
    for s in starts:
        seg = y[:, s:s + length]
        out.append(np.vstack([seg, seg]) if seg.shape[0] == 1 else seg[:2])
    return out


def render_chain(dsp, x: np.ndarray, chain) -> np.ndarray:
    """apply_chain для стерео: подавление резонансов в DSPEngine работает только с моно, поэтому по каналам."""
    from app.audio.vocal_enhancer import VocalEnhancer
    from app.dataset.dataset_schema import ChainConfig
    rest = [p for p in chain.plugins if p.name != "resonance_suppressor"]
    if len(rest) != len(chain.plugins):
        phys = dsp.get_physical_params(next(p for p in chain.plugins if p.name == "resonance_suppressor"))
        enhancer = VocalEnhancer(sr=SR)
        x = np.vstack([enhancer.suppress_resonances(ch, threshold_db=phys["threshold_db"],
                                                    max_attenuation_db=phys["max_attenuation_db"])
                       for ch in x]).astype(np.float32)
    return dsp.apply_chain(x, SR, ChainConfig(plugins=rest))


def make_block(task: tuple[int, list[str]], keep_audio: bool = False) -> dict | None:
    """
    Один блок банка: признаки 16 рендеров, метки цепочек и фрагментов, флаги плагинов
    (keep_audio — еще и сами рендеры (CHAINS, CONTENTS) для проверки других метрик).
    """
    import warnings
    warnings.filterwarnings("ignore")
    from app.audio.dsp_engine import DSPEngine
    from app.dataset.dataset_generator import DatasetGenerator
    from app.eval.style_chains import chain_flags, sample_realistic_chain
    from app.eval.style_embedding import style_features
    from app.mixing.chain_optimizer import CHAIN_INPUT_LUFS, normalize_lufs

    seed, files = task
    rng = random.Random(seed)
    paths = [Path(f) for f in files]
    seconds = EXCERPT_S + WARMUP_S
    for _ in range(10):
        a = rng.choice(paths)
        others = [p for p in paths if DatasetGenerator.song_key(p) != DatasetGenerator.song_key(a)]
        b = rng.choice(others)
        c = rng.choice([p for p in others if DatasetGenerator.song_key(p) != DatasetGenerator.song_key(b)])
        exc = _read_excerpts(a, rng, 2, seconds)
        exc_b, exc_c = _read_excerpts(b, rng, 1, seconds), _read_excerpts(c, rng, 1, seconds)
        if len(exc) == 2 and exc_b and exc_c:
            break
    else:
        return None
    contents = [normalize_lufs(x, SR, CHAIN_INPUT_LUFS + rng.uniform(-2.0, 2.0)) for x in exc + exc_b + exc_c]
    chains = [sample_realistic_chain(rng) for _ in range(CHAINS)]
    dsp = DSPEngine()
    warm = int(WARMUP_S * SR)
    audio = [[None] * CONTENTS for _ in range(CHAINS)]
    feats = np.zeros((CHAINS, CONTENTS, 2, 64, int(EXCERPT_S * SR) // 512 + 1), dtype=np.float16)
    import contextlib
    import io
    for i, chain in enumerate(chains):
        for k, x in enumerate(contents):
            with contextlib.redirect_stdout(io.StringIO()):
                y = render_chain(dsp, x, chain)
            y = np.nan_to_num(np.asarray(y, dtype=np.float32))
            if y.ndim == 1:
                y = np.vstack([y, y])
            y = y[:, warm:warm + int(EXCERPT_S * SR)]
            if keep_audio:
                audio[i][k] = y
            f = style_features(y, SR)
            feats[i, k] = f[..., :feats.shape[-1]]
    return {"feats": feats, "flags": np.stack([chain_flags(c) for c in chains]),
            "chains": [[{"name": p.name, "params": p.params} for p in c.plugins] for c in chains],
            "files": [str(a), str(a), str(b), str(c)], **({"audio": audio} if keep_audio else {})}


def build_bank(args) -> None:
    from app.eval.style_chains import content_pool
    train, val = content_pool()
    files = [str(f) for f in (train if args.split == "train" else val)]
    print(f"[*] Пул {args.split}: {len(files)} файлов; блоков {args.blocks}")
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    base_seed = 1_000_000 if args.split == "val" else 0
    tasks = [(base_seed + args.offset + i, files) for i in range(args.blocks)]
    feats, flags, meta = [], [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, block in enumerate(pool.map(make_block, tasks, chunksize=2), start=1):
            if block is not None:
                feats.append(block["feats"])
                flags.append(block["flags"])
                meta.append({"chains": block["chains"], "files": block["files"]})
            if n % 50 == 0:
                print(f"    {n}/{len(tasks)} за {time.perf_counter() - started:.0f} с", flush=True)
    out = BANK_DIR / f"bank_{args.split}{args.suffix}.npz"
    np.savez(out, feats=np.stack(feats), flags=np.stack(flags))
    (BANK_DIR / f"bank_{args.split}{args.suffix}.json").write_text(json.dumps(meta, ensure_ascii=False, default=float),
                                                                  encoding="utf-8")
    print(f"[+] {out}: {len(feats)} блоков за {time.perf_counter() - started:.0f} с")


def supcon_loss(emb, labels, temperature: float):
    import torch
    sim = emb @ emb.T / temperature
    n = emb.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * pos).sum(1).div(pos.sum(1).clamp_min(1)).mean()


def bank_metrics(net, feats: np.ndarray, device) -> dict:
    """
    На блоках: доля троек, где та же цепочка на другом фрагменте ближе, чем другая цепочка на том же фрагменте,
    и top-1 поиска (ближайший сосед из других фрагментов — с той же цепочкой).
    """
    import torch
    from app.eval.style_embedding import WINDOW_FRAMES
    net.eval()
    embs = []
    with torch.no_grad():
        for blk in feats:
            x = torch.from_numpy(blk.reshape(-1, *blk.shape[2:]).astype(np.float32)).to(device)
            t0 = (x.shape[-1] - WINDOW_FRAMES) // 2
            e, _ = net(x[..., t0:t0 + WINDOW_FRAMES])
            embs.append(e.view(CHAINS, CONTENTS, -1).cpu().numpy())
    embs = np.stack(embs)  # (B, chains, contents, D)
    wins, total = 0, 0
    for e in embs:
        d = 1 - np.einsum("ikd,jld->ikjl", e, e)
        for i in range(CHAINS):
            for k in range(CONTENTS):
                for l in range(CONTENTS):
                    if l == k:
                        continue
                    for j in range(CHAINS):
                        if j == i:
                            continue
                        wins += d[i, k, i, l] < d[i, k, j, k]
                        total += 1
    flat = embs.reshape(-1, embs.shape[-1])
    blk = np.repeat(np.arange(len(embs)), CHAINS * CONTENTS)
    chain = np.tile(np.repeat(np.arange(CHAINS), CONTENTS), len(embs)) + blk * CHAINS
    content = np.tile(np.arange(CONTENTS), CHAINS * len(embs)) + blk * CONTENTS
    sim = flat @ flat.T
    sim[content[:, None] == content[None, :]] = -9  # сосед только с другого фрагмента
    top1 = float(np.mean(chain[sim.argmax(1)] == chain))
    net.train()
    return {"triplet_acc": wins / max(total, 1), "top1": top1}


def random_eq_curve(rng: np.random.Generator, n_mels: int, std_db: float) -> np.ndarray:
    """Гладкая кривая EQ в масштабе признаков (дБ / 20): 1-3 колокола по оси мел-полос и наклон."""
    bins = np.arange(n_mels)
    curve = np.zeros(n_mels)
    for _ in range(rng.integers(1, 4)):
        curve += rng.normal(0.0, std_db) * np.exp(-0.5 * ((bins - rng.uniform(0, n_mels)) / rng.uniform(2, 12)) ** 2)
    curve += rng.normal(0.0, std_db / 3) * (bins / n_mels - 0.5) * 2
    return (curve / 20.0).astype(np.float32)


def eq_hard_negatives(blk: np.ndarray, rng: np.random.Generator, prob: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Аугментация в пространстве признаков: log-mel после стационарного EQ ≈ log-mel + |H|² в дБ, поэтому
    последняя цепочка блока с вероятностью prob заменяется копией первой с добавленной кривой EQ (2-6 дБ)
    на всех 4 фрагментах. Получается новая «цепочка»: позитивы по-прежнему на разном содержании, а на том же
    содержании появляется трудный негатив, отличающийся только эквализацией.
    """
    blk = blk.copy()
    replaced = rng.random(len(blk)) < prob
    for b in np.flatnonzero(replaced):
        curve = random_eq_curve(rng, blk.shape[-2], rng.uniform(2.0, 6.0))
        blk[b, -1] = blk[b, 0] + curve[None, None, :, None]
    return blk, replaced


class _Banks:
    """Несколько банков как один массив по оси блоков (memmap .npy, чтобы не держать 8+ ГБ в памяти)."""

    def __init__(self, arrays: list[np.ndarray]):
        self.arrays = arrays
        self.offsets = np.cumsum([0] + [len(a) for a in arrays])
        self.shape = (int(self.offsets[-1]), *arrays[0].shape[1:])

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        idx = np.atleast_1d(idx)
        bank = np.searchsorted(self.offsets, idx, side="right") - 1
        return np.stack([self.arrays[b][i - self.offsets[b]] for b, i in zip(bank, idx)])


def load_banks(suffixes: list[str]) -> tuple[_Banks, np.ndarray]:
    """Банки train: .npz при первом обращении распаковывается в .npy рядом (для memmap)."""
    feats, flags = [], []
    for suffix in suffixes:
        npy = BANK_DIR / f"bank_train{suffix}_feats.npy"
        if not npy.exists():
            with np.load(BANK_DIR / f"bank_train{suffix}.npz") as z:
                np.save(npy, z["feats"])
                np.save(BANK_DIR / f"bank_train{suffix}_flags.npy", z["flags"])
        feats.append(np.load(npy, mmap_mode="r"))
        flags.append(np.load(BANK_DIR / f"bank_train{suffix}_flags.npy"))
    return _Banks(feats), np.concatenate(flags)


def train(args) -> None:
    import torch
    from app.eval.style_chains import FLAG_PLUGINS
    from app.eval.style_embedding import WINDOW_FRAMES, StyleNet
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    feats, flags = load_banks(args.train_banks)
    val = np.load(BANK_DIR / "bank_val.npz")["feats"]
    print(f"[*] train {feats.shape}, val {val.shape}, устройство {device}")

    config = {"width": args.width, "embed_dim": args.embed_dim}
    net = StyleNet(**config).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    deadline = time.perf_counter() + args.minutes * 60
    steps_est = args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_est, pct_start=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    best = -1.0
    t_frames = feats.shape[-1]
    for step in range(1, steps_est + 1):
        idx = rng.choice(len(feats), args.blocks_per_batch, replace=False)
        blk = feats[idx].astype(np.float32)  # (b, chains, contents, 2, M, T)
        replaced = np.zeros(len(idx), dtype=bool)
        if args.eq_aug > 0:
            blk, replaced = eq_hard_negatives(blk, rng, args.eq_aug)
        b = len(idx)
        x = blk.reshape(b * CHAINS * CONTENTS, 2, blk.shape[-2], t_frames)
        starts = rng.integers(0, t_frames - WINDOW_FRAMES + 1, size=len(x))
        x = np.stack([x[i, ..., s:s + WINDOW_FRAMES] for i, s in enumerate(starts)]).astype(np.float32)
        # Сдвиг уровня (±1.5 дБ): устойчивость к неточному выравниванию громкости
        x += rng.uniform(-0.075, 0.075, size=(len(x), 1, 1, 1)).astype(np.float32)
        labels = np.repeat(np.arange(b * CHAINS), CONTENTS)
        blk_flags = flags[idx].copy()
        blk_flags[replaced, -1] = blk_flags[replaced, 0]
        blk_flags[replaced, -1, FLAG_PLUGINS.index("eq")] = 1.0
        y_flags = np.repeat(blk_flags.reshape(b * CHAINS, -1), CONTENTS, axis=0)
        xt = torch.from_numpy(x).to(device)
        emb, logits = net(xt)
        loss_con = supcon_loss(emb, torch.from_numpy(labels).to(device), args.temperature)
        loss_flags = bce(logits, torch.from_numpy(y_flags).to(device))
        loss = loss_con + args.flags_weight * loss_flags
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == steps_est or time.perf_counter() > deadline:
            m = bank_metrics(net, val, device)
            row = {"step": step, "loss": loss_con.item(), "flags": loss_flags.item(), **m,
                   "minutes": round((time.perf_counter() - deadline) / 60 + args.minutes, 1)}
            log.append(row)
            print(json.dumps(row), flush=True)
            if m["triplet_acc"] > best:
                best = m["triplet_acc"]
                torch.save({"model": net.state_dict(), "config": config, "step": step, "val": m,
                            "args": vars(args)}, OUT_DIR / args.out_name)
            if time.perf_counter() > deadline:
                print("[*] Время вышло")
                break
    (OUT_DIR / args.out_name.replace(".pt", "_log.json")).write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[+] Лучший triplet_acc на val: {best:.3f} -> {OUT_DIR / args.out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Кодировщик стиля обработки")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_bank = sub.add_parser("bank")
    p_bank.add_argument("--blocks", type=int, default=2000)
    p_bank.add_argument("--split", choices=["train", "val"], default="train")
    p_bank.add_argument("--workers", type=int, default=16)
    p_bank.add_argument("--offset", type=int, default=0)
    p_bank.add_argument("--suffix", default="")
    p_train = sub.add_parser("train")
    p_train.add_argument("--train-banks", nargs="*", default=[""])
    p_train.add_argument("--minutes", type=float, default=60.0)
    p_train.add_argument("--steps", type=int, default=8000)
    p_train.add_argument("--blocks-per-batch", type=int, default=12)
    p_train.add_argument("--lr", type=float, default=2e-3)
    p_train.add_argument("--temperature", type=float, default=0.1)
    p_train.add_argument("--flags-weight", type=float, default=0.2)
    p_train.add_argument("--width", type=int, default=32)
    p_train.add_argument("--embed-dim", type=int, default=128)
    p_train.add_argument("--eval-every", type=int, default=250)
    p_train.add_argument("--cpu", action="store_true")
    p_train.add_argument("--eq-aug", type=float, default=0.0,
                         help="Вероятность трудного негатива «та же цепочка + EQ» на блок (см. eq_hard_negatives)")
    p_train.add_argument("--out-name", default="style_encoder.pt")
    parsed = parser.parse_args()
    build_bank(parsed) if parsed.cmd == "bank" else train(parsed)
