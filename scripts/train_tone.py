"""
Дообучение MixingAIModel с ошибкой по звуку для эффектов тона (app/ml/tone_training.py).

    python scripts/train_tone.py --dataset-dir data/processed/dataset_v6 --init data/models/v6/best_model.pth \
        --save-dir data/models/v7_tone

Лосс = BCE выбора плагинов + регрессия ручек вне тона + gamma · ошибка спектра (предсказанная цепочка тона
на сухом фрагменте против истинной) + слабая регрессия ручек тона (reg_tone: держит ручки в разумной области,
пока звук их не различает). Эпоха 0 — оценка исходного чекпоинта на той же валидации: с ней и сравнивать.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset.dataset_schema import CHAIN_ORDER, dataset_param_ranges  # noqa: E402
from app.ml.checkpoint import load_model  # noqa: E402
from app.ml.model import MixingAIModel  # noqa: E402
from app.ml.tone_training import TONE_NAMES, ToneDataset, render, spectral_loss, tone_mask  # noqa: E402

SR = 44100
TONE_IDX = [CHAIN_ORDER.index(n) for n in TONE_NAMES]


def _reg(pred, true, mask):
    diff = 0.65 * (pred - true).abs() + 0.35 * (pred - true) ** 2
    return (diff * mask).sum() / (mask.sum() + 1e-8)


def step(model, batch, device, args, tmask, train: bool) -> dict:
    spec, desc, chain, params, pmask, audio, eq = (t.to(device) for t in batch)
    logits, pred = model(spec, chain if train else None, desc)
    probs = torch.sigmoid(logits)
    with torch.no_grad():
        target = render(audio, eq, params, chain, SR)
    gates = probs if train else (probs > 0.5).to(probs.dtype)
    sound = spectral_loss(render(audio, eq, pred, gates, SR), target)
    loss_class = torch.nn.functional.binary_cross_entropy_with_logits(logits, chain)
    reg_other = _reg(pred, params, pmask * (1 - tmask))
    reg_tone = _reg(pred, params, pmask * tmask)
    loss = loss_class + 12.0 * reg_other + args.reg_tone * 12.0 * reg_tone + args.gamma * sound.mean()
    return {"loss": loss, "class": loss_class.item(), "reg_other": reg_other.item(), "reg_tone": reg_tone.item(),
            "sound": sound.detach().cpu().numpy(), "pred_on": (probs > 0.5).float().cpu().numpy(),
            "true_on": chain.cpu().numpy()}


def evaluate(model, loader, device, args, tmask) -> dict:
    model.eval()
    sounds, pred_on, true_on, other, tone = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = step(model, batch, device, args, tmask, train=False)
            sounds.append(out["sound"]), pred_on.append(out["pred_on"]), true_on.append(out["true_on"])
            other.append(out["reg_other"]), tone.append(out["reg_tone"])
    p, t = np.vstack(pred_on), np.vstack(true_on)
    f1 = []
    for i in range(len(CHAIN_ORDER)):
        tp, fp, fn = [np.sum((p[:, i] == a) & (t[:, i] == b)) for a, b in ((1, 1), (1, 0), (0, 1))]
        prec, rec = (tp / (tp + fp) if tp + fp else 1.0), (tp / (tp + fn) if tp + fn else 1.0)
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"sound": float(np.mean(np.concatenate(sounds))), "f1": float(np.mean(f1)),
            "f1_tone": float(np.mean([f1[i] for i in TONE_IDX])),
            "reg_other": float(np.mean(other)), "reg_tone": float(np.mean(tone))}


def _fmt(m: dict) -> str:
    return (f"звук {m['sound']:.4f} | F1 {m['f1']:.3f} (тон {m['f1_tone']:.3f}) | "
            f"рег. вне тона {m['reg_other']:.4f} | рег. тона {m['reg_tone']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="data/processed/dataset_v6")
    parser.add_argument("--init", default="data/models/v6/best_model.pth", help="чекпоинт для дообучения; '' — с нуля")
    parser.add_argument("--save-dir", default="data/models/v7_tone")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=4.0, help="вес ошибки по звуку")
    parser.add_argument("--reg-tone", type=float, default=0.1, help="доля прежней регрессии для ручек тона")
    parser.add_argument("--crop-s", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=0, help="ограничить train (проверка), 0 — весь")
    parser.add_argument("--max-val", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_set = ToneDataset(args.dataset_dir, "train", SR, args.crop_s, augment=True)
    val_set = ToneDataset(args.dataset_dir, "val", SR, args.crop_s)
    if args.max_train:
        train_set.samples = train_set.samples[:args.max_train]
    if args.max_val:
        val_set.samples = val_set.samples[:args.max_val]
    loader = lambda ds, shuffle: DataLoader(ds, args.batch_size, shuffle=shuffle, drop_last=shuffle,  # noqa: E731
                                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    train_loader, val_loader = loader(train_set, True), loader(val_set, False)
    print(f"[*] {device} | train {len(train_set)} | val {len(val_set)} | gamma {args.gamma} | reg_tone {args.reg_tone}")

    if args.init:
        model, _ = load_model(args.init, device)
    else:
        model = MixingAIModel().to(device)
    tmask = tone_mask().to(device)
    param_ranges, _ = dataset_param_ranges(args.dataset_dir)

    best = evaluate(model, val_loader, device, args, tmask)
    print(f"Эпоха 00 (исходный чекпоинт) | {_fmt(best)}", flush=True)
    best_score = best["sound"] + best["reg_other"] - 0.2 * best["f1"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out = step(model, batch, device, args, tmask, train=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(out["loss"].item())
        scheduler.step()
        m = evaluate(model, val_loader, device, args, tmask)
        score = m["sound"] + m["reg_other"] - 0.2 * m["f1"]
        print(f"Эпоха {epoch:02d} | train {np.mean(losses):.4f} | {_fmt(m)} | {time.time() - t0:.0f} с", flush=True)
        if score < best_score:
            best_score, stale = score, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val": m,
                        "param_ranges": param_ranges, "dataset_dir": str(args.dataset_dir), "init": args.init,
                        "tone_training": {"gamma": args.gamma, "reg_tone": args.reg_tone}},
                       save_dir / "best_model.pth")
            print(f"  [+] сохранено -> {save_dir / 'best_model.pth'}", flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[!] ранняя остановка на эпохе {epoch}")
                break


if __name__ == "__main__":
    main()
