import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.mix_full import SR, tone_only


def test_tone_only_writes_preview_and_warning(tmp_path):
    t = np.arange(SR) / SR
    dry = np.vstack([0.1 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    tone = (0.8 * dry, {"plugins": [{"name": "compressor", "params": {"ratio": 0.3}}]})
    reasons = ["в вокал протек бас или бочка (85% энергии ниже 150 Гц)"]

    summary = tone_only(dry, tmp_path, tone=tone, reasons=reasons, truth=dry)

    assert (tmp_path / "preview.wav").exists() and (tmp_path / "1_tone.wav").exists()
    assert summary["symmetric_preset"] is None
    assert summary["logmel_symmetric"] == summary["logmel_tone"]
    assert json.loads((tmp_path / "chain.json").read_text(encoding="utf-8"))["space"] is None
    readme = (tmp_path / "README.txt").read_text(encoding="utf-8")
    assert "ВНИМАНИЕ" in readme and reasons[0] in readme
