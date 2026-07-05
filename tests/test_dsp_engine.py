import pytest
import numpy as np
from app.core.dsp_engine import DSPEngine
from app.core.dataset_schema import ChainConfig, PluginConfig

def test_norm_denorm_value():
    dsp = DSPEngine()
    
    # Normal bounds
    assert dsp.denormalize_value(0.0, -10.0, 10.0) == -10.0
    assert dsp.denormalize_value(0.5, -10.0, 10.0) == 0.0
    assert dsp.denormalize_value(1.0, -10.0, 10.0) == 10.0
    
    # Clamping
    assert dsp.denormalize_value(1.5, -10.0, 10.0) == 10.0
    assert dsp.denormalize_value(-0.5, -10.0, 10.0) == -10.0
    
    # Normalize
    assert dsp.normalize_value(0.0, -10.0, 10.0) == 0.5
    assert dsp.normalize_value(10.0, -10.0, 10.0) == 1.0
    assert dsp.normalize_value(-10.0, -10.0, 10.0) == 0.0

def test_apply_chain_length():
    dsp = DSPEngine()
    sr = 44100
    # Generate 1s of silence
    y = np.zeros(sr, dtype=np.float32)
    
    # Empty chain
    chain = ChainConfig()
    y_out = dsp.apply_chain(y, sr, chain)
    assert len(y_out) == len(y)
    
    # Reverb chain
    chain_reverb = ChainConfig(plugins=[
        PluginConfig(name="reverb", params={"room_size": 0.5, "damping": 0.5, "wet_level": 0.2, "dry_level": 0.8})
    ])
    y_out_reverb = dsp.apply_chain(y, sr, chain_reverb)
    assert len(y_out_reverb) == len(y)
    
    # EQ + Compressor + Delay chain
    chain_full = ChainConfig(plugins=[
        PluginConfig(name="eq", params={
            "band1_freq_hz": 0.2, "band1_gain_db": 0.5,
            "band2_freq_hz": 0.5, "band2_gain_db": 0.5,
            "band3_freq_hz": 0.8, "band3_gain_db": 0.5
        }),
        PluginConfig(name="compressor", params={
            "threshold_db": 0.5, "ratio": 0.2, "attack_ms": 0.1, "release_ms": 0.2
        }),
        PluginConfig(name="delay", params={
            "delay_seconds": 0.3, "feedback": 0.2, "mix": 0.1
        })
    ])
    y_out_full = dsp.apply_chain(y, sr, chain_full)
    assert len(y_out_full) == len(y)
