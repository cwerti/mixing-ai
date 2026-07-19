import pytest
import numpy as np
from app.audio.dsp_engine import DSPEngine
from app.dataset.dataset_schema import ChainConfig, PluginConfig

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
            "band1_freq_hz": 0.1, "band1_gain_db": 0.5,
            "band2_freq_hz": 0.2, "band2_gain_db": 0.5,
            "band3_freq_hz": 0.3, "band3_gain_db": 0.5,
            "band4_freq_hz": 0.5, "band4_gain_db": 0.5,
            "band5_freq_hz": 0.6, "band5_gain_db": 0.5,
            "band6_freq_hz": 0.8, "band6_gain_db": 0.5,
            "band7_freq_hz": 0.9, "band7_gain_db": 0.5
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

def test_vst_fallback_chain():
    dsp = DSPEngine()
    sr = 44100
    y = np.zeros(sr, dtype=np.float32)
    
    # Chain with VST3 plugins that will trigger fallback
    chain = ChainConfig(plugins=[
        PluginConfig(name="fabfilter_pro_q_3", params={
            "band1_freq": 0.2, "band1_gain": 0.5,
            "band2_freq": 0.5, "band2_gain": 0.5
        }),
        PluginConfig(name="fabfilter_pro_c_2", params={
            "threshold": 0.5, "ratio": 0.2
        }),
        PluginConfig(name="valhalla_vintage_verb", params={
            "mix": 0.15, "decay": 0.5
        })
    ])
    
    y_out = dsp.apply_chain(y, sr, chain)
    assert len(y_out) == len(y)

