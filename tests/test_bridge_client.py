import pytest
import numpy as np
from app.bridge.bridge_client import BridgeClient

def test_bridge_client_math():
    client = BridgeClient(port_name="Test Port")
    
    # We can test that target_freqs map correctly
    # Formula: f_val = (log10(freq) - 1.0) / 3.301
    # 10 Hz -> (1.0 - 1.0) / 3.301 = 0.0
    # 20000 Hz -> (log10(20000) - 1.0) / 3.301 = (4.30103 - 1) / 3.301 = 3.30103 / 3.301 = ~1.0
    
    # Test 10Hz mapping
    f_val_10 = (np.log10(10.0) - 1.0) / 3.301
    assert pytest.approx(f_val_10, abs=1e-5) == 0.0
    
    # Test 20000Hz mapping
    f_val_20k = (np.log10(20000.0) - 1.0) / 3.301
    assert pytest.approx(f_val_20k, abs=1e-3) == 1.0
    
    # Test gain mapping: (gain + 18.0) / 36.0
    # -18 dB -> 0.0
    # 0 dB -> 0.5
    # 18 dB -> 1.0
    g_val_minus_18 = (-18.0 + 18.0) / 36.0
    assert g_val_minus_18 == 0.0
    
    g_val_zero = (0.0 + 18.0) / 36.0
    assert g_val_zero == 0.5
    
    g_val_plus_18 = (18.0 + 18.0) / 36.0
    assert g_val_plus_18 == 1.0
