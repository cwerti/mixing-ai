import pytest
import math
from pathlib import Path
from app.bridge.vst_parameter_mapper import VSTParameterMapper

def test_eq_linear_mapping():
    mapper = VSTParameterMapper("fruity_parametric_eq_2")
    assert mapper.get_param_index("band1_gain") == 1
    
    # 0 dB should map to 0.5 (middle of -18 to 18 dB)
    norm = mapper.map_physical_to_normalized("band1_gain", 0.0)
    assert math.isclose(norm, 0.5, abs_tol=1e-5)
    
    phys = mapper.map_normalized_to_physical("band1_gain", 0.5)
    assert math.isclose(phys, 0.0, abs_tol=1e-5)

    # -18 dB should map to 0.0
    assert math.isclose(mapper.map_physical_to_normalized("band1_gain", -18.0), 0.0, abs_tol=1e-5)
    
    # 18 dB should map to 1.0
    assert math.isclose(mapper.map_physical_to_normalized("band1_gain", 18.0), 1.0, abs_tol=1e-5)

def test_eq_logarithmic_mapping():
    mapper = VSTParameterMapper("fruity_parametric_eq_2")
    assert mapper.get_param_index("band1_freq") == 0
    
    # Logarithmic mapping of 10Hz to 20000Hz
    # Min = 10, Max = 20000
    # Center frequency in log scale: sqrt(10 * 20000) = sqrt(200000) ~ 447.21Hz should be 0.5
    center_freq = math.sqrt(10.0 * 20000.0)
    norm = mapper.map_physical_to_normalized("band1_freq", center_freq)
    assert math.isclose(norm, 0.5, abs_tol=1e-5)

    phys = mapper.map_normalized_to_physical("band1_freq", 0.5)
    assert math.isclose(phys, center_freq, abs_tol=1e-2)

def test_clamping_and_boundaries():
    mapper = VSTParameterMapper("fruity_parametric_eq_2")
    # Out of range physical values should clamp
    assert math.isclose(mapper.map_physical_to_normalized("band1_gain", 30.0), 1.0, abs_tol=1e-5)
    assert math.isclose(mapper.map_physical_to_normalized("band1_gain", -30.0), 0.0, abs_tol=1e-5)

    # Out of range normalized values should clamp
    assert math.isclose(mapper.map_normalized_to_physical("band1_gain", 1.5), 18.0, abs_tol=1e-5)
    assert math.isclose(mapper.map_normalized_to_physical("band1_gain", -0.5), -18.0, abs_tol=1e-5)

def test_vst3_mapping_fabfilter():
    mapper = VSTParameterMapper("fabfilter_pro_q_3")
    assert mapper.get_param_index("band1_freq") == 2
    assert mapper.get_param_index("band1_gain") == 3

    # Min = 10.0, Max = 30000.0
    center_freq = math.sqrt(10.0 * 30000.0)
    assert math.isclose(mapper.map_physical_to_normalized("band1_freq", center_freq), 0.5, abs_tol=1e-5)
