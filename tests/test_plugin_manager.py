import pytest
from app.core.plugin_manager import PLUGIN_DB, PluginLoader

def test_plugin_db_structure():
    # Check that required plugins exist in the DB
    required_plugins = [
        "Fruity Parametric EQ 2",
        "Fruity Limiter",
        "Fruity Compressor",
        "Fruity Blood Overdrive",
        "Soundgoodizer",
        "Fruity Reeverb 2",
        "Fruity Delay 3",
        "Fruity Stereo Enhancer"
    ]
    
    for p in required_plugins:
        assert p in PLUGIN_DB
        plugin_info = PLUGIN_DB[p]
        assert plugin_info.name == p
        assert plugin_info.param_count > 0
        assert isinstance(plugin_info.params, dict)

def test_eq2_params():
    eq = PLUGIN_DB["Fruity Parametric EQ 2"]
    # Check that all 7 bands exist and have parameters (Freq, Gain, BW, Type, Order)
    for band in range(7):
        base = band * 5
        # Freq
        assert base in eq.params
        assert eq.params[base].name == f"Band {band+1} Freq"
        # Gain
        assert base+1 in eq.params
        assert eq.params[base+1].name == f"Band {band+1} Gain"
        assert eq.params[base+1].default == 0.5
        # BW
        assert base+2 in eq.params
        assert eq.params[base+2].name == f"Band {band+1} BW"
        # Type
        assert base+3 in eq.params
        # Order
        assert base+4 in eq.params

def test_plugin_loader_mock():
    class DummyController:
        def __init__(self):
            self.shell = None
            self.focused = False
            
        def focus_fl(self):
            self.focused = True
            
        def press_key(self, key_code):
            pass
            
    controller = DummyController()
    loader = PluginLoader(controller)
    
    # Since controller.shell is None, it should return False
    assert loader.load("Fruity Limiter") is False
    assert controller.focused is True
