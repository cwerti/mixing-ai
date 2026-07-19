import pytest
import sys
from pathlib import Path

# Добавляем корень проекта в пути импорта
sys.path.append(str(Path(__file__).parent.parent.absolute()))

def test_gui_imports():
    """Проверяет корректность импорта классов графического интерфейса."""
    from app.gui.app_window import EQVisualizer, DropZoneWidget, DragOutWidget, MixingAIApp
    assert EQVisualizer is not None
    assert DropZoneWidget is not None
    assert DragOutWidget is not None
    assert MixingAIApp is not None
