import pyflp
# Совместимость с Python 3.12+ (исправление пустых Enum в PyFLP)
pyflp.EventEnum._member_names_ = ['DUMMY']
pyflp.EventEnum._member_map_ = {'DUMMY': 999}
import sys
from pathlib import Path

def inspect_fst(file_path: str | Path):
    """
    Инспектирует базовую структуру и метаданные бинарного файла .fst с помощью PyFLP.
    Выводит тип плагина и количество событий внутри файла.
    
    Args:
        file_path: Путь к файлу пресета (.fst).
    """
    try:
        preset = pyflp.parse(str(file_path))
        print(f"Тип объекта пресета: {type(preset)}")
        
        # Для .fst файлов это обычно объект состояния плагина (PluginState)
        if hasattr(preset, 'plugin'):
             print(f"Имя плагина: {preset.plugin.name}")
        
        # Определение количества внутренних параметров/событий FL Studio
        has_evs = hasattr(preset, 'events')
        print(f"Количество событий: {len(preset.events) if has_evs else 'Отсутствует'}")
        
    except Exception as e:
        print(f"Ошибка при инспектировании файла пресета: {e}")

if __name__ == "__main__":
    inspect_fst("FlProject/template.fst")
