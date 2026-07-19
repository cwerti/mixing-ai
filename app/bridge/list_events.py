import pyflp
# Совместимость с Python 3.12+ (исправление пустых Enum в PyFLP)
pyflp.EventEnum._member_names_ = ['DUMMY']
pyflp.EventEnum._member_map_ = {'DUMMY': 999}
from pathlib import Path

def list_events(file_path: str | Path):
    """
    Печатает список внутренних событий FL Studio из файла пресета с помощью PyFLP.
    
    Args:
        file_path: Путь к файлу пресета (.fst).
    """
    try:
        preset = pyflp.parse(str(file_path))
        print(f"Тип объекта пресета: {type(preset)}")
        
        # В PyFLP 2.x события проекта/пресета доступны через свойство events
        if hasattr(preset, 'events'):
            print(f"Всего событий обнаружено: {len(preset.events)}")
            for i, event in enumerate(preset.events):
                try:
                    print(f"Событие {i}: ID={event.id}, Value={event.value}")
                except Exception as ex:
                    print(f"Событие {i}: {event} (Ошибка получения атрибутов: {ex})")
        else:
            print("У объекта пресета отсутствует атрибут 'events'.")
            
    except Exception as e:
        print(f"Ошибка при чтении событий пресета: {e}")

if __name__ == "__main__":
    template_file = "FlProject/template.fst"
    if Path(template_file).exists():
        list_events(template_file)
    else:
        print(f"Файл {template_file} не найден.")
