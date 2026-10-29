from pathlib import Path

from app.fl.events import FLParseError, read_fl_file


def parse_fl_events(file_path: str | Path) -> list:
    """
    Парсит поток событий формата Image-Line (FL Studio Event stream) из бинарного файла (.flp / .fst).
    Позволяет извлечь параметры автоматизации и состояния загруженных VST-плагинов.

    Совместимая обёртка над ``app.fl.events.read_fl_file`` (раньше здесь были неверные
    диапазоны размеров событий). Новый код должен использовать ``app.fl`` напрямую.

    Args:
        file_path: Путь к файлу пресета (.fst) или проекта (.flp).

    Returns:
        list: Список кортежей (event_id, value), содержащих события FL;
        пустой список, если файл не является корректным потоком событий FL.
    """
    try:
        fl_file = read_fl_file(file_path)
    except FLParseError:
        return []
    return [(event.id, event.value) for event in fl_file.events]


if __name__ == "__main__":
    # Локальный тест на файле шаблона
    template_file = "FlProject/template.fst"
    if Path(template_file).exists():
        evs = parse_fl_events(template_file)
        # Данные (состояние) плагина в .fst/.flp передаются событием 213 (Plugin Data)
        for eid, val in evs:
            if eid == 213:
                print(f"Обнаружен блок Plugin Data (Размер: {len(val)} байт)")
                print(val.hex()[:100] + "...")
            else:
                print(f"ID события: {eid:3} | Значение: {val}")
    else:
        print(f"Шаблон {template_file} не найден для локального теста.")
