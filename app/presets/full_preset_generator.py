from pathlib import Path

from app.fl.events import read_fl_file
from app.fl.plugins import eq2


class FullPresetGenerator:
    """
    Сборка пресета FL Studio (.fst) из шаблона: в блоб Fruity Parametric EQ 2 шаблона записываются частоты
    и усиления полос (раскладка — app/fl/plugins/eq2.py). Длина блоба не меняется, поэтому он заменяется
    в файле на месте. Остальные плагины цепочки пока не экспортируются.
    """

    # Ширина peak-полос как у DSPEngine (PeakFilter q=1.0), чтобы пресет в FL звучал как превью
    PEAK_Q = 1.0

    def __init__(self, template_path: str = "FlProject/template.fst"):
        """
        Args:
            template_path: Пресет-шаблон с одним Fruity Parametric EQ 2 (полки на 1-й и 7-й полосах).
        """
        self.template_path = Path(template_path)

    def generate_vocal_chain(self, bands: list, output_name: str = "mixing_ai_vocal_track.fst",
                             output_dir: str | Path = "data/processed") -> Path:
        """
        Записывает полосы EQ в копию шаблона.

        Args:
            bands: Список полос [(freq_hz, gain_db), ...] в порядке полос EQ 2 (до 7).
            output_name: Имя выходного файла пресета.
            output_dir: Папка для результата.

        Returns:
            Path: Путь к сохраненному пресету.

        Raises:
            FileNotFoundError: Нет шаблона.
            ValueError: В шаблоне нет данных Fruity Parametric EQ 2.
        """
        if not self.template_path.exists():
            raise FileNotFoundError(f"Файл шаблона канала не найден по пути: {self.template_path}")
        raw = self.template_path.read_bytes()
        blobs = [e.data for e in read_fl_file(self.template_path).events if e.id == 213]
        if not blobs:
            raise ValueError(f"В шаблоне {self.template_path} нет данных Fruity Parametric EQ 2")
        old = blobs[0]
        new = eq2.encode_bands(old, [(float(f), float(g)) for f, g in bands], peak_q=self.PEAK_Q)
        pos = raw.find(old)
        output_path = Path(output_dir) / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw[:pos] + new + raw[pos + len(old):])
        return output_path
