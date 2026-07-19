import numpy as np
import time
import librosa
from pathlib import Path
from app.audio.audio_processor import AudioProcessor
from app.bridge.bridge_client import BridgeClient
from app.bridge.plugin_manager import PluginLoader

class MixingAIPipeline:
    """
    Класс сквозного выполнения (E2E) пайплайна сведения.
    Координирует загрузку и анализ звука, подключение к FL Studio,
    выбор свободного трека микшера, загрузку нужных плагинов и перенос параметров.
    """
    
    def __init__(self, port_name: str = 'Mixing Ai'):
        """
        Инициализирует сквозной пайплайн сведения.
        
        Args:
            port_name: Имя виртуального MIDI-порта для управления FL Studio.
        """
        self.processor = AudioProcessor()
        self.client = BridgeClient(port_name=port_name)
        self.loader = PluginLoader(self.client)

    def run(self, source_path: str = "data/raw/source/source.wav", ref_path: str = "data/raw/reference/reference.wav") -> bool:
        """
        Запускает полный E2E процесс сведения вокала.
        
        Args:
            source_path: Путь к исходному (сухому) файлу вокала.
            ref_path: Путь к референсному файлу вокала.
            
        Returns:
            bool: True в случае успешного выполнения, иначе False.
        """
        print("\n=== Mixing-AI Интегрированный Пайплайн ===")
        
        # 1. Загрузка и анализ аудиофайлов
        src_file = Path(source_path)
        ref_file = Path(ref_path)
        
        if src_file.exists() and ref_file.exists():
            print(f"[*] Обработка аудиофайлов:\n  Source: {src_file}\n  Reference: {ref_file}")
            
            # Акустический анализ
            y_src = self.processor.load_audio(src_file)
            y_ref = self.processor.load_audio(ref_file)
            
            env_src = self.processor.get_spectral_envelope(y_src)
            env_ref = self.processor.get_spectral_envelope(y_ref)
            delta = env_ref - env_src
            freqs = librosa.fft_frequencies(sr=self.processor.sr, n_fft=2048)
            
            # Извлечение спектральных характеристик вокала
            features_src = self.processor.extract_features(y_src)
            features_ref = self.processor.extract_features(y_ref)
            print(f"[*] Source пик-фактор: {features_src['crest_factor']:.2f}, динамический диапазон: {features_src['dynamic_range_db']:.2f} dB")
            print(f"[*] Reference пик-фактор: {features_ref['crest_factor']:.2f}, динамический диапазон: {features_ref['dynamic_range_db']:.2f} dB")
        else:
            print("[-] Исходный или референсный файл не найден. Использование тестовых случайных данных.")
            freqs = np.linspace(20, 20000, 1024)
            delta = np.random.uniform(-5, 5, 1024)

        # 2. Подключение к MIDI-порту
        if not self.client.connect_midi():
            print("[-] Не удалось подключиться по MIDI. Проверьте, активен ли виртуальный порт и запущена ли FL Studio.")
            return False

        # 3. Подготовка канала микшера
        if not self.client.prepare_empty_track():
            print("[-] Не удалось подготовить пустой канал микшера.")
            return False

        # 4. Автоматическая загрузка цепочки эффектов в FL Studio
        chain = ["Fruity Parametric EQ 2", "Fruity Limiter"]
        for plugin in chain:
            if not self.loader.load(plugin):
                print(f"[-] Не удалось загрузить плагин: {plugin}")
            time.sleep(2.0)

        # 5. Ожидание полной инициализации интерфейсов плагинов
        print("[*] Ожидание инициализации плагинов в FL Studio...")
        time.sleep(2.0) 

        # 6. Отправка и применение параметров АЧХ на эквалайзер
        print("[*] Применение настроек полос эквалайзера...")
        if self.client.set_eq_params(freqs, delta, "Fruity Parametric EQ 2"):
            print("[+] УСПЕХ: Настройки эквалайзера и вокальная цепочка применены в FL Studio.")
            return True
        else:
            print("[-] Не удалось применить параметры эквалайзера.")
            return False

if __name__ == "__main__":
    pipeline = MixingAIPipeline()
    pipeline.run()
