import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from app.core.audio_processor import AudioProcessor

def run_analysis():
    # Инициализация процессора
    processor = AudioProcessor(sr=44100)
    
    # Пути к файлам (учитывая опечатку sourse.wav)
    source_path = Path("data/raw/source/sourse.wav")
    ref_path = Path("data/raw/reference/reference.wav")
    
    if not source_path.exists() or not ref_path.exists():
        print(f"Ошибка: Файлы не найдены!\nSource: {source_path.exists()}\nRef: {ref_path.exists()}")
        return

    print("--- Загрузка аудио ---")
    y_source = processor.load_audio(source_path)
    y_ref = processor.load_audio(ref_path)
    
    print("--- Анализ спектра ---")
    env_source = processor.get_spectral_envelope(y_source)
    env_ref = processor.get_spectral_envelope(y_ref)
    
    # Вычисляем дельту (разницу АЧХ)
    # env_ref - env_source даст нам кривую, которую нужно применить к source
    delta = env_ref - env_source
    
    # Сглаживание дельты для адекватного маппинга на EQ
    # Используем скользящее среднее
    window_size = 20
    delta_smoothed = np.convolve(delta, np.ones(window_size)/window_size, mode='same')
    
    # Частотная ось
    freqs = librosa.fft_frequencies(sr=44100, n_fft=2048)
    
    print("--- Визуализация ---")
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs, env_source, label='Source (Dry)')
    plt.semilogx(freqs, env_ref, label='Reference (Wet)', alpha=0.7)
    plt.title('Spectral Envelopes')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.subplot(2, 1, 1).set_xlim([20, 20000])

    plt.subplot(2, 1, 2)
    plt.semilogx(freqs, delta_smoothed, label='Delta (Target EQ Curve)', color='r')
    plt.title('Delta Curve (Source -> Reference)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlim([20, 20000])
    
    plt.tight_layout()
    
    # Сохраняем результат
    output_plot = Path("data/processed/analysis_result.png")
    plt.savefig(output_plot)
    print(f"Анализ завершен. График сохранен в: {output_plot}")

if __name__ == "__main__":
    run_analysis()
