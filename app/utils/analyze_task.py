import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from app.audio.audio_processor import AudioProcessor

def run_analysis():
    """
    Выполняет спектральный анализ исходного и референсного вокала,
    рассчитывает разницу АЧХ (delta) со сглаживанием
    и визуализирует графики в файл data/processed/analysis_result.png.
    """
    # Инициализация процессора
    processor = AudioProcessor(sr=44100)
    
    # Определение путей к аудиофайлам
    source_path = Path("data/raw/source/source.wav")
    ref_path = Path("data/raw/reference/reference.wav")
    
    if not source_path.exists() or not ref_path.exists():
        print(f"Ошибка: Не удалось найти аудиофайлы для визуального анализа!\nDry существует: {source_path.exists()}\nRef существует: {ref_path.exists()}")
        return

    print("--- Загрузка аудиофайлов ---")
    y_source = processor.load_audio(source_path)
    y_ref = processor.load_audio(ref_path)
    
    print("--- Вычисление спектральных огибающих ---")
    env_source = processor.get_spectral_envelope(y_source)
    env_ref = processor.get_spectral_envelope(y_ref)
    
    # Разница АЧХ (сколько частот нужно прибавить или убавить)
    delta = env_ref - env_source
    
    # Сглаживание кривой дельты скользящим средним для адекватного перевода в эквалайзер
    window_size = 20
    delta_smoothed = np.convolve(delta, np.ones(window_size)/window_size, mode='same')
    
    # Частотная ось
    freqs = librosa.fft_frequencies(sr=44100, n_fft=2048)
    
    print("--- Построение и сохранение графиков ---")
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs, env_source, label='Source (Сухой вокал)')
    plt.semilogx(freqs, env_ref, label='Reference (Референс)', alpha=0.7)
    plt.title('Спектральные огибающие')
    plt.ylabel('Амплитуда (dB)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlim([20, 20000])

    plt.subplot(2, 1, 2)
    plt.semilogx(freqs, delta_smoothed, label='Delta (Целевая кривая EQ)', color='r')
    plt.title('Разностная кривая (АЧХ Source -> Reference)')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Усиление (dB)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlim([20, 20000])
    
    plt.tight_layout()
    
    # Сохранение итогового изображения
    output_plot = Path("data/processed/analysis_result.png")
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot)
    print(f"[+] График анализа спектра успешно сохранен в: {output_plot}")

if __name__ == "__main__":
    run_analysis()
