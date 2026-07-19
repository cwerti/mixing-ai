import sys
from pathlib import Path
import librosa
import soundfile as sf
import subprocess
import shutil

def main():
    print("=== ОБРЕЗКА, РАЗДЕЛЕНИЕ DEMUCS И ТЕСТИРОВАНИЕ ===")
    
    # 1. Поиск файла в Загрузках
    downloads_dir = Path("C:/Users/cwert/Downloads")
    mp3s = list(downloads_dir.glob("*VILLIAN*.mp3"))
    if not mp3s:
        print("[-] Ошибка: Файл с подстрокой 'VILLIAN' не найден в Загрузках.")
        sys.exit(1)
        
    mp3_path = mp3s[0]
    print(f"[+] Найден файл: {mp3_path}")
    
    # 2. Вырезаем фрагмент 13-18 сек во временный файл
    print("[*] Вырезаем аудио с 13.0 по 18.0 сек...")
    y, sr = librosa.load(mp3_path, sr=44100, offset=13.0, duration=5.0)
    
    temp_dir = Path("data/raw/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_mix_path = temp_dir / "temp_mix.wav"
    sf.write(temp_mix_path, y, sr)
    print(f"[+] Вырезанный микс сохранен во временный файл: {temp_mix_path}")
    
    # 3. Выделение вокала через Demucs
    print("[*] Запуск Demucs для отделения вокала от бита...")
    demucs_out_dir = Path("data/raw/temp/demucs_out")
    demucs_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Команда разделения
    cmd_demucs = [
        "demucs",
        "--two-stems=vocals",
        "-o", str(demucs_out_dir.absolute()),
        str(temp_mix_path.absolute())
    ]
    try:
        subprocess.run(cmd_demucs, check=True)
        print("[+] Вокал успешно отделен!")
    except Exception as e:
        print(f"[-] Ошибка при запуске Demucs: {e}")
        print("[!] Убедитесь, что Demucs установлен в вашей системе.")
        sys.exit(1)
        
    # Путь к извлеченному вокалу
    # По умолчанию Demucs создает в папке вывода структуру: htdemucs / <имя_файла> / vocals.wav
    separated_vocal = demucs_out_dir / "htdemucs" / temp_mix_path.stem / "vocals.wav"
    if not separated_vocal.exists():
        print(f"[-] Ошибка: Не найден выделенный вокал по пути {separated_vocal}")
        sys.exit(1)
        
    # 4. Финальные пути
    source_out = Path("data/raw/source/cropped_villian_dry.wav")
    ref_out = Path("data/raw/reference/cropped_villian_mix.wav")
    source_out.parent.mkdir(parents=True, exist_ok=True)
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    
    # Копируем файлы на свои места
    shutil.copy(separated_vocal, source_out)
    shutil.copy(temp_mix_path, ref_out)
    
    print(f"[+] Файлы подготовлены:")
    print(f"  - Сухой вокал (source): {source_out}")
    print(f"  - Референс микс (ref): {ref_out}")
    
    # 5. Запуск рендеринга
    output_wav = Path("data/processed/res_villian.wav")
    print("\n[*] Запуск модели для переноса стиля вокала из песни...")
    cmd_render = [
        "poetry", "run", "python", "scripts/render_ml_vocal.py",
        "--source", str(source_out),
        "--ref", str(ref_out),
        "--output", str(output_wav)
    ]
    subprocess.run(cmd_render)
    print("=== ОБРАБОТКА ЗАВЕРШЕНА ===")

if __name__ == "__main__":
    main()
