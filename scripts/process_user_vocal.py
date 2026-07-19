import sys
from pathlib import Path
import librosa
import soundfile as sf
import subprocess
import shutil

def main():
    print("=== СВЕДЕНИЕ ВОКАЛА ПОЛЬЗОВАТЕЛЯ ПО РЕФЕРЕНСУ VILLIAN ===")
    
    # 1. Пути
    downloads_dir = Path("C:/Users/cwert/Downloads")
    user_dry_path = Path("D:/uniit/mixing-ai/data/чистый.wav")
    
    if not user_dry_path.exists():
        print(f"[-] Ошибка: Исходный вокал не найден по пути: {user_dry_path}")
        sys.exit(1)
        
    mp3s = list(downloads_dir.glob("*VILLIAN*.mp3"))
    if not mp3s:
        print("[-] Ошибка: Файл с подстрокой 'VILLIAN' не найден в Загрузках.")
        sys.exit(1)
        
    mp3_path = mp3s[0]
    print(f"[+] Найден файл референса: {mp3_path}")
    print(f"[+] Найден сухой вокал пользователя: {user_dry_path}")
    
    # 2. Вырезаем фрагмент 3-13 сек референсной песни во временный файл
    print("[*] Вырезаем альтернативный референс с 3.0 по 13.0 сек...")
    y_ref, sr_ref = librosa.load(mp3_path, sr=44100, offset=3.0, duration=10.0)
    
    temp_dir = Path("data/raw/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_mix_path = temp_dir / "temp_villian_mix_alt.wav"
    sf.write(temp_mix_path, y_ref, sr_ref)
    
    # 3. Выделение чистого вокала из песни-референса через Demucs
    print("[*] Запускаем Demucs для извлечения вокала-ориентира (3-13 сек) из референса...")
    demucs_out_dir = Path("data/raw/temp/demucs_out_ref_alt")
    demucs_out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd_demucs = [
        "demucs",
        "--two-stems=vocals",
        "-o", str(demucs_out_dir.absolute()),
        str(temp_mix_path.absolute())
    ]
    
    try:
        subprocess.run(cmd_demucs, check=True)
        print("[+] Вокал альтернативного референса успешно изолирован!")
    except Exception as e:
        print(f"[-] Ошибка при запуске Demucs: {e}")
        sys.exit(1)
        
    # Путь к изолированному вокалу референса
    separated_ref_vocal = demucs_out_dir / "htdemucs" / temp_mix_path.stem / "vocals.wav"
    if not separated_ref_vocal.exists():
        print(f"[-] Ошибка: Не найден выделенный вокал референса по пути {separated_ref_vocal}")
        sys.exit(1)
        
    # Копируем в папку референсов
    ref_final_path = Path("data/raw/reference/villian_ref_vocal_alt.wav")
    ref_final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(separated_ref_vocal, ref_final_path)
    print(f"[+] Изолированный вокал альтернативного референса сохранен в: {ref_final_path}")
    
    # 4. Запускаем рендеринг вокала пользователя
    output_wav = Path("data/processed/res_user_mixed_alt.wav")
    print(f"\n[*] Запуск переноса альтернативного стиля сведения:")
    print(f"  - Источник (сухой вокал пользователя): {user_dry_path}")
    print(f"  - Референс (стиль вокала из песни 3-13 сек): {ref_final_path}")
    print(f"  - Результат: {output_wav}")
    
    cmd_render = [
        "poetry", "run", "python", "scripts/render_ml_vocal.py",
        "--source", str(user_dry_path),
        "--ref", str(ref_final_path),
        "--output", str(output_wav)
    ]
    subprocess.run(cmd_render)
    print("=== ПРОЦЕСС СВЕДЕНИЯ УСПЕШНО ЗАВЕРШЕН ===")

if __name__ == "__main__":
    main()
