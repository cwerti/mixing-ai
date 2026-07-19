import os
import shutil
import zipfile
import tempfile
from pathlib import Path

def process_downloaded_zips(downloads_dir="C:/Users/cwert/Downloads", dest_dir="data/raw/dry_vocals/vctk"):
    """
    Сканирует Downloads на наличие файлов *_Full.zip, распаковывает их,
    находит чистые лид-вокальные WAV-дорожки, копирует их в датасет
    и удаляет ZIP-архивы для очистки места.
    """
    downloads_path = Path(downloads_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if not downloads_path.exists():
        print(f"[-] Директория загрузок не найдена: {downloads_path}")
        return

    # Поиск всех архивов мультитреков
    zip_files = list(downloads_path.glob("*_Full.zip"))
    print(f"[*] Найдено архивов мультитреков для обработки: {len(zip_files)}")
    
    vocal_keywords = ["vox", "vocal", "lead", "sing", "vox_lead", "lead_vocal", "solo", "ldvx", "lv"]
    backing_keywords = ["back", "bg", "double", "choir", "harm", "dub", "group", "tuned"]

    for zip_path in zip_files:
        song_name = zip_path.stem.replace("_Full", "")
        print(f"\n[*] Обработка '{song_name}' ({zip_path.name})...")
        
        # Временная папка для распаковки
        temp_dir = Path(tempfile.gettempdir()) / f"extract_{song_name}"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Распаковка архива
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except Exception as e:
            print(f"  [-] Ошибка при распаковке {zip_path.name}: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue

        # 2. Поиск и перенос сухих вокалов
        found_vocals = []
        for file in temp_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in [".wav", ".flac"]:
                name_lower = file.name.lower()
                
                # Проверка: является ли файл вокалом и не является ли он бэком/хором
                is_vocal = any(kw in name_lower for kw in vocal_keywords)
                is_backing = any(kw in name_lower for kw in backing_keywords)
                
                if is_vocal and not is_backing:
                    out_name = f"cambridge_{song_name}_{file.name}"
                    target_file = dest_path / out_name
                    shutil.copy2(file, target_file)
                    found_vocals.append(file.name)
                    
        # 3. Отчет о результатах
        if found_vocals:
            print(f"  [SUCCESS] Скопировано вокальных файлов ({len(found_vocals)} шт.):")
            for f in found_vocals:
                print(f"    - {f}")
        else:
            print("  [-] Предупреждение: Вокальные файлы не были найдены автоматически.")

        # 4. Очистка временных файлов и удаление исходного ZIP
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            zip_path.unlink()
            print(f"  [+] Исходный архив {zip_path.name} успешно удален из загрузок.")
        except Exception as e:
            print(f"  [-] Не удалось удалить {zip_path.name}: {e}")

    print(f"\n[+] Вся обработка успешно завершена! Файлы сухого вокала перенесены в: {dest_path}")

if __name__ == "__main__":
    process_downloaded_zips()
