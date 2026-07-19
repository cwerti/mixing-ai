import os
import sys
import zipfile
import urllib.request
import tempfile
import shutil
from pathlib import Path

# Список популярных качественных мультитреков с сухим вокалом на Cambridge MT
DEFAULT_URLS = [
    "https://multitracks.cambridge-mt.com/SwissPopLory_Full.zip",        # Swiss Pop Lory (Поп-вокал)
    "https://multitracks.cambridge-mt.com/JonasSvennemar_Full.zip",     # Jonas Svennemar (Мужской вокал)
    "https://multitracks.cambridge-mt.com/Lushlife_Full.zip",           # Lushlife (Рэп/Хип-хоп вокал)
    "https://multitracks.cambridge-mt.com/BlueDot_Full.zip",             # Blue Dot (Женский инди-вокал)
    "https://multitracks.cambridge-mt.com/SignsOfOne_Full.zip"          # Signs Of One (Поп-рок вокал)
]

def download_and_extract_vocals(urls=None, dest_dir="data/raw/dry_vocals/vctk"):
    """
    Скачивает ZIP-архивы мультитреков с Cambridge MT, находит в них 
    сухие дорожки вокала, копирует их в dest_dir и удаляет остальной инструментал.
    """
    if urls is None:
        urls = DEFAULT_URLS
        
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Использование временной папки ОС для распаковки гигабайтных инструменталов
    temp_dir = Path(tempfile.gettempdir()) / "cambridge_temp"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        song_name = url.split("/")[-1].replace("_Full.zip", "")
        print(f"\n[*] Обработка песни '{song_name}'...")
        
        zip_path = temp_dir / f"{song_name}.zip"
        temp_extract = temp_dir / song_name
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Скачивание ZIP-архива
        try:
            print(f"  [+] Скачивание архива ({url})...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            print(f"  [-] Ошибка при скачивании {url}: {e}")
            continue
            
        # 2. Распаковка архива
        try:
            print("  [+] Распаковка архива...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract)
        except Exception as e:
            print(f"  [-] Ошибка распаковки: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue
            
        # 3. Поиск вокальных дорожек
        # Ключевые слова для поиска вокала в именах файлов
        vocal_keywords = ["vox", "vocal", "lead", "sing", "vox_lead", "lead_vocal", "solo"]
        found_any = False
        
        # Обход всех распакованных файлов
        for file in temp_extract.rglob("*"):
            if file.is_file() and file.suffix.lower() in [".wav", ".flac"]:
                name_lower = file.name.lower()
                
                # Исключаем бэк-вокалы (backing, double, bg), чтобы брать только основной сухой лид
                is_backing = any(kw in name_lower for kw in ["back", "bg", "double", "choir", "harm", "dub"])
                is_vocal = any(kw in name_lower for kw in vocal_keywords)
                
                if is_vocal and not is_backing:
                    # Копируем чистый вокальный трек
                    out_name = f"cambridge_{song_name}_{file.name}"
                    target_file = dest_path / out_name
                    
                    shutil.copy2(file, target_file)
                    print(f"  [SUCCESS] Найден и скопирован сухой вокал: {file.name} -> {out_name}")
                    found_any = True
                    
        if not found_any:
            print("  [-] Предупреждение: Не удалось автоматически найти сухой лид-вокал в архиве.")
            
        # Очистка временных файлов для экономии места на диске
        print("  [+] Очистка временных файлов...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n[+] Сбор сухих мультитрековых вокалов завершен! Файлы сохранены в: {dest_path}")

if __name__ == "__main__":
    download_and_extract_vocals()
