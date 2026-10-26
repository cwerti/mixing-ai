import argparse
import json
import os
import re
import sys
import shutil
import random
import tempfile
import zipfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

# Настройка кодировки консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

VOCAL_KEYWORDS = [
    "vocal", "vox", "lead", "backing", "harmony", "mic", 
    "voice", "sing", "chorus", "acapella", "dry", "bvx"
]

INSTRUMENT_KEYWORDS = [
    "kick", "snare", "drum", "hihat", "cymbal", "ride", "tom", "perc", 
    "overhead", "bass", "guitar", "gtr", "synth", "piano", "keyboard", 
    "organs", "strings", "brass", "horns", "loop", "click", "fx", "ambient",
    "shaker", "tambourine", "claps"
]

INSTRUMENTAL_GENRES = [
    "solo guitar", "solo acoustic guitar", "solo piano", "orchestral recording",
    "concerto", "string quartet", "brass band", "classical chamber concert",
    "chamber orchestra"
]

DIRECT_DOMAIN = "mtkdata.cambridgemusictechnology.co.uk"

def load_catalog(html_file="data/cambridge_page.html"):
    """
    Парсит HTML-страницу каталога Cambridge MT и возвращает структурированный список треков.
    """
    html_path = Path(html_file)
    if not html_path.exists():
        print(f"[-] Файл {html_path} не найден!")
        sys.exit(1)
        
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")
        
    artists = soup.find_all("div", class_="c-mtk__artist")
    catalog = []
    
    for artist_el in artists:
        title_bar = artist_el.find("div", class_="m-container__title-bar")
        artist_name = "Unknown Artist"
        genre = "General"
        
        if title_bar:
            h4 = title_bar.find("h4")
            if h4:
                artist_name = h4.get_text().strip()
            genre_span = title_bar.find("span", class_="m-container__title-bar-item")
            if genre_span:
                genre = genre_span.get_text().strip()
                
        tracks = artist_el.find_all("li", class_="m-mtk-track")
        for t in tracks:
            name_el = t.find("span", class_="m-mtk-track__name")
            track_name = name_el.text.strip().strip("'\"") if name_el else "Unknown Track"
            
            excerpt_dl = None
            full_dl = None
            
            for dl in t.find_all("li", class_="m-mtk-download"):
                dl_type_el = dl.find("div", class_="m-mtk-download__type")
                dl_link_el = dl.find("a", href=True)
                if not dl_link_el:
                    continue
                    
                type_str = dl_type_el.text.lower() if dl_type_el else ""
                href = dl_link_el["href"]
                size = dl_link_el.text.strip()
                
                if not href.lower().endswith(".zip"):
                    continue
                    
                dl_info = {
                    "url": href,
                    "size": size,
                    "is_direct": DIRECT_DOMAIN in href
                }
                
                if "excerpt" in type_str or "edited" in type_str:
                    excerpt_dl = dl_info
                elif "full" in type_str or "_full.zip" in href.lower():
                    full_dl = dl_info
                else:
                    if excerpt_dl is None:
                        excerpt_dl = dl_info
                        
            mixes = []
            mix_links = t.find_all("a", href=re.compile(r"\.(mp3|wav)$", re.I))
            for m in mix_links:
                mixes.append({
                    "url": m["href"],
                    "label": m.text.strip()
                })
                
            if excerpt_dl or full_dl:
                has_direct = (excerpt_dl and excerpt_dl["is_direct"]) or (full_dl and full_dl["is_direct"])
                catalog.append({
                    "artist": artist_name,
                    "genre": genre,
                    "song": track_name,
                    "excerpt": excerpt_dl,
                    "full": full_dl,
                    "mixes": mixes,
                    "is_direct": has_direct
                })
                
    return catalog

def download_file(url, dest_path):
    """
    Скачивает файл с прогресс-баром.
    """
    req = urllib.request.Request(
        url, 
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    
    with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name, leave=False) as pbar:
            while True:
                buffer = response.read(64 * 1024)
                if not buffer:
                    break
                out_file.write(buffer)
                pbar.update(len(buffer))

def extract_vocals_from_zip(zip_path, dest_dir, prefix):
    """
    Распаковывает только вокальные дорожки из ZIP-архива.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_info in z.infolist():
            if file_info.is_dir() or "__MACOSX" in file_info.filename:
                continue
                
            filename = Path(file_info.filename).name
            name_lower = filename.lower()
            
            # Игнорируем macOS метаданные и слишком маленькие файлы
            if filename.startswith("._") or file_info.file_size < 5000:
                continue
                
            if not (name_lower.endswith(".wav") or name_lower.endswith(".aif") or name_lower.endswith(".flac")):
                continue
                
            is_vocal = any(kw in name_lower for kw in VOCAL_KEYWORDS)
            is_instrument = any(kw in name_lower for kw in INSTRUMENT_KEYWORDS)
            
            if is_vocal and not is_instrument:
                safe_name = f"{prefix}_{filename}"
                out_file_path = dest_dir / safe_name
                
                with z.open(file_info) as source, open(out_file_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
                extracted_files.append(out_file_path)
                
    return extracted_files

def select_diverse_tracks(tracks, target_count):
    """
    Выбирает target_count треков, сбалансированных по разным жанрам.
    """
    genre_groups = defaultdict(list)
    for t in tracks:
        genre_groups[t["genre"]].append(t)
        
    # Перемешиваем треки внутри каждого жанра
    rng = random.Random(42) # Фиксированный seed для воспроизводимости
    for g in genre_groups:
        rng.shuffle(genre_groups[g])
        
    selected = []
    genres_list = list(genre_groups.keys())
    rng.shuffle(genres_list)
    
    # Круговой выбор (Round-Robin) по 1 треку из каждого жанра
    while len(selected) < target_count:
        added_in_round = 0
        for g in genres_list:
            if genre_groups[g]:
                selected.append(genre_groups[g].pop(0))
                added_in_round += 1
                if len(selected) >= target_count:
                    break
        if added_in_round == 0:
            break
            
    return selected

def main():
    parser = argparse.ArgumentParser(description="Парсер и загрузчик мультитреков с Cambridge MT")
    parser.add_argument("--list", action="store_true", help="Вывести список доступных треков")
    parser.add_argument("--all", action="store_true", help="Показать весь каталог при --list")
    parser.add_argument("--search", type=str, default=None, help="Поиск по исполнителю или песне")
    parser.add_argument("--genre", type=str, default=None, help="Фильтрация по жанру (Pop, Rock, Metal, Acoustic и т.д.)")
    parser.add_argument("--num-projects", "-n", type=int, default=200, help="Сколько проектов скачать (по умолчанию 200)")
    parser.add_argument("--mode", choices=["excerpt", "full"], default="excerpt", 
                        help="Режим: 'excerpt' (нарезка ~25-45 МБ) или 'full' (полный трек ~200-500 МБ)")
    parser.add_argument("--output-dir", type=str, default="data/raw/dry_vocals/cambridge", 
                        help="Папка для сохранения изолированных вокалов")
    parser.add_argument("--diverse", action="store_true", default=True, 
                        help="Балансировать выборку по разнообразию жанров")
    
    args = parser.parse_args()
    
    catalog = load_catalog()
    
    # Оставляем только прямые доступные треки
    available = [t for t in catalog if t["is_direct"]]
    
    # Фильтруем чисто инструментальные жанры
    vocal_ready = [
        t for t in available 
        if not any(ig in t["genre"].lower() for ig in INSTRUMENTAL_GENRES)
    ]
    
    print(f"[*] Доступно для мгновенной загрузки: {len(vocal_ready)} вокальных треков (из {len(catalog)} всего).")
    
    if args.genre:
        g = args.genre.lower()
        vocal_ready = [t for t in vocal_ready if g in t["genre"].lower()]
        print(f"[*] Отфильтровано по жанру '{args.genre}': {len(vocal_ready)} треков.")
        
    if args.search:
        q = args.search.lower()
        vocal_ready = [t for t in vocal_ready if q in t["artist"].lower() or q in t["song"].lower()]
        print(f"[*] Найдено по запросу '{args.search}': {len(vocal_ready)} треков.")
        
    if args.list:
        limit = len(vocal_ready) if args.all else (args.num_projects if args.num_projects != 200 else 30)
        print(f"\n--- СПИСОК ДОСТУПНЫХ ТРЕКОВ (показано {min(limit, len(vocal_ready))} из {len(vocal_ready)}) ---")
        for idx, item in enumerate(vocal_ready[:limit], 1):
            ex_size = item["excerpt"]["size"] if item["excerpt"] else "—"
            fu_size = item["full"]["size"] if item["full"] else "—"
            print(f"{idx:3d}. [{item['genre']}] {item['artist']} — '{item['song']}' [Excerpt: {ex_size}, Full: {fu_size}]")
        return
        
    # Выборка треков
    target_count = min(args.num_projects, len(vocal_ready))
    if args.diverse and not args.genre and not args.search:
        selected = select_diverse_tracks(vocal_ready, target_count)
    else:
        selected = vocal_ready[:target_count]
        
    # Подсчет разнообразия жанров
    genres_count = len(set(t["genre"] for t in selected))
    print(f"\n[*] Отобрано проектов для загрузки: {len(selected)} (охвачено уникальных жанров: {genres_count})")
    
    dest_dir = Path(args.output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.gettempdir()) / "cambridge_dl"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    total_vocals_count = 0
    skipped_count = 0
    
    for i, item in enumerate(selected, 1):
        target_dl = item["excerpt"] if args.mode == "excerpt" and item["excerpt"] and item["excerpt"]["is_direct"] else item["full"]
        if not target_dl or not target_dl["is_direct"]:
            target_dl = item["full"] if item["full"] and item["full"]["is_direct"] else item["excerpt"]
            
        if not target_dl or not target_dl["is_direct"]:
            continue
            
        url = target_dl["url"]
        size_str = target_dl["size"]
        prefix = re.sub(r'[^a-zA-Z0-9_]', '', f"{item['artist']}_{item['song']}".replace(" ", "_"))[:30]
        
        # Проверяем, не скачаны ли уже вокалы для этого трека
        existing = list(dest_dir.glob(f"{prefix}_*"))
        if existing:
            print(f"[{i}/{len(selected)}] [Пропуск] {item['artist']} — '{item['song']}' (уже извлечено файлов: {len(existing)})")
            total_vocals_count += len(existing)
            skipped_count += 1
            continue
            
        print(f"\n[{i}/{len(selected)}] [{item['genre']}] {item['artist']} — '{item['song']}' ({size_str})...")
        zip_path = temp_dir / f"temp_{i}.zip"
        
        try:
            download_file(url, zip_path)
            vocals = extract_vocals_from_zip(zip_path, dest_dir, prefix)
            
            if vocals:
                print(f"    [+] Извлечено вокалов: {len(vocals)} (например: {vocals[0].name})")
                total_vocals_count += len(vocals)
            else:
                print(f"    [-] Вокальных дорожек в архиве не обнаружено.")
                
            if zip_path.exists():
                zip_path.unlink()
                
        except Exception as e:
            print(f"    [-] Ошибка скачивания '{item['song']}': {e}")
            if zip_path.exists():
                zip_path.unlink()
                
    # Очистка временной папки
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
        
    print(f"\n==========================================")
    print(f"[+] ЗАГРУЗКА 200 ПРОЕКТОВ УСПЕШНО ЗАВЕРШЕНА!")
    print(f"[+] Всего чистых вокальных дорожек в папке: {total_vocals_count}")
    print(f"[+] Пропущено ранее скачанных: {skipped_count}")
    print(f"[+] Папка с вокалом: {dest_dir.absolute()}")

if __name__ == "__main__":
    main()
