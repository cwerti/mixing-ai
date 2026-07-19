import zipfile
from pathlib import Path

def main():
    downloads_dir = Path("C:/Users/cwert/Downloads")
    target_dir = Path("D:/uniit/mixing-ai/data/raw/cambridge_raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_names = [
        "AMContra_HeartPeripheral_Full.zip",
        "AnaMady_PeleEstranha_Full.zip",
        "BalazsDanielBoogieWoogieTrio_OwnWayToBoogie_Full.zip",
        "BenFlowers_Retry_Full.zip",
        "MR0903_Moosmusic_Full.zip",
        "TrickBird_Window_Full.zip",
        "TytillidieXXollin_Bankroll_Full.zip",
        "TytillidieXXollin_Beauty_Full.zip",
        "TytillidieXXollin_Teleport_Full.zip"
    ]

    print("=== РАСПАКОВКА НОВЫХ ПРОЕКТОВ ИЗ ЗАГРУЗОК ===")
    for name in zip_names:
        zip_path = downloads_dir / name
        if zip_path.exists():
            print(f"[*] Распаковка {name}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Распаковываем в отдельную папку по названию проекта
                    dest_sub_dir = target_dir / zip_path.stem
                    dest_sub_dir.mkdir(parents=True, exist_ok=True)
                    zip_ref.extractall(dest_sub_dir)
                print(f"[+] Успешно распаковано в: {dest_sub_dir.name}")
            except Exception as e:
                print(f"[-] Ошибка распаковки {name}: {e}")
        else:
            print(f"[-] Архив не найден в загрузках: {name}")

if __name__ == "__main__":
    main()
