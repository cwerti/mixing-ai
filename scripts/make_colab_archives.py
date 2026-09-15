import zipfile
import os
from pathlib import Path

def zip_directory(src_dir, zip_name, exclude_dirs=None, exclude_files=None):
    if exclude_dirs is None:
        exclude_dirs = []
    if exclude_files is None:
        exclude_files = []
        
    src_path = Path(src_dir)
    print(f"[*] Создание архива {zip_name} из папки {src_path}...")
    
    count = 0
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(src_path):
            root_path = Path(root)
            
            # Фильтруем папки на месте, чтобы os.walk не заходил в них
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if file.startswith('.') or file in exclude_files:
                    continue
                file_path = root_path / file
                
                # Проверяем, не находится ли файл в исключенных папках
                parts = file_path.parts
                if any(ex in parts for ex in exclude_dirs):
                    continue
                    
                rel_path = file_path.relative_to(src_path)
                zip_file.write(file_path, rel_path)
                count += 1
                
    print(f"[+] Архив {zip_name} успешно создан! (Добавлено файлов: {count})")

def main():
    project_root = Path(__file__).parent.parent.absolute()
    os.chdir(project_root)
    
    # 1. Создаем архив исходного кода проекта (без тяжелых данных и кэшей)
    zip_directory(
        src_dir=".",
        zip_name="mixing-ai.zip",
        exclude_dirs=["data", ".venv", ".git", "__pycache__", ".pytest_cache", ".idea", ".vscode", "runs"],
        exclude_files=["mixing-ai.zip", "dry_vocals.zip"]
    )
    
    # 2. Создаем архив сухих вокалов
    dry_vocals_dir = Path("data/raw/dry_vocals")
    if dry_vocals_dir.exists():
        zip_directory(
            src_dir=str(dry_vocals_dir),
            zip_name="dry_vocals.zip"
        )
    else:
        print("[!] Папка data/raw/dry_vocals не найдена. Пожалуйста, добавьте сухой вокал перед запуском.")

if __name__ == "__main__":
    main()
