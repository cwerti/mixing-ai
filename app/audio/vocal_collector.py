import os
import sys
import subprocess
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Optional

# Настройка кодировки консоли для избежания вылетов при выводе кириллицы на Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

class VocalCollector:
    """
    Класс для сбора сухого вокала.
    Управляет скачиванием треков с SoundCloud (через yt-dlp), 
    выделением вокала из миксов (через Demucs) и нарезкой аудио на сэмплы
    с фильтрацией тишины (Voice Activity Filter).
    """
    
    def __init__(self, sr: int = 44100):
        """
        Инициализирует сборщик вокала.
        
        Args:
            sr: Рабочая частота дискретизации аудио. По умолчанию 44100 Гц.
        """
        self.sr = sr
        # Автоматическое добавление FFmpeg, установленного через winget, в переменную окружения PATH
        try:
            import os
            winget_base = Path(os.path.expanduser("~")) / "AppData/Local/Microsoft/WinGet/Packages"
            if winget_base.exists():
                ffmpeg_bins = list(winget_base.rglob("ffmpeg.exe"))
                if ffmpeg_bins:
                    ffmpeg_bin_dir = ffmpeg_bins[0].parent
                    if str(ffmpeg_bin_dir.absolute()) not in os.environ["PATH"]:
                        os.environ["PATH"] = str(ffmpeg_bin_dir.absolute()) + os.pathsep + os.environ["PATH"]
        except Exception:
            pass

    def download_soundcloud(self, urls_file: Path, download_dir: Path) -> List[Path]:
        """
        Скачивает аудиофайлы с SoundCloud из списка URL-адресов с помощью yt-dlp.
        
        Args:
            urls_file: Путь к текстовому файлу со списком ссылок.
            download_dir: Папка для сохранения скачанных аудиофайлов.
            
        Returns:
            List[Path]: Список путей к скачанным файлам в формате WAV.
        """
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        if not urls_file.exists():
            print(f"[-] Файл со ссылками SoundCloud не найден: {urls_file}")
            return downloaded_files

        with open(urls_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        print(f"[*] Найдено ссылок для скачивания с SoundCloud: {len(urls)}")
        archive_file = download_dir.parent / "soundcloud_download_archive.txt"
        for url in urls:
            print(f"[*] Скачивание трека/плейлиста: {url}...")
            try:
                # Используем yt-dlp для скачивания и извлечения в wav
                out_tmpl = str(download_dir / "%(title)s.%(ext)s")
                cmd = [
                    "yt-dlp",
                    "-x", "--audio-format", "wav",
                    "--yes-playlist",
                    "--ignore-errors",
                    "--download-archive", str(archive_file.absolute()),
                    "-o", out_tmpl,
                    url
                ]
                subprocess.run(cmd, check=True)
                print(f"[+] Успешно обработана ссылка {url}")
            except Exception as e:
                print(f"[-] Ошибка скачивания {url}: {e}")

        # Собираем все скачанные файлы
        for f_path in download_dir.glob("*.wav"):
            downloaded_files.append(f_path)
            
        return downloaded_files

    def separate_vocals(self, audio_path: Path, separation_dir: Path) -> Optional[Path]:
        """
        Выделяет вокал из музыкального трека с помощью утилиты Demucs.
        
        Args:
            audio_path: Путь к исходному полному треку.
            separation_dir: Папка для сохранения результатов разделения.
            
        Returns:
            Optional[Path]: Путь к извлеченной дорожке вокала (vocals.wav) или None в случае ошибки.
        """
        separation_dir.mkdir(parents=True, exist_ok=True)
        
        song_name = audio_path.stem
        vocals_file = separation_dir / "htdemucs" / song_name / "vocals.wav"
        
        if vocals_file.exists():
            print(f"[+] Извлеченный вокал уже существует: {vocals_file}. Пропуск разделения.")
            return vocals_file
            
        print(f"[*] Запуск разделения вокала через Demucs для: {audio_path.name}...")
        
        try:
            # Команда для выделения только вокала (две дорожки: вокал и остальное)
            cmd = [
                "demucs",
                "--two-stems=vocals",
                "-o", str(separation_dir.absolute()),
                str(audio_path.absolute())
            ]
            subprocess.run(cmd, check=True)
            
            vocals_file = separation_dir / "htdemucs" / song_name / "vocals.wav"
            if vocals_file.exists():
                print(f"[+] Извлеченный вокал сохранен в: {vocals_file}")
                return vocals_file
            else:
                print(f"[-] Не удалось найти результат разделения по пути: {vocals_file}")
                return None
        except Exception as e:
            print(f"[-] Ошибка выделения вокала через Demucs: {e}")
            return None

    def slice_audio(self, audio_path: Path, output_dir: Path, prefix: str, segment_sec: float = 8.0, min_rms: float = 0.015, min_vocal_ratio: float = 0.5) -> List[Path]:
        """
        Нарезает аудиофайл на отрезки фиксированной длительности, 
        пропуская фрагменты с низкой вокальной активностью (Voice Activity Filter).
        
        Args:
            audio_path: Путь к аудиофайлу.
            output_dir: Папка для сохранения нарезанных сегментов.
            prefix: Префикс для имен выходных файлов.
            segment_sec: Длительность сегмента в секундах. По умолчанию 8.0 с.
            min_rms: Порог RMS для определения наличия голоса.
            min_vocal_ratio: Минимальная доля активных кадров в сегменте для его сохранения.
            
        Returns:
            List[Path]: Список путей к сохраненным сегментам аудио.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        sliced_paths = []
        
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            seg_samples = int(segment_sec * sr)
            total_samples = len(y)
            num_segments = total_samples // seg_samples
            
            print(f"[*] Нарезка {audio_path.name} на сегменты (Фильтр вокала: min_vocal_ratio={min_vocal_ratio})...")
            
            # Параметры фреймов для расчета RMS (окно 50мс, шаг 25мс)
            frame_len = int(0.05 * sr)
            hop_len = int(0.025 * sr)
            
            # Обработка слишком коротких файлов (дополнение нулями до 8 секунд)
            if total_samples < seg_samples:
                if total_samples >= frame_len:
                    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
                    frame_rms = np.sqrt(np.mean(frames**2, axis=0))
                    vocal_frames = np.sum(frame_rms > min_rms)
                    vocal_ratio = vocal_frames / len(frame_rms)
                    
                    if vocal_ratio >= min_vocal_ratio:
                        padded = np.zeros(seg_samples, dtype=np.float32)
                        padded[:total_samples] = y
                        segment_path = output_dir / f"{prefix}_short.wav"
                        sf.write(segment_path, padded, sr)
                        sliced_paths.append(segment_path)
                return sliced_paths
                
            for i in range(num_segments):
                start = i * seg_samples
                end = start + seg_samples
                chunk = y[start:end]
                
                # Расчет RMS фреймов
                frames = librosa.util.frame(chunk, frame_length=frame_len, hop_length=hop_len)
                frame_rms = np.sqrt(np.mean(frames**2, axis=0))
                
                # Оценка доли фреймов, превысивших порог вокальной активности
                vocal_frames = np.sum(frame_rms > min_rms)
                vocal_ratio = vocal_frames / len(frame_rms)
                
                if vocal_ratio < min_vocal_ratio:
                    continue  # Пропускаем отрезок с тишиной или шепотом
                
                segment_path = output_dir / f"{prefix}_{i:03d}.wav"
                sf.write(segment_path, chunk, sr)
                sliced_paths.append(segment_path)
                
            return sliced_paths
        except Exception as e:
            print(f"[-] Нарезка файла {audio_path} завершилась ошибкой: {e}")
            return []
