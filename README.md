# Mixing-AI (FL Studio Vocal Assistant)

ИИ-ассистент для автоматического сведения вокала в FL Studio по принципу **Style Transfer** (перенос характера сведения с референса на ваш вокал с использованием нативных плагинов).

---

## 🚀 Быстрый старт

### 1. Установка зависимостей
Проект использует `poetry` для управления виртуальным окружением.
```bash
poetry install
```

### 2. Настройка FL Studio (Один раз)
При первом запуске пайплайна или клиента скрипт автоматически устанавливает MIDI-мост по верному пути в ваших документах (поддерживает русские папки и OneDrive).

1. Запустите FL Studio.
2. Нажмите **F10** (MIDI Settings).
3. В разделе **Input** найдите порт **`Mixing-AI`** (или `Mixing Ai`).
4. Нажмите кнопку **Enable** (Включить).
5. В списке **Controller type** выберите **`Mixing-AI Super Bridge`** (он появится в списке после первого запуска пайплайна).

---

## 🛠️ Основные команды

### 1. Верификация и тесты
Запуск автоматических тестов (проверка DSP-движка, маппинга, нормализации громкости и буфера обмена):
```bash
poetry run pytest
```
С подсчетом покрытия (coverage):
```bash
poetry run pytest --cov=app/core
```

### 2. Сквозной тест пайплайна (E2E)
Запуск анализа файлов `source.wav` и `reference.wav`, автозагрузки плагинов в FL Studio и настройки параметров эквалайзера:
```bash
poetry run python -m app.core.full_pipeline
```

### 3. Генерация датасета (Этап 2)
Синтез пар `dry` и `wet` вокала для обучения нейросетей с помощью оффлайн-движка `pedalboard` (Spotify) и разделения вокала от бита через `Demucs`:

* **Шаг 1 (Опционально): Загрузка облегченного VCTK-минисета (~10-20 МБ)**:
  Если у вас нет полной версии VCTK, вы можете скачать 100 высококачественных WAV-файлов дикторов с Hugging Face:
  ```bash
  poetry run python scripts/download_vctk_subset.py
  ```
* **Шаг 2: Генерация сэмплов**:
  * **Быстрый тест (нарезка вашего source.wav на 10 сэмплов)**:
    ```bash
    poetry run python scripts/prepare_dataset.py --max-samples 10
    ```
  * **Полноценная генерация со SoundCloud и VCTK (с пропорцией 70% SC / 30% VCTK)**:
    1. Добавьте ссылки на плейлисты/треки в `data/raw/soundcloud_urls.txt`
    2. Запустите генерацию с указанием папки VCTK и соотношения:
       ```bash
       poetry run python scripts/prepare_dataset.py --max-samples 1000 --sc-ratio 0.7 --vctk-dir data/raw/vctk_subset --vctk-limit 15
       ```

---

## 📂 Структура проекта

* `app/core/` — Ядро ассистента:
  * `audio_processor.py` — Загрузка аудио, silence trimming, LUFS-нормализация, извлечение Mel-спектрограмм и фичей.
  * `bridge_client.py` — Единый контроллер FL Studio (фокусировка окна, отправка SysEx, раскладко-независимая вставка текста, MIDI CC параметры).
  * `dsp_engine.py` — Оффлайн обработка на базе `pedalboard` (EQ, Compressor, Reverb, Delay), нормализация параметров к `[0..1]`.
  * `vocal_collector.py` — Загрузчик с SoundCloud (плейлисты/альбомы) + разделение стема вокала через Demucs, Voice Activity фильтр.
  * `dataset_generator.py` — Генератор разметки `index.jsonl` (17-мерный вектор с маскированием) и `.npz` спектрограмм.
* `MixingAI_Bridge/` — MIDI-скрипт устройства для FL Studio.
* `scripts/` — Вспомогательные CLI скрипты:
  * `prepare_dataset.py` — Оркестрация сборки и генерации датасета.
  * `download_vctk_subset.py` — Загрузчик мини-версии VCTK с Hugging Face.
* `tests/` — Автоматические юнит-тесты.
* `data/` — Локальные файлы данных (WAV, URL-файлы, кэши спектрограмм).
