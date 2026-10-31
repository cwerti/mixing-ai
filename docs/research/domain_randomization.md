# Доменная рандомизация: разные реализации эффектов в данных модели и кодировщика стиля

Дата: 2026-10-30. Ветка `feat/chain-benchmark`. Продолжение `docs/research/model_v5.md` и `style_metric_v2.md`.

**Проблема.** Модель v5 и кодировщик стиля v2b учились только на нашем DSP (`DSPEngine`: Freeverb, pedalboard
Compressor, tanh-дисторшн и т.д.). На 8 реальных парах модель в режиме model не видит реальный реверб Whiptails
и включает не те эффекты: она узнает «наш алгоритм», а не «что сделано». Инженеры работают другими плагинами
(Valhalla, FabFilter, стоковые FL), поэтому каждый эффект в данных должен приходить в разных реализациях.

## Коротко

1. **v6 в режиме model теперь видит реверб Whiptails.** Вероятность reverb на Whiptails 0.001 → 1.00
   (same и cross). Режим model впервые в среднем ближе к цели, чем сухой исходник. Ошибка log-mel по 8 парам:
   v5 — 2.76 / 2.86 дБ (same / cross), v6 — **1.77 / 2.19 дБ**, сухой — 2.47.
2. **model+optimizer на cross теперь лучше optimizer.** v9: 2.18 против 2.58 дБ; в v8 (v5) было наоборот:
   2.98 против 2.47. Выигрыш на whiptails (4.23 против 4.70), oxeye_leadvox (0.82 против 1.36) и
   oxeye_backing1 (0.99 против 3.19). Остальные пары — вничью. Регрессии v8 на ronnie cross (7.74) больше нет:
   теперь 3.89.
3. **Цена — палитра без рандомизации.** На val dataset_v4 (только наш DSP) macro-F1 0.825 → 0.762:
   stereo_enhancer 0.89 → 0.62, distortion 0.52 → 0.40, chorus 0.54 → 0.46. На рандомизированном val
   dataset_v5: v5 0.666 → **v6 0.742**. FPR компрессора и деэссера у v6 выше: 0.95 и 0.80.
4. **Кодировщик v3 инвариантнее к реализации и не хуже на нашей палитре.** Тройки «та же цепочка через другую
   реализацию»: 0.678 (v2b) → **0.770**. Обычные тройки на старом val: 0.817 → 0.820. На реальных парах
   порядок режимов по ошибке log-mel чуть лучше: same 22/40 → 24/40 (τ +0.07 → +0.18), cross 28/43 у обоих
   (τ +0.28 → +0.32). Тон кодировщик по-прежнему не различает. Критерий (a) стал хуже: 7/8 → 6/8.
   Кодировщик по умолчанию не заменен.
5. Новая подозрительная деталь: v6 включает distortion на Whiptails (p 0.77–0.86). Истинной цепочки нет,
   нужно прослушивание.

## 1. Что собрано

### Импульсные характеристики

Скачаны только архивы с WAV. Скрипты загрузки и распаковки лежали в scratchpad сессии, в репозиторий не попали.
Загрузка шла параллельными range-запросами с повторами (прокси нестабилен). Распаковка — через `zipfile` с
проверкой путей: без `..`, абсолютных путей и симлинков. Извлекались только `.wav` и тексты лицензий.
В каждой папке `data/raw/irs/<source>/` лежит `README.md` с адресом источника и лицензией.

| Источник | Лицензия | Что это | ИХ в банке | RT60, медиана (мин–макс) |
|---|---|---|---|---|
| `voxengo` — Voxengo IMreverbs (voxengo.com/impulses) | Voxengo: royalty-free для любых целей, без продажи и перераспространения | модельные залы, комнаты, эхо-холлы (Impulse Modeler); 4 кабинета исключены | 34 | 0.99 с (0.46–8.6) |
| `york_vr_game_spaces` — Popp & Murphy, Zenodo 10518139 | CC BY 4.0 | реверб игрового движка (FMOD/Steam Audio), 12 виртуальных комнат | 12 | 0.59 с (0.23–4.1) |
| `ccrma_shoebox` — May, Farzaneh, Das, Abel, Zenodo 15739328 | CC BY 4.0 | одна комната, 8 позиций, разные методы: замер (свип, шар), image method, ray tracing, SDN, modal, Sabine, FEM+IM | 90 | 5.84 с (2.3–8.0) |
| `detmold_srir` — Amengual Garí и др., Zenodo 4116247 | CC BY 4.0 | камерный зал, концертхаус (в т.ч. 12 ИХ с электронной системой реверберации, RT 1.8–5 с), театр; бинауральная голова (стерео) и канал 0 массива | 97 | 1.55 с (0.95–10.1) |
| синтетика (`app/audio/synthetic_reverbs.py`) | наш код | плейт Даторро 61, FDN 8/16 линий 84, пружина 28, гейт 27 | 200 | 0.17–10 с |
| **итого** | | | **433** (25 мин ИХ) | |

Не взяты:
- **Ely Cathedral Lady Chapel** (Zenodo 5150020). Zenodo указывает CC BY, но `LICENCE.txt` внутри архива —
  CC BY-NC-SA 4.0. Взята более строгая лицензия, ИХ исключены из банка. Файлы остались на диске.
- **OpenAIR**: у сайта истек TLS-сертификат. Проверку TLS не отключали.
- **EchoThief**: сервер недоступен через прокси.
- **MIT IR Survey**: у авторов нет явной лицензии (CC BY стоит только у сторонних зеркал на HF).
- **Plate/spring Martínez Ramírez** (Zenodo 3746119, CC BY): это пары dry/wet гитары, а не ИХ. Выделять из них ИХ
  деконволюцией не стали, пока хватает синтетических плейтов и пружин.

Кэш `data/raw/irs/_cache/` строит `scripts/build_ir_cache.py`:
- `irs.npy` — float16, memmap, страницы общие для всех процессов-генераторов;
- `index.json` — источник, лицензия и RT60 каждой ИХ.

Обработка каждой ИХ:
- 44.1 кГц; стерео остается стерео, B-format и массивы — канал 0;
- обрезка до начала звука;
- прямой звук приглушен окном 0→5 мс: сухой сигнал микшируется отдельно;
- хвост до −60 дБ по кривой Шрёдера, не длиннее 12 с;
- энергия канала нормирована к 4, как у `decay_reverb`, поэтому wet_level значит то же, что у Freeverb.

На диске: скачанное — 0.8 ГБ, кэш — 0.25 ГБ.

### VST3

`scripts/vst_inventory.py` ищет `.vst3` в стандартных папках: `Common Files/VST3`, `VSTPlugins`, Steinberg,
Image-Line, `%LOCALAPPDATA%/Programs/Common/VST3`. Каждый плагин грузится `pedalboard.load_plugin` в отдельном
процессе с таймаутом. Опись пишет имена параметров, категорию по имени, детерминизм и конечность двух оффлайн-рендеров шума,
а также скорость (x реального времени) в `data/raw/vst_inventory.json`.

**На этой машине нет ни одного VST3.** Нет ни папок VST3/VST2, ни FL Studio, ни ключей реестра VST. Поиск `*.vst3`
по дискам C: и D: ничего не нашел. Ничего не устанавливалось. VST-путь в `effect_variants` реализован, но не
используется. Тест VST-пути пропускается, пока опись пуста.

Что можно поставить вручную (бесплатные VST3) и перезапустить `python scripts/vst_inventory.py`:

| Категория | Плагин |
|---|---|
| reverb | Valhalla Supermassive, Valhalla Freq Echo (дилей), TAL-Reverb-4, Dragonfly Reverb (GPL) |
| compressor / limiter | TDR Kotelnikov, Klanghelm DC1A, LoudMax |
| multiband / OTT | Xfer OTT, TDR Nova (динамический EQ) |
| saturation | Softube Saturation Knob, Klanghelm IVGI, Variety of Sound (FerricTDS, Tessla) |
| EQ | TDR Nova, TDR SlickEQ, MeldaProduction MEqualizer |
| modulation | TAL-Chorus-LX, Blue Cat Chorus, MeldaProduction MFlanger/MChorus |
| pitch | Auburn Sounds Graillon 2 Free, MAutoPitch (Melda Free Bundle) |
| набор | MeldaProduction MFreeFXBundle (~37 плагинов, есть всё выше) |

## 2. Дизайн

### `app/audio/effect_variants.py` — только для данных

`EffectVariants.apply_chain(y, chain, rng)` обходит цепочку:
- каждый плагин с вероятностью `own_prob` (0.35) рендерится нашим `DSPEngine`, иначе — случайной
  альтернативой с весами;
- подряд идущие «свои» плагины рендерятся одним вызовом `DSPEngine.apply_chain(..., final_limiter=False)`,
  поэтому слои harmony и экситер работают как раньше;
- в конце стоит тот же лимитер −1 дБ;
- сломавшаяся реализация (например, VST) заменяется своей;
- метка не меняется (плагин схемы и нормализованные параметры), физические параметры переносятся
  приблизительно, с разбросом.

Путь оптимизатора не тронут: единственное изменение в `DSPEngine` — флаг `final_limiter`, по умолчанию `True`.
`own_prob = 1` дает ровно `DSPEngine.apply_chain` (тест).

| Плагин | Альтернативы |
|---|---|
| reverb | свёртка с внешней ИХ, подобранной по RT60 (room_size → RT60 по замеру Freeverb, `FREEVERB_RT60`); свёртка с синтетической (плейт/FDN/пружина/гейт); Freeverb как send с другими predelay, фильтрами и шириной; `decay_reverb` с разбросом RT60 |
| compressor | feed-forward RMS (окно 3–50 мс, колено 0–12 дБ); пиковый; feedback «опто» с программно-зависимым release (numba); pedalboard с разбросом. Порог ±3 дБ, ratio ×0.75–1.33, атака/release ×0.5–2 |
| deesser | динамическая полоса с границей 4.5–8 кГц; широкополосный с ВЧ-сайдчейном |
| multiband_compressor | 3 полосы с другими частотами раздела, вверх + вниз со сглаживанием; только вниз (Maximus/C6-подобный) |
| distortion | tanh (insert, микс 0.5–1); ламповая асимметрия; лента (arctan, потеря верха, head bump); жесткий клип; биткраш. Громкость выравнивается по RMS |
| chorus | многоголосый стерео (2–4 голоса); флэнжер с обратной связью; «dimension»; pedalboard Chorus с разбросом |
| delay | пинг-понг; ленточное эхо (темнеющие насыщенные повторы); разные времена L/R ×1.25–1.6. Время ±3 % |
| stereo_enhancer | MicroShift (±5–15 центов + задержка); декоррелятор (фазовращатели → Side); классический Хаас |
| eq | другие добротности полос; линейно-фазовая кривая в частотной области |
| exciter | «air»-полка; выпрямитель + tanh на ВЧ |
| VST3 | если в описи есть плагин категории: параметры с понятными именами — по нормализованным параметрам ±0.15, прочие — 0.3–0.7 |

Тюн, harmony и подавление резонансов всегда рендерятся нашим DSP: альтернатив не нашлось. Скорость: 20
цепочек по 3.5 с — 3.7 с против 3.4 с у своего DSP.

### Генератор датасета

- `DatasetGenerator.generate(domain_rand_own_prob=0.35)` и `prepare_dataset.py --domain-rand 0.35`.
- Реализации выбирает отдельный генератор (seed от номера сэмпла), поэтому файлы и сплит те же, что у
  dataset_v4.
- В сэмпле — `ref_impls`, в `stats.json` — `ref_impl_counts` и `param_ranges`.
- Добавлены `--workers` и `--resume`: прерванную генерацию можно дописать, сэмпл детерминирован номером.

### Правки по ревью

- **Диапазоны в чекпоинте.** `train_ml.py` раньше писал в чекпоинт текущие диапазоны схемы. Теперь он берет
  их из датасета через `dataset_param_ranges`: сначала `stats.json["param_ranges"]`; для dataset_v4 —
  `DATASET_V4_PARAM_RANGES` (он сгенерирован до подъема потолков реверба в 4b90141); иначе схема с
  предупреждением. Чекпоинт v5 обучен до 4b90141 и хранит верные старые диапазоны.
- **Сэмплирование wet.** Генераторы сэмплировали нормализованные wet_level и room_size, а потолки подняли:
  wet до 0.60, room до 0.99. Без правки физический wet датасета сместился бы до ~0.57. Теперь они
  сэмплируются в прежних диапазонах и пересчитываются (`_legacy_reverb_norm`): медиана wet 0.22, 90 % ниже
  0.33. В стилях pluggnb_rnb, rage и hyperpop в 12 % случаев wet берется 0.36–0.55 (всего 4.8 % реверба).
  Метка в текущих диапазонах совпадает с рендером. Из-за лишнего случайного числа в этих стилях цепочки
  dataset_v5 не совпадают с dataset_v4 один в один.

### dataset_v5

Как dataset_v4: 30000 сэмплов (25500 / 4500), seed 0, cross_ratio 1.0, realistic, harmony 0.35, песни теста
исключены. Плюс `--domain-rand 0.35`. Доля своей реализации по плагинам 0.33–0.36. Реверб:
- внешняя ИХ — 23 %;
- синтетическая — 23 %;
- Freeverb-send — 10 %;
- decay-send — 9 %;
- свой — 35 %.

Генерация ~2.5 ч на 10–20 процессах (с прерыванием и `--resume`), 7.6 ГБ.

### Модель v6

Рецепт v5: `python scripts/train_ml.py --dataset-dir data/processed/dataset_v5 --save-dir data/models/v6
--epochs 35 --patience 8 --pos-weight none`. Ранняя остановка на эпохе 32, лучшая эпоха 24: val macro-F1 0.765,
MAE 0.139 на рандомизированном val. Лог: `data/models/train_v6.log`.

### Кодировщик стиля v3: что считать позитивом

Блок, как раньше: 4 цепочки × 4 фрагмента. Фрагменты 0 и 2 (разные песни) рендерятся нашим DSP, 1 и 3 — одним
случайным профилем реализаций на цепочку. Профиль задается seed: те же реализации и тот же разброс на обоих
фрагментах (`style_chains.render_chain(variants=...)`, `train_style_encoder.py bank --domain-rand 0.35`).

Позитивы в SupCon:
- та же цепочка той же реализацией на другом содержании — вес 1 (обязательное условие);
- та же цепочка другой реализацией — **вес 0.5** (`--cross-impl-weight`).

Почему не 0 и не 1. В подборе сравниваются наш рендер и чужой референс, поэтому метрике нужна инвариантность
именно к реализации. Но соответствие параметров между реализациями приблизительное: ±30 % по временам,
±3 дБ по порогу, другой тип реверба. Полный вес учил бы игнорировать сами параметры. Половинный вес тянет
реализации друг к другу, но оставляет приоритет точному совпадению.

Данные и обучение:
- банки: два старых без рандомизации (по 3000 блоков) и новый `_dr` (3000 блоков, 40 мин на 8 процессах);
- val — `bank_val_dr` (240 блоков);
- обучение как у v2b: 10000 шагов, трудные негативы EQ 0.5, 18 мин GPU;
- лучший шаг 7500 → `data/models/style_encoder/style_encoder_v3.pt`;
- `style_encoder.pt` (v2b) не тронут.

## 3. Результаты

### Валидация модели (сырой порог 0.5, `compare_models.py val/report`, `data/runs/domain_rand/compare*`)

| Набор | macro-F1 v5 | macro-F1 v6 | MAE v5 | MAE v6 |
|---|---|---|---|---|
| val dataset_v4 (наш DSP) | **0.825** | 0.762 | **0.113** | 0.133 |
| val dataset_v5 (рандомизация) | 0.666 | **0.742** | — | ниже по всем плагинам, кроме distortion |

На рандомизированном val v6 лучше по F1 везде:
- distortion 0.35 → 0.57;
- delay 0.80 → 0.92;
- reverb 0.90 → 0.97;
- multiband 0.51 → 0.59.

На val dataset_v4 v6 хуже по stereo_enhancer (0.89 → 0.62), distortion и chorus. Калибровка у v6 уже хорошая
(ECE 0.02–0.15), поэтому Platt для v6 не сохранялся. Оговорка: MAE на val dataset_v4 сравнивает метки в старых
диапазонах реверба с предсказаниями v6 в новых, поэтому MAE реверба (0.098 → 0.129) завышен для v6.

### Кодировщики на синтетических val-банках (тройки / top-1)

| Кодировщик | bank_val (только наш DSP) | bank_val_dr: все | same_impl | cross_impl |
|---|---|---|---|---|
| v1 | 0.819 / 0.33 | 0.716 / 0.19 | 0.791 | 0.678 |
| v2b (по умолчанию) | 0.817 / 0.37 | 0.716 / 0.23 | 0.792 | 0.678 |
| **v3** | **0.820** / 0.36 | **0.781 / 0.24** | **0.803** | **0.770** |

### 8 реальных пар

Бюджет 60 с, patience 60, seed 0, метрика подбора `embedding` (v2b), как в v8. Прогоны:
- `data/runs/benchmark/real_pairs_v9.jsonl` — v6;
- `real_pairs_v9_v5model.jsonl` — v5 model-only тем же кодом;
- `real_pairs_v8_embedding.jsonl` — v5, режимы с оптимизатором.

Ошибка log-mel к wet A, дБ (среднее по 8 парам):

| Режим | same v5/v8 | same v6/v9 | cross v5/v8 | cross v6/v9 |
|---|---|---|---|---|
| сухой | 2.47 | 2.47 | 2.47 | 2.47 |
| model | 2.76 | **1.77** | 2.86 | **2.19** |
| model+optimizer | 1.94 | 2.02 | 2.98 | **2.18** |
| optimizer | 2.01 | 2.02 | 2.47 | 2.58 |

- mrstft (cross): model 2.47 → 2.60, model+opt 3.04 → 2.20.
- f0: model+opt 6.3–6.5 ц у обоих; тюн выбирают правила.
- optimizer-only немного отличается между v8 и v9 (backing1 cross 2.38 → 3.19) из-за правок оптимизатора
  после v8 (26bca72).

Плагины режима model:

| Пара | v5 | v6 |
|---|---|---|
| whiptails same/cross | eq, compressor, deesser (p reverb 0.001) | eq, compressor, deesser, **distortion, reverb** (p reverb 1.00, distortion 0.77–0.86) |
| ronnie tape cross | eq, pitch, compressor, distortion, exciter | eq, compressor, distortion, exciter (без тюна) |
| oxeye ×6 | eq ± compressor/deesser | eq ± compressor/deesser |

Стерео v6 на Whiptails не предсказывает (p 0.004–0.008). Ширина Whiptails, видимо, идет от стерео-реверба.

Согласие порядка режимов с прямой ошибкой: выходы v9 — сухой, model, model+opt, optimizer; пары с
|Δ| ≥ 0.05 дБ / 0.5 ц:

| Кодировщик | log-mel same | log-mel cross | f0 same | f0 cross | (a) wet ближе сухого |
|---|---|---|---|---|---|
| v2b | 22/40 (τ +0.07) | 28/43 (τ +0.28) | 8/20 | 10/20 | 7/8 |
| v3 | **24/40 (τ +0.18)** | 28/43 (**τ +0.32**) | 8/20 | 7/20 | 6/8 |

Прослушивание — `data/processed/real_pairs_v9/<пара>/`, основная папка — cross:
- `model.wav`, `model_optimizer.wav`, `optimizer_only.wav` — v6;
- `*_style.wav` — те же режимы из прогона v8 (v5, для A/B), суффикс исторический;
- `dry.wav`, `wet_true.wav`, `reference_used.wav`, `chains.txt`.

## 4. Выводы

1. Доменная рандомизация решила главную жалобу: модель перестала пропускать реальный реверб. Режим model
   стал полезным стартом, и model+optimizer на cross впервые лучше подбора с нуля. Выборка маленькая: 8 пар
   из 3 песен, 6 из них — один тюн. Это сигнал, а не доказательство.
2. Платой стала точность на нашей палитре: −0.06 macro-F1 на val dataset_v4, в основном стерео и сатурация.
   Для подбора это не критично: оптимизатор все равно рендерит нашим DSP. Если терять нельзя, можно поднять
   `own_prob` до 0.5 или дообучить v6 на смеси dataset_v4 и dataset_v5.
3. Distortion на Whiptails у v6 нужно проверить на слух. Рандомизированная сатурация (лента, клип) могла
   научить модель путать плотный реверб с сатурацией.
4. Кодировщик v3 лучше v2b на межреализационных тройках и не хуже на нашей палитре. На реальных парах
   прирост маленький: τ с log-mel +0.07 → +0.18 same, +0.28 → +0.32 cross, а (a) 7/8 → 6/8. Делать его
   метрикой по умолчанию пока рано. Следующий шаг — прогон подбора с `--metric embedding` на v3 (сейчас
   кодировщик подбора берется из `DEFAULT_CHECKPOINT`).
5. VST3 на машине нет. После установки бесплатных плагинов из списка выше достаточно перезапустить
   `scripts/vst_inventory.py` и пересобрать данные: VST войдут как еще одна реализация.

## Воспроизведение

```
python scripts/vst_inventory.py
python scripts/build_ir_cache.py --synthetic 200
python scripts/prepare_dataset.py --max-samples 30000 --output-dir data/processed/dataset_v5 --seed 0 \
    --prior realistic --harmony-ratio 0.35 --domain-rand 0.35 --workers 10 [--resume]
python scripts/train_ml.py --dataset-dir data/processed/dataset_v5 --save-dir data/models/v6 --epochs 35 \
    --patience 8 --pos-weight none
python scripts/train_style_encoder.py bank --blocks 240 --split val --suffix _dr --domain-rand 0.35 --workers 8
python scripts/train_style_encoder.py bank --blocks 3000 --split train --suffix _dr --domain-rand 0.35 --workers 8
python scripts/train_style_encoder.py train --train-banks "" _2 _dr --val-suffix _dr --minutes 50 --steps 10000 \
    --eq-aug 0.5 --cross-impl-weight 0.5 --out-name style_encoder_v3.pt
python scripts/real_pairs.py --model-path data/models/v6/best_model.pth --metric embedding \
    --modes model model+optimizer optimizer --out data/runs/benchmark/real_pairs_v9.jsonl \
    --audio-tmp data/runs/benchmark/real_pairs_audio_v9 --listen-dir data/processed/real_pairs_v9 \
    --compare-audio-tmp data/runs/benchmark/real_pairs_audio_v8_embedding
```

Тесты: `tests/test_effect_variants.py`. Каждая реализация дает конечный стерео той же длины. Свёртка с ИХ
соблюдает wet/dry. Выбор ИХ по RT60. `own_prob = 1` совпадает с `DSPEngine`. Рандомизация воспроизводима по
seed. VST-путь пропускается без плагинов. Опция генератора не меняет метки. Проверены `--resume`, диапазоны
датасета и физическое распределение wet. Полный pytest: 132 passed, 1 skipped, rc=0. Этапы и логи:
`data/runs/domain_rand/`.
