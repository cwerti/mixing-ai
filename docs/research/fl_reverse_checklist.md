# Чек-лист: пресеты стоковых плагинов FL для реверса .fst

Цель: получить пары «известное значение ручки → байты». Каждый пресет отличается от `default` **одной** ручкой,
поэтому дифф блоба (событие 213, первый int32 = версия данных) сразу показывает, где лежит параметр и как он
кодируется (линейно, логарифмически, со знаком). Combo-пресет нужен для проверки декодера.

Имена ручек взяты из онлайн-мануала Image-Line (ссылки внизу). Где мануал не даёт диапазон, пишем `min`/`max`
(ручку довести до упора), а точное значение берём из hint bar.

## 0. Подготовка (один раз, ~5 мин)

1. Создайте новый пустой проект. Поставьте плагин в свободный слот на insert-треке микшера.
2. Создайте папки `D:\work\any\mixing-ai\data\raw\presets\reverse\<plugin>\` с такими slug:
   `limiter`, `reeverb2`, `delay3`, `maximus`, `eq2`, `chorus`, `flangus`, `mbcomp`.
3. В каждой папке заведите `notes.txt`. Первая строка: версия FL (Help → About). Дальше по строке на пресет:
   `<имя файла> | <ручка> | <точный текст из hint bar>`. Это эталон, даже если значение получилось не круглым.

### Как сохранить пресет

Окно плагина → меню-стрелка в левом верхнем углу (Plugin Menu) → **Presets → Save preset as…** → нужная
папка → имя файла → сохранить (формат `.fst`). Не путайте с **Save channel state as…** (там лишний state обёртки).

### Как ставить точные значения

| Действие | Как |
|---|---|
| Ввести число | ПКМ по ручке → **Type in value** |
| Тонкая подстройка | тянуть с зажатым **Ctrl**, смотреть hint bar |
| min / max | довести ручку до упора (тянуть дальше края) |
| Сброс к дефолту | **Alt+ЛКМ**, средняя кнопка мыши или ПКМ → reset |
| Проверка | навести на ручку, сверить hint bar; записать в `notes.txt` |

### Правила

- `default` = **свежий** экземпляр плагина, сохранённый сразу после вставки. Ничего не трогать.
- После каждого пресета верните ручку в дефолт (Alt+ЛКМ) и сверьте hint bar с дефолтом. По форуму, на части
  ручек Alt+ЛКМ ставит центр, а не дефолт. Если сомневаетесь, загрузите `__default.fst` заново (двойной клик
  в браузере или Presets → открыть).
- Колесо мыши над окном плагина крутит ручки, будьте осторожны. Вкладки, вид и анализатор не менять: они тоже
  могут попасть в блоб.
- Если значение совпадает с дефолтом (например, дефолт уже `max`), пресет пропускаем и пишем `= default` в notes.
- Если значение недостижимо (диапазон уже), ставим ближайшее и записываем текст hint bar.

### Имена файлов

`<plugin>__<knob>__<value>.fst`, ручка в lower_snake, при необходимости с префиксом секции/полосы
(`comp_thresh`, `low_pre`). Как писать значение: минус → `m`, десятичная точка → `d`, единицы слитно:
`m20db`, `6db`, `2d5s`, `10ms`, `1khz`, `50pct`, `4to1`, `min`, `max`, `on`, `pingpong`.
Примеры: `limiter__comp_thresh__m20db.fst`, `delay3__model__pingpong.fst`, `maximus__low_pre__max.fst`.

## Порядок и время

Один пресет занимает ~40 с (поставить, сохранить, записать, сбросить).

| # | Плагин | Пресетов | Время |
|---|---|---|---|
| 1 | Fruity Limiter | 46 | ~30 мин |
| 2 | Fruity Reeverb 2 | 40 | ~25 мин |
| 3 | Fruity Delay 3 | 57 | ~35 мин |
| 4 | Maximus | 56 | ~40 мин |
| 5 | Fruity Parametric EQ 2 (добить 6 вопросов) | 12 | ~10 мин |
| 6 | Fruity Chorus | 25 | ~15 мин |
| 7 | Fruity Flangus | 18 | ~10 мин |
| 8 | Fruity Multiband Compressor | 30 | ~20 мин |
| | **Итого** | **284** | **~3 ч 05 мин** |

Каждый плагин самодостаточен: можно остановиться после любого. Каждая папка полезна сама по себе,
даже если следующие плагины не сделаны.

---

## 1. Fruity Limiter (`limiter`), 46 шт.

Три вкладки: LIMIT, COMP, NOISE (гейт). У вкладок одинаковые имена ATT/REL/CURVE/SUSTAIN. Хранятся ли они
раздельно, покажет дифф, поэтому проверяем в обеих вкладках.

| Секция | Ручка | Значения → суффиксы | Шт. |
|---|---|---|---|
| — | default | `__default` | 1 |
| — | вкладка (только переключить на COMP, ничего не крутить) | `ui_tab__comp` | 1 |
| LIMIT | GAIN | `min`, `max`, `6db` | 3 |
| LIMIT | SAT | `min`, `max` | 2 |
| LIMIT | CEIL (макс. +12 dB) | `min`, `max`, `m1db` | 3 |
| LIMIT | ATT (0 ms = без латентности) | `0ms`, `max`, `10ms` | 3 |
| LIMIT | CURVE (1…8) | `1`, `8` | 2 |
| LIMIT | REL | `min`, `max`, `100ms` | 3 |
| LIMIT | SUSTAIN (0…1000 ms) | `0ms`, `1000ms` | 2 |
| COMP | GAIN | `min`, `max`, `6db` | 3 |
| COMP | THRESH (0 dB = выкл) | `min`, `m20db` | 2 |
| COMP | RATIO | `min`, `max`, `4to1` | 3 |
| COMP | KNEE | `min`, `max` | 2 |
| COMP | ATT | `min`, `max` | 2 |
| COMP | CURVE | `1`, `8` | 2 |
| COMP | REL | `min`, `max` | 2 |
| COMP | SUSTAIN | `0ms`, `1000ms` | 2 |
| NOISE | GAIN (100% = гейт выкл) | `0pct`, `50pct` | 2 |
| NOISE | THRESH | `min`, `max` | 2 |
| NOISE | REL | `min`, `max` | 2 |
| COMP | SIDECHAIN (по желанию: нужен роутинг трека 1 на этот трек через sidechain send) | `track1` | 1 |
| — | combo: COMP THRESH −18 dB, RATIO 3:1, ATT 5 ms, REL 80 ms; LIMIT CEIL −0.3 dB, GAIN 3 dB | `combo__1` | 1 |

Префиксы: `limit_`, `comp_`, `noise_` (пример: `limiter__noise_rel__max.fst`).

## 2. Fruity Reeverb 2 (`reeverb2`), 40 шт.

| Ручка | Значения → суффиксы | Шт. |
|---|---|---|
| default | `__default` | 1 |
| MID/SIDE (дефолт MID) | `side` | 1 |
| L.CUT | `min`, `max`, `200hz` | 3 |
| H.CUT | `min`, `max`, `5khz` | 3 |
| DELAY (predelay) | `min`, `max`, `50ms` | 3 |
| TEMPO (predelay по темпу) | `on` | 1 |
| SIZE | `min`, `max`, `50` | 3 |
| DIFF | `min`, `max` | 2 |
| MOD | `min`, `max` | 2 |
| SPEED (mod speed) | `min`, `max` | 2 |
| BASS (multiplier) | `min`, `max` | 2 |
| CROSS (bass crossover) | `min`, `max` | 2 |
| DEC (до 20 s) | `min`, `max`, `2s` | 3 |
| DAMP (на max = OFF) | `min`, `max`, `5khz` | 3 |
| DRY | `0pct`, `max` | 2 |
| ER | `0pct`, `max` | 2 |
| WET | `0pct`, `max` | 2 |
| SEP | `min`, `max` | 2 |
| combo: L.CUT 150 Hz, H.CUT 8 kHz, DELAY 20 ms, DEC 1.5 s, DRY 80%, WET 25% | `combo__1` | 1 |

Ручки «SORT» в мануале Reeverb 2 нет. Если в вашей версии она есть, добавьте `sort__min`/`sort__max`.

## 3. Fruity Delay 3 (`delay3`), 57 шт.

| Секция | Ручка | Значения → суффиксы | Шт. |
|---|---|---|---|
| — | default | `__default` | 1 |
| Input | Wet | `min`, `max` | 2 |
| Time | Time (sync off, 1…1000 ms) | `1ms`, `1000ms`, `250ms` | 3 |
| Time | Tempo Sync on (+ Time в шагах) | `sync__on`, `sync_on_time__16steps`, `sync_on_time__1steps` | 3 |
| Time | Keep Pitch | `on` | 1 |
| Time | Smoothing | `min`, `max` | 2 |
| Time | Offset/Pan | `min`, `max` | 2 |
| Model | Mono / Stereo / Ping pong / Off (кроме дефолта) | `mono`, `stereo`, `pingpong`, `off` | 3 |
| Model | Stereo spread | `min`, `max` | 2 |
| Feedback | Level (бывает >100%) | `0pct`, `max`, `50pct` | 3 |
| Feedback | Cutoff | `min`, `max` | 2 |
| Feedback | Res | `0pct`, `max` | 2 |
| Feedback | Filter Type LP/HP/BP/Off (кроме дефолта) | `lp`, `hp`, `bp`, `off` | 3 |
| Feedback | Samp Rate | `min`, `max` | 2 |
| Feedback | Bits | `min`, `max` | 2 |
| Modulation | Rate (0…20 Hz) | `0hz`, `20hz` | 2 |
| Modulation | Time | `min`, `max` | 2 |
| Modulation | Cutoff | `min`, `max` | 2 |
| Diffusion | Level | `min`, `max` | 2 |
| Diffusion | Spread | `min`, `max` | 2 |
| Distortion | Type Limit/Sat (кроме дефолта) | `limit` / `sat` | 1 |
| Distortion | Knee (виден при Sat) | `min`, `max` | 2 |
| Distortion | Symmetry (виден при Sat) | `min`, `max` | 2 |
| Distortion | Level | `min`, `max` | 2 |
| Output | Wet | `min`, `max` | 2 |
| Output | Tone (LP ← off → HP) | `min`, `max` | 2 |
| Output | Dry | `min`, `max` | 2 |
| — | combo: sync off, Time 375 ms, Model Ping pong, Feedback 40%, Output Wet и Dry на заметных значениях (из hint) | `combo__1` | 1 |

Префиксы: `in_`, `fb_`, `mod_`, `diff_`, `dist_`, `out_` (пример: `delay3__fb_cutoff__min.fst`).
Knee/Symmetry сохраняйте при Type = Sat. Если Sat не дефолт, это два изменения, укажите это в notes.

## 4. Maximus (`maximus`), 56 шт.

В мануале нет ручек threshold/ratio/knee above/below. Порог и кривая компрессии **рисуются огибающей**
(точки), поэтому для них отдельный блок ниже. Ручки одной полосы, скорее всего, повторяются с шагом
(stride). LOW делаем полностью, у остальных полос хватит 3 пресетов, чтобы подтвердить шаг. Полоса
выбирается кнопками LOW / MID / HIGH / MASTER (сам выбор тоже сохраните: `ui_band__mid`).

| Секция | Ручка | Значения → суффиксы | Шт. |
|---|---|---|---|
| — | default | `__default` | 1 |
| LOW | PRE | `min`, `max` | 2 |
| LOW | POST | `min`, `max` | 2 |
| LOW | THRES (saturator, A ← → B) | `min`, `max` | 2 |
| LOW | CEIL (saturator) | `min`, `max` | 2 |
| LOW | ATT | `min`, `max` | 2 |
| LOW | CURVE (1…8) | `1`, `8` | 2 |
| LOW | REL (0 = выкл) | `0ms`, `max` | 2 |
| LOW | REL2 (0 = выкл) | `0ms`, `max` | 2 |
| LOW | SUSTAIN (0…1000 ms) | `0ms`, `1000ms` | 2 |
| LOW | PEAK/RMS (кроме дефолта) | `rms` / `peak` | 1 |
| LOW | режим: COMP OFF / MUTED / OFF / SOLO | `compoff`, `muted`, `off`, `solo` | 4 |
| LOW | Stereo separation (центр = выкл) | `min`, `max` | 2 |
| MID, HIGH, MASTER | PRE `max`, ATT `max`, режим `muted` (MASTER: вместо muted ставим `compoff`) | `mid_pre__max` … | 9 |
| MID | Stereo separation | `max` | 1 |
| — | выбор полосы (только клик MID) | `ui_band__mid` | 1 |
| Crossover | LOW (частота) | `min`, `max` | 2 |
| Crossover | HIGH (частота) | `min`, `max` | 2 |
| Crossover | Low / High Band Slope 12↔24 (кроме дефолта) | `lowslope__24`, `highslope__24` | 2 |
| Crossover | LIN | `on` | 1 |
| Global | LOW CUT (дефолт 20 Hz) | `min`, `max` | 2 |
| Global | LMH DEL | `min`, `max` | 2 |
| Global | LMH MIX (0…100%) | `0pct`, `50pct` | 2 |
| Огибающая | LOW: тулбар → Flip vertically | `low_curve__flip` | 1 |
| Огибающая | LOW: сдвинуть одну точку с Ctrl, текст hint в notes | `low_curve__pt1_moved` | 1 |
| Меню | Oversample / Master Mid mode / 0.2 dB safety (каждый отдельно) | `oversample__on`, `mastermid__on`, `safety02__on` | 3 |
| — | combo: LOW PRE 3 dB, LOW ATT и REL на заметных значениях, LMH MIX 50%, crossover LOW из hint | `combo__1` | 1 |

Кривая хранится точками огибающей, а не ручками. Скорее всего, это переменный по длине список (тип точки
Single/Double/Hold/Stairs/Smooth stairs). Двух пресетов хватит, чтобы найти начало списка и формат точки.

## 5. Fruity Parametric EQ 2 (`eq2`), 12 шт.

Основная карта уже есть в `data/raw/presets/decoded/eq2.json`. Эти пресеты закрывают 6 открытых вопросов.
Частоту и ширину ставьте с Ctrl (тонко), пока hint bar не покажет нужное число, или через Type in value,
если оно доступно. Ширину (BW) записывайте **ровно так, как её показывает hint bar** (октавы или Q).

| Вопрос | Пресет (полоса и действие) | Файл | Шт. |
|---|---|---|---|
| — | свежий экземпляр | `eq2__default` | 1 |
| знак order | полоса 1 → High Pass, 1000 Hz, order 2 (база) | `eq2__b1_hp1k__order2` | 1 |
| | то же, Steep 8 (тянуть точки вниз) | `eq2__b1_hp1k__steep8` | 1 |
| | то же, Gentle 8 (тянуть точки вверх) | `eq2__b1_hp1k__gentle8` | 1 |
| формула ширины | полоса 4 → Peaking, 1000 Hz, +6 dB, BW → Q 1.0 | `eq2__b4_peak1k__q1d0` | 1 |
| | то же, Q 4.0 | `eq2__b4_peak1k__q4d0` | 1 |
| Main level | ЛКМ вне полос, Main −6 dB | `eq2__main__m6db` | 1 |
| маска полос | полоса 3 выключена (ЛКМ по Disable switch) | `eq2__b3__muted` | 1 |
| LIN | LIN вкл | `eq2__lin__on` | 1 |
| ширина полки | полоса 7 → High Shelf, 8 kHz, +6 dB, BW = A (записать hint) | `eq2__b7_hshelf8k__bw_a` | 1 |
| | то же, BW = B (сильно другая) | `eq2__b7_hshelf8k__bw_b` | 1 |
| проверка | b2 Low Shelf 120 Hz −3 dB, b5 Peak 3 kHz +2 dB Q 2, Main −2 dB | `eq2__combo__1` | 1 |

Если hint bar не показывает Q, а показывает октавы: Q 1.0 ≈ 1.39 окт, Q 4.0 ≈ 0.36 окт. Ставьте октавы
и пишите их в notes.

## 6. Fruity Chorus (`chorus`), 25 шт.

| Ручка | Значения → суффиксы | Шт. |
|---|---|---|
| default | `__default` | 1 |
| Delay (0…30 ms) | `0ms`, `30ms`, `10ms` | 3 |
| Depth (0 = статично) | `0`, `max` | 2 |
| Stereo (0…360°) | `0deg`, `360deg`, `90deg` | 3 |
| LFO 1 / 2 / 3 Freq | `lfo1_freq__min/max`, `lfo2_…`, `lfo3_…` | 6 |
| LFO 1 Wave (все не-дефолтные варианты, в т.ч. Multi-sine) | `lfo1_wave__<имя>` | ~3 |
| LFO 2 / 3 Wave (один другой вариант) | `lfo2_wave__<имя>`, `lfo3_wave__<имя>` | 2 |
| Cross Type (Process HP ↔ Process LP, кроме дефолта) | `crosstype__lp` / `__hp` | 1 |
| Cross Cutoff | `min`, `max` | 2 |
| Wet Only | `on` | 1 |
| combo: Delay 12 ms, Stereo 180°, Depth на заметном значении, Wet Only on | `combo__1` | 1 |

## 7. Fruity Flangus (`flangus`), 18 шт.

| Ручка | Значения → суффиксы | Шт. |
|---|---|---|
| default | `__default` | 1 |
| ORD (order) | `min`, `max` | 2 |
| DEPTH | `min`, `max` | 2 |
| SPD (speed) | `min`, `max` | 2 |
| DEL (delay) | `min`, `max` | 2 |
| SPRD (spread) | `min`, `max` | 2 |
| CROSS (−100…100) | `m100`, `100` | 2 |
| DRY (−100…100) | `m100`, `0` | 2 |
| WET (−100…100) | `m100`, `0` | 2 |
| combo: ORD на 2 шага от дефолта, CROSS 50, DRY 100, WET 50 | `combo__1` | 1 |

## 8. Fruity Multiband Compressor (`mbcomp`), 30 шт.

LOW делаем полностью, у MID и HIGH по 2 пресета (для stride).

| Секция | Ручка | Значения → суффиксы | Шт. |
|---|---|---|---|
| — | default | `__default` | 1 |
| Global | IIR/FIR (кроме дефолта) | `fir` / `iir` | 1 |
| Global | LIMITER | `on` | 1 |
| Global | 4 cutoff (LP upper, Mid lower, Mid upper, HP lower) | `cut1__min/max` … `cut4__min/max` | 8 |
| LOW | A/M/B | `muted`, `bypass` | 2 |
| LOW | Threshold (0…−60 dB) | `m60db`, `m20db` | 2 |
| LOW | Ratio (1:1…∞:1) | `max`, `4to1` | 2 |
| LOW | KNEE (0…100%) | `0pct`, `100pct` | 2 |
| LOW | Attack (0.1…100 ms) | `0d1ms`, `100ms` | 2 |
| LOW | Release (10…1000 ms) | `10ms`, `1000ms` | 2 |
| LOW | Gain (−∞…17 dB) | `min`, `17db` | 2 |
| MID, HIGH | Threshold `m20db`, Gain `17db` | `mid_thresh__m20db` … | 4 |
| — | combo: LOW Thr −24 dB / Ratio 3:1 / Att 10 ms / Rel 200 ms, MID Gain 3 dB, LIMITER on | `combo__1` | 1 |

---

## После сессии

- Проверьте, что в каждой папке есть `__default.fst`, а в `notes.txt` есть строка на каждый файл.
- Пресет, сохранённый с ошибкой, **перезапишите** под тем же именем. Не оставляйте лишних файлов.
- Дальше дифф по блобу и декодеры, как для EQ 2 (`data/raw/presets/decoded/`).

## Источники

- [Fruity Limiter](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/Fruity%20Limiter.htm)
- [Fruity Reeverb 2](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/Fruity%20Reeverb%202.htm)
- [Fruity Delay 3](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/Fruity%20Delay%203.htm)
- [Maximus Controls](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/Maximus_Controls.htm)
- [Fruity Parametric EQ 2](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/Fruity%20Parametric%20EQ%202.htm)
- [Fruity Chorus](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/Fruity%20Chorus.htm)
- [Fruity Flangus](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/Fruity%20Flangus.htm)
- [Fruity Multiband Compressor](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/Fruity%20Multiband%20Compressor.htm)
- [Wrapper: Presets → Save preset as](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/wrapper.htm)
- [Basics: Type in value, Alt+ЛКМ reset](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/basics_interface.htm)
- [Форум: Alt+click на части ручек сбрасывает к центру](https://forum.image-line.com/viewtopic.php?p=1433048)
