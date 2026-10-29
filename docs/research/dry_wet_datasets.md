# Реальные пары «сухой / обработанный» вокал: где взять ground truth

Собрано 2026-10-29 веб-поиском. Ничего не скачивалось и не регистрировалось. Часть страниц (cambridge-mt.com
отдает 403 боту, PDF MoisesDB) не открылась, поэтому такие пункты помечены «не проверено».

**Зачем это нужно.** Сейчас мы меряем модель на синтетических случайных цепочках, и метрика расходится со слухом.
Нам нужен один и тот же дубль в двух версиях: сырой и после звукорежиссера. Лучше всего, если обработанная версия —
отдельный вокальный стем, а не весь микс.

---

## 0. Что уже лежит локально (`data/raw/dry_vocals/cambridge`, 770 файлов)

Cambridge-MT («Mixing Secrets») по правилам библиотеки раздает **сырые** мультитреки: ZIP с WAV
«without any effects or processing». Обработанного вокального стема там обычно нет. Финальный стерео-микс обычно
дается отдельно как MP3-превью на странице проекта, в ZIP его нет (страницу открыть не удалось, это не проверено).
Бывают исключения, когда автор кладет в мультитрек напечатанную обработку. В нашей выборке нашлись такие:

| Файлы | Что это | Годится? |
|---|---|---|
| `Albert_Kader_Whiptails_11_Vox_Dry.wav` / `12_Vox_Wet.wav` | настоящая пара dry/wet от автора. Wet вдвое больше, вероятно стерео | ★★★: первая реальная пара, проверить сегодня |
| `oxeye_Sorry_12..23_*` и `*_NoPitchProcessing` | 6 пар: вокал с питч-коррекцией и без нее (лид, дабл, 4 бэка) | ★★: эталон только для тюна, но это наш спорный этап («autotune шкурит») |
| `The_Complainiacs_Etc_12_LeadVox` / `13_LeadVoxRoom` | второй (комнатный) микрофон, а не обработка | ✗ |

Стоит прогнать по всем скачанным ZIP Cambridge (не только по вокальным файлам) поиск по именам
`Wet|FX|Print|Processed|Tuned|Verb|Delay|NoPitch`, а также посмотреть «Stems» и «Full Mix», если такие файлы есть.

---

## 1. Датасеты и источники

Обозначения: **wet** — что именно обработано (вокальный стем или полный микс). Оценка ★★★ / ★★ / ★ — насколько
источник полезен именно нам: вокал, жанр, отдельный wet-стем, доступность.

| Источник | Ссылка | Лицензия | Пар вокала | Жанр | Wet | Доступ | Размер | Нам |
|---|---|---|---|---|---|---|---|---|
| **MedleyDB 1.0 + 2.0** (raw vs stems) | https://medleydb.weebly.com/description.html · Zenodo: https://zenodo.org/records/1649325 , https://zenodo.org/record/1715175 | CC BY-NC-SA 4.0 | ≈76 треков, где вокальный стем получен из одного raw (отбор DiffVox), из них 70 годных; всего 196 мультитреков | singer/songwriter, pop, rock, **rap** (немного), jazz, classical | **вокальный стем** (EQ/комп/ревер/панорама инженера); raw моно, stem стерео | форма запроса на Zenodo, только некоммерческое использование | десятки ГБ (точно не указано) | **★★★** |
| **DiffVox presets** (Sony / QMUL, DAFx 2025) | https://huggingface.co/datasets/aim-qmul/diffvox · код https://github.com/SonyResearch/diffvox · статья https://arxiv.org/abs/2504.14735 | MIT | 435 пресетов (365 Internal + 70 MedleyDB) + 20 тестовых. **Аудио нет**, только параметры EQ/DRC/delay/reverb | Internal: «modern mainstream Western» | — (параметры, подогнанные под реальные пары) | открыто, без регистрации | МБ | **★★★** как prior и распределение реальных цепочек; для аудио нужен MedleyDB |
| **MUSDB18-HQ** ↔ сырые треки Cambridge/MedleyDB | https://zenodo.org/records/3338373 (HQ) · https://sigsep.github.io/datasets/musdb.html | для учебных и исследовательских целей; 100 треков взяты из Mixing Secrets | до ~100 (DSD100-часть) + 46 из MedleyDB | rock/pop/electronic/rap | **vocals-стем** = сумма *всех* вокалов (лид + бэки + общий ревер); mixture = сумма стемов | Zenodo, без формы (проверить) | ~30 ГБ | **★★**: пару надо собирать самим — Cambridge raw LeadVox + MUSDB vocals, выравнивать по кросс-корреляции; бэки мешают |
| **Mix Rescue (Sound On Sound) / Cambridge** с проектом Reaper | https://www.soundonsound.com/techniques/mix-rescue-media · проекты в библиотеке Cambridge, напр. `#JamesMay` | для учебных целей, без коммерции | десятки проектов, у части есть **полный проект ремикса Reaper** от Майка Сеньора | rock/pop/indie | можно **самим отрендерить вокальный стем** из проекта (ReaPlugs), параметры известны | бесплатно, без регистрации | 0.3–1 ГБ на проект | **★★**: GT даже по параметрам, но жанр не наш и нужен Reaper |
| **Mix Evaluation Dataset** (De Man & Reiss, DAFx 2017) | https://dafx.de/paper-archive/details/UnYR1t0GX1zguWmAtrpMFg · multitrack.eecs.qmul.ac.uk | CC (по треку) | ~18 песен × 8–10 миксов = 180 миксов **с настройками DAW** | rock/jazz/pop | полный микс, но в сессиях есть настройки плагинов по трекам | частично открыт, часть сессий по запросу | ~ГБ | **★**: полезно для «как разные инженеры обрабатывают один вокал»; рэпа нет |
| **MixologyDB** (JSON-параметры поверх Mix Evaluation) | https://github.com/mclemcrew/MixologyDB · https://huggingface.co/datasets/mclemcrew/MixologyDB | MIT (JSON) | не указано | как выше | параметры EQ/комп по трекам, без аудио | открыто | МБ | ★ |
| **RawStems** (Music Source Restoration, 2025) | https://arxiv.org/abs/2505.21827 · https://huggingface.co/datasets/yongyizang/RawStems | «see original files» (Mixing Secrets) | 578 песен, вокал = сырые треки | разное | **нет wet**: деградации синтетические (EQ, комп, дисторшн, ревер, кодек) | HF, возможно gated | 214 ГБ | ★: это та же Cambridge, только размеченная; удобно как сухие исходники |
| **MSR Challenge / MSRBench** | https://arxiv.org/abs/2601.04343 · https://arxiv.org/html/2510.10995v1 · msrchallenge.com | не проверено | бенчмарк: студийные миксы + сырые референс-стемы | разное | полный микс | регистрация на челлендж (?) | ? | ★ (не проверено) |
| **MoisesDB** | https://music.ai/blog/news/introducing-moisesdb-the-ultimate-multitrack-dataset-for-source-separation-beyond-4-stems · `pip install moises-db` | CC BY-NC-SA 4.0 | 240 песен, 47 артистов, 12 жанров | pop/rock/rap/electronic | треки и стемы — **обработанные** части микса; сырых дублей, похоже, нет (не проверено) | форма на сайте Music.ai | ~80 ГБ | ★: wet без dry; годится только как вокальные референсы |
| **MUSDB18** (без HQ) / **DSD100** | https://sigsep.github.io/datasets/dsd100.html | как MUSDB | см. MUSDB18-HQ | — | AAC 256k | — | — | ✗: брать HQ |
| **SingNet** (2025) | https://arxiv.org/abs/2505.09325 · singnet-dataset.github.io | не проверено | 768 сэмпл-паков размечены на dry/wet, ~3000 ч пения | pop/EDM/rap (сэмпл-паки) | метки dry/wet по клипу, **пары не гарантированы** | релиз аудио не подтвержден | — | ★ |
| **ToneTwist AFx** | https://github.com/mcomunita/tonetwist-afx-dataset | MIT | 0 (гитара и бас) | — | реальные педали и усилители | открыто | — | ✗ |
| **Сэмпл-паки с dry+wet** | DABRO «Hyper Vocals» https://www.loopmasters.com/genres/151-Future-Pop/products/18163-Hyper-Vocals?bid=13913 (10+10, ~$23) · ADSR «Hyperpop Flowers» (4+4, ~$8) · Vocal Roads «Rapping Flow» https://www.loopmasters.com/products/11169 (~$30) · Splice (подписка; во многих вокальных паках есть папки Dry/Wet) · Function Loops «Vocal Pop 2021» (раздавался бесплатно, акция могла закончиться) | royalty-free, использование в музыке; ML/research — проверять EULA | 5–80 на пак | **hyperpop / rap / pop** | **вокальный стем** (хуки, эдлибы, акапеллы) | покупка или подписка | 0.5–2 ГБ | **★★★ по жанру**: единственный источник именно нашего звука; минус — EULA и обработка «под пак», а не под микс |
| **Ремикс-контесты** (Skiomusic, Kreasound, Splice и т.п.) | — | у каждого свои правила | обычно только wet-акапелла, dry бывает редко | pop/rap | вокальный стем | регистрация | — | ★: искать конкурсы, где дают обе версии |
| «Weiss Mix contest» и подобные | — | — | ничего публичного с raw+mixed вокалом не найдено | — | — | — | — | ✗ |

Бесплатных пиратских раздач паков («audioz», «annas archive» и т.п.) в выдаче много — не брать.

---

## 2. Статьи с кодом: оценка вокальных эффектов и перенос стиля

| Работа | Код | На чем оценивали | Можно ли скачать eval |
|---|---|---|---|
| **DiffVox** (Yu et al., DAFx 2025), PEQ + комп + delay + FDN-ревер, подгонка к парам | https://github.com/SonyResearch/diffvox | 70 пар MedleyDB + 365 пар Internal | MedleyDB — по запросу; Internal — нет, выложены только пресеты |
| **ST-ITO + Gaussian prior для вокала** (Yu et al., WASPAA 2025) | демо https://yoyolicoris-diffvox-ito.hf.space/ (raw vocal + reference) ; статья https://arxiv.org/abs/2505.11315 | MedleyDB vocals, слушательский тест (16 чел.) | через MedleyDB. Демо — готовый бейзлайн, с которым можно сравниться на тех же парах |
| **ST-ITO** (Steinmetz et al., ISMIR 2024) | статья https://arxiv.org/abs/2410.21233 | бенчмарк из 4 частей, в основном синтетика | частично |
| **DeepAFx-ST** (Steinmetz et al., ICASSP 2022) | https://github.com/adobe-research/DeepAFx-ST | речь (VCTK/DAPS) + музыка MTG-Jamendo, **синтетические** цели | да, но цели синтетические, как у нас |
| **Diff-MST** (Vanka et al., ISMIR 2024) | https://sai-soum.github.io/projects/diffmst/ , эффекты в https://github.com/csteinmetz1/dasp-pytorch | MedleyDB (196) + Cambridge (535) multitrack → mix, референсы MUSDB18 | нет; данные собираются из публичных наборов, но уровень — полный микс |
| **Mixing graphs pruning / GRAFx** (Lee et al., DAFx 2024) | https://sh-lee97.github.io/grafx-prune · GRAFx (библиотека) | MedleyDB 188, MixingSecrets 472, Internal 579 — только dry-треки + финальный микс | графы и примеры на странице; Internal — нет |
| **Music mixing style transfer / FXencoder** (Koo et al., ICASSP 2023) | https://github.com/jhtonyKoo/music_mixing_style_transfer | MUSDB18 с нормализацией эффектов | да (MUSDB) |
| **Blind estimation of audio effects** (Peladeau & Peeters, 2023) | arXiv https://arxiv.org/abs/2310.11781 | синтетика и реальные пары (не вокал) | — |

Общий вывод литературы (Lee et al. 2024 прямо пишет): **публичного датасета с dry-вокалом и wet-вокальным стемом,
кроме MedleyDB, нет**. Все крупные работы с реальными парами (Sony Internal) держат аудио закрытым.

---

## 3. Рекомендация: топ-3

1. **MedleyDB 1.0 + 2.0 (raw → vocal stem) вместе с пресетами DiffVox.** Это единственный публичный набор реальных
   пар «сырой дубль → вокальный стем после инженера», и на нем же оценивают DiffVox и ST-ITO. Значит, наш результат
   можно напрямую сравнить с опубликованными. Жанр в основном не наш (немного rap/pop), около 70 пар.
   *Что сделать пользователю:* на Zenodo нажать «Request access» для обеих записей (MedleyDB Audio и MedleyDB 2.0 Audio),
   указать некоммерческую исследовательскую цель и дождаться одобрения (обычно дни). Пресеты DiffVox и код открыты,
   ничего делать не нужно. Отбор 70 треков есть в скриптах репозитория DiffVox.
2. **Сэмпл-паки с dry+wet в нашем жанре** (DABRO «Hyper Vocals», ADSR «Hyperpop Flowers», Vocal Roads «Rapping Flow»,
   Splice «UK Rap & Grime Vocals»). Это единственный источник hyperpop/rap-обработки «как у людей», он и пойдет на
   слуховую проверку. Пар мало (10–40 на пак), но для эталонного набора на слух этого хватит.
   *Что сделать:* купить 1–2 пака (~$8–30) или взять месяц Splice. Перед использованием прочитать EULA: обычно запрещено
   распространять сами сэмплы, но не внутренний анализ; паки не выкладывать в репозиторий.
3. **То, что уже есть, плюс Cambridge/MUSDB18-HQ.** Сразу: `Albert_Kader_Whiptails` Vox_Dry/Vox_Wet и 6 пар
   `oxeye_Sorry` (тюн / без тюна). Дальше прогнать имена во всех ZIP Cambridge по ключам Wet/FX/Print/NoPitch.
   Как расширение — MUSDB18-HQ: для ~100 треков Cambridge есть общий вокальный стем после микса. Пару
   «LeadVox raw → vocals stem» собрать самим с выравниванием и брать только треки без бэк-вокала. Проекты
   Mix Rescue с полным проектом Reaper дают даже параметры цепочки.
   *Что сделать:* для MUSDB18-HQ скачать с Zenodo (~30 ГБ, учебная лицензия); для Mix Rescue нужен установленный
   Reaper, чтобы отрендерить вокальную шину.

Не тратить время: ToneTwist (нет вокала), MUSDB18 без HQ (AAC), RawStems как источник wet (там нет wet),
MoisesDB как пары (нет dry).
