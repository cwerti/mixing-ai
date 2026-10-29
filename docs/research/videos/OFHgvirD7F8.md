# СВЕДЕНИЕ В СТИЛЕ MADK1D – SEXYSWAG2010! + ПРЕСЕТ

- Ссылка: https://www.youtube.com/watch?v=OFHgvirD7F8
- Канал: misssantrophy («мизантроп»), 2025-03-24, 2:21
- Трек-референс: madk1d — «SEXYSWAG2010» (из альбома sexyswag; в кадре текст «этот свэг малыш 2010, они себя продают»)
- Источники: кадры 1080p, плагины вырезаны из исходного разрешения (`frames/`; кропы в scratchpad).
  Речь — автосабы YouTube ru (`autosubs.txt`), whisper не делался (сеть). Автор быстро перечисляет
  плагины и почти не объясняет.
- DAW: FL Studio. Пресет — через TG (QR-код в конце).

## Маршрутизация (речь 01:22–01:45, имена каналов в микшере)

`SeXy2010` (основной вокал, фейдер −1.9) → посыл на `SeXydoubler2010` (фейдер −13.5, речь: «громкость почти в ноль»)
→ с него посыл на `SeXyFRESHAIR2010` (фейдер −7.7) → все идут в `SeXyMASTER2010` (мастер вокала) → Master.

## Основной вокал SeXy2010 (список слотов, кадр 01:01)

| # | Плагин | Ключевые настройки | Время | Источник |
|---|---|---|---|---|
| 1 | FabFilter Pro-Q 3 | Шкала 30 dB. Крутой HPF ≈160 Гц; ≈−1.5 дБ около 600 Гц; ≈+3 дБ около 1.3 кГц; узкий колокол ≈+7 дБ около 2.8 кГц; **очень большой подъем ≈+15 дБ около 10 кГц**; крутой LPF ≈18–19 кГц. Значения сняты по сетке | 01:01 | кадр |
| 2 | Antares Auto-Tune Artist | Advanced, Input Type Instrument, Key F, Scale Major, Retune Speed 2, Flex-Tune 0, Humanize 0, Natural Vibrato 0.0, Transpose 0, Detune 440 Hz, Tracking 50 | 01:03 | кадр |
| 3 | Waves RVox | Gain −0.9, Output −1.0; значение Comp на круглом регуляторе частично скрыто («…7») | 01:04 | кадр |
| 4 | Waves CLA-76 | Preset «Start Me Up»; Input ≈24, Output ≈18, Attack ≈5, Release 7 (самый быстрый), Ratio 4, Revision Blacky; Mix около максимума | 01:06 | кадр |
| 5 | Waves CLA-2A | Gain ≈40; Peak Reduction не попал в кадр; Compress | 01:07 | кадр |
| 6 | Waves DeEsser | Настройки не читаются | 01:07 | кадр (слот) |
| 7 | Waves SSL G-EQ (SSLEQ) | HP ≈80–120 Гц; LF 0; LMF ≈+1–2 дБ около 1 кГц; HMF ≈+3–4 дБ около 3 кГц; HF ≈+4–5 дБ около 8 кГц; Analog ON. По положениям ручек, без цифр | 01:09 | кадр |
| 8 | Waves GTR Stomp 2 — OverDrive | Drive низко (≈8 часов), Tone ≈1 час, Level ≈2 часа | 01:10–01:13 | кадр + речь: «за счет него создается эффект голоса мэдкида» |
| 9 | Fruity Chorus | Delay 5.87 ms, Depth 4.02 ms, Stereo 40°, LFO1 0.45 Hz, LFO2 1.31 Hz, LFO3 3.11 Hz (все sin), Cross type Process HF, Cross cutoff 320 Hz, Wet only — no | 01:14–01:18 | кадр + речь: «дополняет, звучит красочнее и ярче» |
| 10 | Valhalla VintageVerb | Preset «Dark Vocal Room», Mode Room, Color 1970s, **Mix 17.2 %**, Predelay 9.00 ms, Decay 1.60 s; HighFreq 8000 Hz, HighShelf −12 dB, BassFreq 600 Hz, BassMult 1.00x; Size 50 %, Attack 18.8 %; Early 49.2 %, Late 80 %; Mod Rate 0.80 Hz, Depth 0 %; HighCut 15000 Hz, LowCut 10 Hz | 01:21 | кадр + речь («Mix 17 %») |

## Посыл SeXydoubler2010 («дабл», уровень почти в ноль)

| # | Плагин | Ключевые настройки | Время | Источник |
|---|---|---|---|---|
| 1 | Waves Doubler2 Stereo | Голос 1: Pan −45, Delay 9.4 ms, Detune +6 центов; голос 2: Pan +45, Delay 23.7 ms, Detune −6 центов; Fdbk 0, Mod Depth 0; Direct 0.0; EQ дабла 0.0 dB @ 101 Hz, +4.1 dB @ 1306 Hz. Значения совпадают с Doubler2 в Nz1LJp5vYyk — вероятно, это дефолтный пресет плагина | 01:25 | кадр |
| 2 | ValhallaRoom | Large Room, Mix 100 %, Predelay 10 ms, Decay 2.00 s, High Cut 8000 Hz, Depth 50 %; Late Size 0.50, Late Cross 1.00, Mod Rate 0.50 Hz, Mod Depth 0.50, Bass Mult 1.00x, Bass Xover 1000 Hz, High Mult 0.50x, High Xover 8000 Hz | 01:28 | кадр |

## Посыл SeXyFRESHAIR2010

| # | Плагин | Ключевые настройки | Время | Источник |
|---|---|---|---|---|
| 1 | FabFilter Pro-Q 3 | Узкий вырез ≈−9 дБ около 105 Гц; широкий ≈−8.5 дБ около 230 Гц; ≈+3.5 дБ около 850 Гц; ≈+4.5 дБ около 4.5 кГц; полка ≈−3 дБ от 9.5 кГц; крутой LPF ≈18 кГц | 01:36 | кадр |
| 2 | Slate Digital Fresh Air | Mid Air 50, High Air 75, Trim 0 | 01:39 | кадр + речь: «верх сильно поднял, мид наполовину» |

## Мастер вокала SeXyMASTER2010

| # | Плагин | Ключевые настройки | Время | Источник |
|---|---|---|---|---|
| 1 | FabFilter Pro-MB | Linear Phase, 4 полосы: <≈200 Гц сжатие ≈−6 дБ; ≈250 Гц–2 кГц сжатие ≈−6 дБ; ≈3.5–6 кГц статически ≈+1.5 дБ, динамически ≈−2 дБ; ≈6–20 кГц статически ≈+2 дБ, динамически ≈−4 дБ; Attack и Release близко к минимуму | 01:45 | кадр |
| 2 | FabFilter Saturn 2 | Warm Tape, Drive высоко; в нижней строке «2 %» (возможно, общий Mix — не уверен) и −1.0 dB out | 01:46–01:48 | кадр |
| 3 | FabFilter Pro-L 2 | Modern, Gain +0.0, True Peak, Out 0.0 dBTP — по сути страховка | 01:49 | кадр |
| 4 | FabFilter Pro-Q 3 | Узкий вырез ≈−6.5 дБ около 180 Гц; динамические провалы ≈−3 дБ около 480 Гц, 2 кГц, 8 кГц; ≈−1 дБ около 900 Гц; полка ≈+3 дБ от ≈13 кГц | 01:50 | кадр |
| 5 | Fruity Soft Clipper | Настройки по умолчанию (по кривой) | 01:52 | кадр |

## Автотюн

Auto-Tune Artist, Retune Speed 2 (почти мгновенный), Flex-Tune 0, Humanize 0, F Major, Input Instrument:
жесткий «роботизированный» тюн. Стоит вторым, после EQ, до компрессии.

## Даблы / бэки

Отдельных дорожек дабла нет: «дабл» делается искусственно (Doubler2 ±45, ±6 центов, 9/24 мс) на посыле
почти без громкости, с большим реверберационным Room 2 с.

## Советы из речи

- Перегруз GTR OverDrive создает «эффект голоса мэдкида» (01:08–01:13).
- Хорус делает голос «красочнее и ярче» (01:13–01:19).
- Fresh Air: верх сильно, мид наполовину (01:36–01:42).

## Неразборчиво

- DeEsser и Peak Reduction CLA-2A не попали в кадр.
- RVox: значение Comp перекрыто.
- Saturn 2 на мастере: «2 %» — не уверен, что это Mix.
- EQ значения сняты по сетке, точных цифр в кадре нет.
