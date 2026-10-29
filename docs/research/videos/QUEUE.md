# Очередь роликов для транскрипции и покадрового разбора

Собрано 2026-10-29 веб-поиском (ничего не скачивалось). Названия и каналы ниже сверены через YouTube oEmbed.
Если стоит «канал не подтвержден», oEmbed не ответил: ролик может быть удален или у него отключено встраивание,
поэтому сначала проверь, открывается ли он. Что именно показано в каждом ролике, известно только из описаний и
сниппетов, цифры настроек еще не проверены. Ролики madk1d из `docs/research/links.md`, раздел 1, сюда
не включены, их разбирает другой агент (туда же относится PryQTaCSxNw).

**Что выписывать при разборе:** плагин → ключевые значения (порог, ratio, атака/релиз, частота/Q/усиление,
speed/formant, wet/dry, время) → порядок в цепочке → сенды/дубли. В колонке «Блок» указано, какой из наших
DSP-модулей проверяет ролик: eq, tune (pitch_corrector), comp, mbc (multiband_compressor), dist, chorus,
stereo, rev, dly, exc, gate.

---

## A. Основы вокальной цепочки в FL Studio (стоковые плагины)

| ★ | Ролик | Ссылка | Яз. | Стиль | Чему учит | Чем полезен | Блок |
|---|---|---|---|---|---|---|---|
| ★★★ | Reid Stefan — How To Mix Vocals in FL Studio: Stock Plugins, EQ, Compression, FX | https://www.youtube.com/watch?v=YeS-i744rmY | EN | trap/pop | EQ → comp → multiband → saturation → reverb/delay на стоковых | Полная стоковая цепь с объяснением «зачем», окна плагинов крупно | eq, comp, mbc, dist, rev, dly |
| ★★★ | Virtual Beats — The Ultimate Stock Vocal Chain | https://www.youtube.com/watch?v=TVrYrlR3x04 | EN | trap/rage | Стоковая цепь от автора бесплатного пресета (см. links.md §3) | Можно сверить кадр и .fst одного автора, то есть проверить и парсер, и цифры | весь стек |
| ★★★ | Sharpe — How To Mix Vocals Like A PRO With STOCK PLUGINS ONLY! | https://www.youtube.com/watch?v=S9ahnYZgNII | EN | rap | Минимальная стоковая цепь, сенды reverb/delay, delay стоит перед reverb | Популярный ролик, в комментариях спор про low cut и 100–200 Гц, это пригодится для границ EQ | eq, comp, rev, dly |
| ★★ | Jordan Baywood — The Secret To Mixing Vocals in FL Studio 20 | https://www.youtube.com/watch?v=NJLx3lRplpM | EN | rap | 9 стоковых плагинов по шагам | Подробная последовательность, хорошо для меток «порядок цепочки» | весь стек |
| ★★ | Ksanx — ЛУЧШЕЕ СВЕДЕНИЕ ВОКАЛА НА СТАНДАРТНЫХ ПЛАГИНАХ В FL STUDIO | https://www.youtube.com/watch?v=PMumfPFKQ-o | RU | рэп/трэп | Стоковая цепь на русском | Русская терминология, значения в кадре | eq, comp, rev |
| ★★ | АЙТИМЕЙК — СВЕДЕНИЕ ВОКАЛА СТАНДАРТНЫМИ ПЛАГИНАМИ FL | https://www.youtube.com/watch?v=hvG9JvKKR08 | RU | рэп | Только встроенные плагины, тюн через Pitcher | Pitcher как стоковый аналог нашего pitch_corrector | eq, comp, tune |
| ★★ | СЭЙМ MUSIC — ТРЕК И СВЕДЕНИЕ НА СТАНДАРТНЫХ ПЛАГИНАХ FL STUDIO + пресеты | https://www.youtube.com/watch?v=ccpL9vT0FNw | RU | новая школа | Стоковое сведение, пресеты в описании | Тот же автор, что и разбор Kai Angel / 9mice (раздел C) | весь стек |

### A2. Отдельные приемы: компрессия, де-эссинг, параллель, сенды, автоматизация

| ★ | Ролик | Ссылка | Яз. | Стиль | Чему учит | Чем полезен | Блок |
|---|---|---|---|---|---|---|---|
| ★★★ | Production Den — VOCAL COMPRESSION in FL Studio: Fruity Limiter on Vocals | https://www.youtube.com/watch?v=vWRfspZqLig | EN | общий | Вкладка COMP в Fruity Limiter: порог, ratio, атака, релиз, knee | Самый частый «компрессор» в любительских .fst, нужен для маппинга его параметров на наш comp | comp |
| ★★ | АЙТИМЕЙК — КОМПРЕССИЯ ВОКАЛА В FL STUDIO | https://www.youtube.com/watch?v=G-dbRo-JjuU | RU | рэп | Компрессия вокала стоковыми средствами | Русская версия предыдущего ролика | comp |
| ★★ | Михаил Кириленко — Как настроить КОМПРЕССОР? АТАКА и РЕЛИЗ | https://www.youtube.com/watch?v=IyUrrjPgk_o | RU | общий | Как атака и релиз меняют звук | Объясняет, почему компрессор настроен именно так, это пригодится для разумных диапазонов поиска | comp |
| ★★ | MusicTech — FL Studio Tips: De-essing in FL Studio | https://www.youtube.com/watch?v=-3LyS5syYiA | EN | общий | Де-эссер на Maximus, потому что в FL нет отдельного де-эссера | Де-эссер в .fst обычно выглядит как Maximus или multiband, поэтому ролик показывает, как такой пресет читать | mbc |
| ★★ | Internet Money — Parallel Compression Tutorial (FL Studio) | https://www.youtube.com/watch?v=gfqQ8eth2fM | EN | trap | Параллельная компрессия через сенд | Известный канал. У нас нет параллельного блока, ролик поможет решить, нужен ли он | comp (параллель) |
| ★ | theonlyjonesy — How to Create Vocal Delay Throws (1 min) | https://www.youtube.com/watch?v=2Kmybz5-u0I | EN | rap | Delay throw с автоматизацией сенда | Короткий ролик. Показывает, что delay бывает не статичным, и это ограничивает нашу модель с фиксированным wet | dly |

---

## B. Автотюн, хард-тюн, формант

| ★ | Ролик | Ссылка | Яз. | Стиль | Чему учит | Чем полезен | Блок |
|---|---|---|---|---|---|---|---|
| ★★★ | Brandon Buhr Music — How to Auto-Tune Vocals using Pitcher in FL Studio (2026 Update) | https://www.youtube.com/watch?v=SmYiSs0QqC0 | EN | общий | Pitcher: тональность, speed, fine-tune | Свежий ролик, окно Pitcher видно целиком, по нему маппим speed Pitcher на retune нашего tune | tune |
| ★★★ | Brandon Buhr Music — How to Install and Use Graillon 3 Autotune in FL Studio | https://www.youtube.com/watch?v=plSLBvmIriw | EN | общий | Graillon 3: correction, движки G2/G3/I1, формант | Graillon часто стоит в бесплатных пресетах новой школы | tune (+формант) |
| ★★ | (FromIHateWaterr) — How to Sound Like PLAYBOI CARTI "MEH" Baby Voice, FL 20 | https://www.youtube.com/watch?v=_yBN1SUskU4 | EN | rage | Baby voice: питч и формант вверх | Детский голос поверх тюна, это граница того, что может наша палитра | tune, формант |
| ★★ | Bxllistic Will — Playboi Carti vocal effects breakdown + preset (FL) | https://www.youtube.com/watch?v=NQU6eBTGJXc | EN | rage | Разбор эффектов Carti с пресетом | Кадр и пресет от одного автора | tune, dist, dly |
| ★ | PAVMusicProduction — Aggressive/Hard Rap Vocal Effects Settings in FL Studio (+Free Presets) | https://www.youtube.com/watch?v=YUKb-9OX9n0 | EN | агрессивный рэп | Hard-tune и дисторшн для агрессивной читки | В названии обещаны settings, канал не подтвержден | tune, dist |

---

## C. Русская новая школа (кроме madk1d)

| ★ | Ролик | Ссылка | Яз. | Стиль | Чему учит | Чем полезен | Блок |
|---|---|---|---|---|---|---|---|
| ★★★ | СЭЙМ MUSIC — СВЕДЕНИЕ В СТИЛЕ Kai Angel & 9mice / FL Studio + Пресеты | https://www.youtube.com/watch?v=UmZp-bxmmno | RU | rage / новая школа | Цепочка под Kai Angel и 9mice | Самый близкий к madk1d стиль (один круг артистов), пресет в описании | tune, dist, rev, dly |
| ★★★ | silverxhaze live — МЕНЯ ВЗЯЛИ В VIPER! // СВЕДЕНИЕ KAI ANGEL & 9MICE "ORPHEUS" | https://www.youtube.com/watch?v=zfSO7kw0nJU | RU | rage | Сведение под конкретный альбом ORPHEUS, пресет | Привязано к конкретному треку («FACE»), значит можно сравнить с референсом | весь стек |
| ★★ | trybiq — СВЕДЕНИЕ в Стиле CUPSIZE | https://www.youtube.com/watch?v=chuJfNUNE2w | RU | поп-панк / гиперпоп | Цепочка под CUPSIZE | CUPSIZE назван эталоном жесткого стиля в links.md §2 | tune, dist, exc |
| ★★ | сведение в стиле cupsize + пресет | https://www.youtube.com/watch?v=hX83t2nhm2E | RU | гиперпоп | То же, пресет в TG | Второй источник по CUPSIZE для сравнения. Канал не подтвержден | tune, dist |
| ★★ | ДЖИМИ — СВЕДЕНИЕ В СТИЛЕ ПЛАТИНА (99.69% СХОЖЕСТИ) | https://www.youtube.com/watch?v=Qcl9qTJpJsM | RU | автотюн-трэп | Сведение под новый альбом Платины | Мягкий автотюн-трэп, контрастный пример к rage | tune, rev, dly |
| ★★ | ОБРАБОТКА ВОКАЛА В СТИЛЕ ПЛАТИНЫ В FL STUDIO 20 + ПРЕСЕТЫ | https://www.youtube.com/watch?v=RRRUuDv4W9s | RU | автотюн-трэп | По словам автора, вся автотюн-школа сведена так же | Обобщенный рецепт. Канал не подтвержден | tune, comp, rev |
| ★★ | VIRTAL — СВЕДЕНИЕ ВОКАЛА В СТИЛЕ OG BUDA *мелодичный Буда* + ПРЕСЕТЫ | https://www.youtube.com/watch?v=f3ObIscwa3E | RU | мелодичный трэп | Мелодичный вокал с тюном, мастеринг | Покрывает OG Buda | tune, rev, dly |
| ★ | ОБРАБОТКА ВОКАЛА В СТИЛЕ MAYOT + ПРЕСЕТЫ | https://www.youtube.com/watch?v=kFbqWuvG4dw | RU | мелодичный трэп | Цепочка под Mayot | Покрывает Mayot. Канал не подтвержден | tune, rev |
| ★ | smxly mute — СВЕДЕНИЕ В СТИЛЕ NEWLIGHTCHILD & ШАЙНИ В FL STUDIO 21 | https://www.youtube.com/watch?v=ObPI2-Y8r2w | RU | гиперпоп / новая школа | Цепочка + пресет | Смежный с LILCAK3 звук (прямых роликов по LILCAK3 не нашлось) | tune, dist |
| ★ | VLAD MINE — СВЕДЕНИЕ ВОКАЛА в FL Studio! Лучшая обработка Голоса (Hyperpop, Trap) | https://www.youtube.com/watch?v=O719_cvpdTU | RU | гиперпоп / трэп | Запись и сведение под hyperpop и trap | Русский гиперпоп-рецепт | весь стек |
| ★ | nice hood — СВЕДЕНИЕ И ОБРАБОТКА ВОКАЛА В СТИЛЕ АНДЕГРАУНД РЭП | https://www.youtube.com/watch?v=uWqtEMRQUhs | RU | андеграунд | Грязный андеграундный вокал | Ближе всего к «рок»-периоду madk1d, где много dist | dist, comp |

---

## D. Hyperpop / rage / sigilkore / pluggnb (англоязычные)

| ★ | Ролик | Ссылка | Яз. | Стиль | Чему учит | Чем полезен | Блок |
|---|---|---|---|---|---|---|---|
| ★★★ | Virtual Beats — How To Mix Rage Vocals With Stock Plugins | https://www.youtube.com/watch?v=rkwRvanKCCc | EN | rage | Rage-цепь только на стоке | Rage на тех же стоковых плагинах, что и наша палитра, плюс бесплатные .fst автора | весь стек |
| ★★★ | Virtual Beats — How To Mix Hyperpop Vocals (FL Studio) | https://www.youtube.com/watch?v=b2HAmbFAbOw | EN | hyperpop | Hyperpop-цепь в FL | Hyperpop от того же автора, удобно сравнить с его rage-цепью | tune, dist, chorus, rev |
| ★★ | Show Your Genius — Hyperpop vocal effect in FL Studio (midwxst, 100 gecs, glaive) | https://www.youtube.com/watch?v=KKhGtauQyvY | EN | hyperpop | Тюн, чопы, дисторшн, формант | Перечисляет все типовые приемы hyperpop | tune, dist |
| ★★ | LTJ — HIGH PITCH HYPERPOP VOCAL TUTORIAL (glaive, kurtains, osquinn) | https://www.youtube.com/watch?v=_Kve5cuWVxA | EN | hyperpop | Питч и формант вверх | Пример экстремального питча за пределами нашего tune | tune, формант |
| ★★ | Bxllistic Will — Destroy Lonely & Ken Carson vocal preset tutorial (FL 21) | https://www.youtube.com/watch?v=he--88X7qN4 | EN | rage (Opium) | Opium-вокал: тюн, дисторшн, дилей | Эталон rage-звука, на который ссылаются русские авторы | tune, dist, dly |
| ★ | overlyvic — I Made a Ken Carson Vocal Preset and It's Insane | https://www.youtube.com/watch?v=RNTCvdHpZDM | EN | rage | Пресет под Ken Carson | Второй источник по Ken Carson | tune, dist |
| ★★ | mixedbymoto — HOW TO PLUGNB VOCALS: 'DETAIL' MIXING BREAKDOWN | https://www.youtube.com/watch?v=TP0cXI1AWbc | EN | pluggnb | Pluggnb: тюн, воздух, широкий дилей | Мягкий полюс стилей | tune, exc, stereo, dly |
| ★ | How To Mix Pluggnb Vocals THE RIGHT WAY (Autumn, Pnb Rock, Koi) [FREE TEMPLATE] | https://www.youtube.com/watch?v=1qVWmeWmwUY | EN | pluggnb | Pluggnb в FL + шаблон | Шаблон годится как тестовые данные. Канал не подтвержден | весь стек |
| ★ | HOW TO MAKE AN AXXTUREL / SIGILKORE TYPE SONG (vocal tutorial, mixing tips, presets) | https://www.youtube.com/watch?v=M8_D0QkxzNE | EN | sigilkore | Sigilkore-вокал: дисторшн, bitcrush, реверб | Единственный найденный ролик по sigilkore, где показано сведение. Канал не подтвержден | dist, rev |
| ★ | MixLikeNix — Create Distorted Ye Rage Vocals with MacDSP FuzzBox | https://www.youtube.com/watch?v=kOqwSAV2CAA | EN | rage (Ye) | Fuzz и клиппинг на вокале | Показывает, как звучит предел нашего distortion | dist |

---

## E. Дубли, адлибы, гармонии, октавы

| ★ | Ролик | Ссылка | Яз. | Стиль | Чему учит | Чем полезен | Блок |
|---|---|---|---|---|---|---|---|
| ★★★ | Make Pop Music — How To Mix Background Vocals Like A Pro (Doubles, Harmonies, FX Vocals, Ad Libs) | https://www.youtube.com/watch?v=NRX4QLcafHQ | EN | pop/rap | 6 принципов: уровень, панорама, EQ и FX для бэков | Главный источник для будущего мульти-трек режима (лид + дубли) | eq, stereo, rev |
| ★★ | Gunnr — How to ACTUALLY Layer YOUR Vocals (Harmonies, Side Vocals, Doubles), FL Studio | https://www.youtube.com/watch?v=v8DL04beUpY | EN | rap/rage | Слои в FL: гармонии, сайд-вокал, дубли | Сделано в FL, видно маршрутизацию | stereo, tune |
| ★★ | АЙТИМЕЙК — СВЕДЕНИЕ И ОБРАБОТКА БЭК-ВОКАЛА В FL STUDIO | https://www.youtube.com/watch?v=DYoEwae_Rfw | RU | рэп | Обработка бэков и даблов | Русскоязычный разбор бэков | eq, stereo, comp |

---

## F. Инженеры объясняют свою логику

| ★ | Ролик | Ссылка | Яз. | Стиль | Чему учит | Чем полезен | Блок |
|---|---|---|---|---|---|---|---|
| ★★ | Sky Jordxn — Alex Tumay Answers STUPID Mixing Questions | https://www.youtube.com/watch?v=o_43pB0tGVg | EN | rap (Young Thug) | Логика инженера Young Thug | Объясняет, почему решения такие, а не цифры. Полезно для априорных диапазонов | — |
| ★★ | Servida Music — The ULTIMATE Guide To Mixing Rap Vocals! | https://www.youtube.com/watch?v=8g7ituDRJa4 | EN | rap | Акустика, EQ, компрессия, де-эссер, сатурация | Хорошо структурирован, каждая стадия с обоснованием | eq, comp, mbc, dist |
| ★ | Jackson Mixed It — Make Rap Vocals Sound EXPENSIVE (Step-by-Step Vocal Chain) | https://www.youtube.com/watch?v=jppFFvbil7s | EN | rap | Профессиональная цепь по шагам | Профессиональная планка для сравнения со стоковыми рецептами | весь стек |

---

## Пробелы

- **LILCAK3, OG Buda и Mayot.** Прямых роликов по LILCAK3 не нашлось. По OG Buda и Mayot нашлись только любительские подражания.
- **Ролики, которые надо проверить.** У части роликов канал не подтвержден (oEmbed не ответил). Их стоит проверить первыми, до транскрипции.
- **Автоматизация.** Ее показывает только delay throws. Наша модель статичная, поэтому эти ролики полезны в основном для оценки потерь.
