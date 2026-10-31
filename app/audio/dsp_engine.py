import numpy as np
from pedalboard import (
    Pedalboard, Compressor, Reverb, Delay, LowShelfFilter, PeakFilter,
    HighShelfFilter, Chorus, Distortion, HighpassFilter, LowpassFilter,
    NoiseGate, Limiter, Gain
)
from app.dataset.dataset_schema import ChainConfig, PluginConfig, DATAGEN_PLUGINS, REVERB_SEND_PARAMS
from app.bridge.vst_parameter_mapper import VSTParameterMapper
from scipy.signal import fftconvolve, sosfilt, butter


def decay_reverb(y: np.ndarray, sr: int, decay_s: float, damping: float, seed: int = 0) -> np.ndarray:
    """
    Мокрый сигнал ревера с явным временем затухания RT60 (Freeverb из pedalboard длиннее ~3 с не тянет,
    а у реальных миксов хвост бывает 4–8 с). Импульсная характеристика — экспоненциально затухающий шум:
    верх (выше ~2 кГц) гаснет быстрее на долю damping, каналы декоррелированы (разный шум) для ширины.
    Энергия ИХ нормирована к 2² — так хвост на том же wet_level звучит примерно как у Freeverb (замер на шуме:
    Freeverb wet=1 громче нормированной ИХ в ~2 раза), и wet_level у обоих реверов значит одно и то же.
    """
    rng = np.random.default_rng(seed)
    # Хвост длиннее сигнала не слышен (свёртка обрезается по его длине), а дальше 1.5·RT60 он ниже -100 дБ
    n = int(min(decay_s * 1.5, y.shape[-1] / sr + 0.1, 30.0) * sr)
    t = np.arange(n) / sr
    hf_decay = decay_s * (1.0 - 0.85 * float(np.clip(damping, 0.0, 1.0)))
    low_sos = butter(2, 2000.0, "low", fs=sr, output="sos")
    high_sos = butter(2, 2000.0, "high", fs=sr, output="sos")
    channels = 1 if y.ndim == 1 else y.shape[0]
    irs = []
    for _ in range(channels):
        noise = rng.standard_normal(n)
        ir = (sosfilt(low_sos, noise) * np.exp(-6.91 * t / decay_s)
              + sosfilt(high_sos, noise) * np.exp(-6.91 * t / max(hf_decay, 0.05)))
        irs.append(2.0 * ir / np.sqrt(np.sum(ir ** 2)))
    if y.ndim == 1:
        return fftconvolve(y, irs[0])[:len(y)].astype(np.float32)
    mono = y.mean(axis=0)
    return np.stack([fftconvolve(mono, ir)[:y.shape[1]] for ir in irs]).astype(np.float32)

class DSPEngine:
    """
    Класс обработки аудиосигналов на основе библиотеки Pedalboard.
    Позволяет применять цепочки эффектов с нормализованными параметрами [0..1],
    а также загружать VST3-плагины или использовать их встроенные оффлайн-аналоги.
    """

    @staticmethod
    def denormalize_value(norm_val: float, min_val: float, max_val: float) -> float:
        """
        Преобразует нормализованное значение [0..1] в физическую величину в диапазоне [min_val..max_val].
        
        Args:
            norm_val: Нормализованное значение от 0.0 до 1.0.
            min_val: Минимальная граница физического диапазона.
            max_val: Максимальная граница физического диапазона.
            
        Returns:
            float: Физическое значение.
        """
        clamped = np.clip(norm_val, 0.0, 1.0)
        return float(min_val + clamped * (max_val - min_val))

    @staticmethod
    def normalize_value(phys_val: float, min_val: float, max_val: float) -> float:
        """
        Преобразует физическое значение в диапазоне [min_val..max_val] в нормализованное [0..1].
        
        Args:
            phys_val: Физическое значение.
            min_val: Минимальная граница физического диапазона.
            max_val: Максимальная граница физического диапазона.
            
        Returns:
            float: Нормализованное значение от 0.0 до 1.0.
        """
        clamped = np.clip(phys_val, min_val, max_val)
        denom = max_val - min_val
        return float((clamped - min_val) / denom if denom > 0 else 0.0)

    def clamp_normalized_params(self, plugin_config: PluginConfig) -> PluginConfig:
        """
        Ограничивает (клемпит) нормализованные параметры плагина [0..1]
        в безопасные профессиональные рамки для предотвращения артефактов.
        
        Args:
            plugin_config: Исходная конфигурация плагина.
            
        Returns:
            PluginConfig: Конфигурация с лимитированными параметрами.
        """
        name = plugin_config.name
        clamped_params = plugin_config.params.copy()
        
        if name == "chorus":
            if "depth" in clamped_params:
                clamped_params["depth"] = min(clamped_params["depth"], 0.15)
            if "mix" in clamped_params:
                # 0.34 при диапазоне mix 0.05–0.45 — те же 0.186 физически, что 0.20 при прежнем 0.12–0.45
                clamped_params["mix"] = min(clamped_params["mix"], 0.34)
            if "rate_hz" in clamped_params:
                clamped_params["rate_hz"] = min(clamped_params["rate_hz"], 0.25)
        elif name == "distortion":
            if "drive_db" in clamped_params:
                # 0.52 при диапазоне 0–20 дБ — те же 10.4 дБ, что 0.40 при прежнем 4–20 дБ
                clamped_params["drive_db"] = min(clamped_params["drive_db"], 0.52)
        elif name == "reverb":
            if "wet_level" in clamped_params:
                # 0.12 при диапазоне 0.06–0.60 — те же 0.126, что 0.22 при прежнем 0.06–0.36
                clamped_params["wet_level"] = min(clamped_params["wet_level"], 0.12)
        elif name == "delay":
            if "mix" in clamped_params:
                clamped_params["mix"] = min(clamped_params["mix"], 0.20)
                
        return PluginConfig(name=name, params=clamped_params)

    def get_physical_params(self, plugin_config: PluginConfig) -> dict:
        """
        Возвращает физические параметры эффекта, преобразованные из нормализованной конфигурации [0..1].
        Для всех стандартных плагинов схемы использует единый источник диапазонов DATAGEN_PLUGINS.
        
        Args:
            plugin_config: Нормализованная конфигурация плагина.
            
        Returns:
            dict: Словарь с физическими параметрами плагина.
        """
        plugin_name = plugin_config.name
        if plugin_name in DATAGEN_PLUGINS:
            ranges = DATAGEN_PLUGINS[plugin_name]["params"]
            phys_params = {}
            for p_name, val in plugin_config.params.items():
                if p_name in ranges:
                    min_v, max_v = ranges[p_name]
                    phys_params[p_name] = self.denormalize_value(val, min_v, max_v)
                else:
                    phys_params[p_name] = val
            return phys_params

        short_names = {
            "eq": "fruity_parametric_eq_2",
            "compressor": "fruity_compressor",
            "reverb": "fruity_reeverb_2",
            "delay": "fruity_delay_3"
        }
        config_name = short_names.get(plugin_name, plugin_name)
        
        try:
            mapper = VSTParameterMapper(config_name)
            phys_params = {}
            for p_name, val in plugin_config.params.items():
                clean_p_name = p_name.replace("_hz", "").replace("_db", "").replace("_ms", "").replace("_seconds", "")
                
                if clean_p_name in mapper.mappings:
                    phys_params[p_name] = mapper.map_normalized_to_physical(clean_p_name, val)
                elif p_name in mapper.mappings:
                    phys_params[p_name] = mapper.map_normalized_to_physical(p_name, val)
                else:
                    phys_params[p_name] = val
            return phys_params
        except Exception as e:
            raise ValueError(f"Плагин '{plugin_name}' не поддерживается DSPEngine или VSTParameterMapper: {e}")

    def apply_chain(
        self,
        y: np.ndarray,
        sr: int,
        chain_config: ChainConfig,
        gate_threshold_db: float = None,
        target_key_index: int = None,
        target_scale: str = None,
        apply_safety_clamps: bool = False,
        final_limiter: bool = True
    ) -> np.ndarray:
        """
        Применяет последовательную цепочку плагинов к аудиосигналу.
        
        Args:
            y: Входной аудиосигнал.
            sr: Частота дискретизации аудиосигнала.
            chain_config: Конфигурация цепочки плагинов и их параметров.
            gate_threshold_db: Порог Noise Gate в дБ. Если None, гейт не применяется.
            target_key_index: Индекс тональности (0-11) для Pitch Corrector.
            target_scale: Лад ('major' или 'minor') для Pitch Corrector.
            apply_safety_clamps: Применять ли дополнительное ограничение параметров (по умолчанию False,
                чтобы при генерации датасета звук на 100% соответствовал метке).
            final_limiter: Ставить ли выходной лимитер −1 дБ. Выключается, когда цепочка рендерится по частям
                (app/audio/effect_variants.py), — там лимитер один, в самом конце.
            
        Returns:
            np.ndarray: Обработанный аудиосигнал.
        """
        effects = []
        if gate_threshold_db is not None:
            effects.append(NoiseGate(threshold_db=gate_threshold_db, attack_ms=2.0, release_ms=100.0))
            
        y_working = y.copy()
        # Лидер, из которого строятся слои гармонии (сигнал на месте первого плагина harmony)
        harmony_source = None
        
        # Применение подавления резонансов (если плагин явно передан в цепочке)
        for plugin in chain_config.plugins:
            if plugin.name == "resonance_suppressor":
                plugin_used = self.clamp_normalized_params(plugin) if apply_safety_clamps else plugin
                phys = self.get_physical_params(plugin_used)
                t_db = phys.get("threshold_db", 7.0)
                att_db = phys.get("max_attenuation_db", 12.0)
                from app.audio.vocal_enhancer import VocalEnhancer
                enhancer = VocalEnhancer(sr=sr)
                y_working = enhancer.suppress_resonances(y_working, threshold_db=t_db, max_attenuation_db=att_db)
                break
        
        short_names = {
            "eq": "fruity_parametric_eq_2",
            "compressor": "fruity_compressor",
            "distortion": "fruity_fast_dist",
            "chorus": "fruity_chorus",
            "reverb": "fruity_reeverb_2",
            "delay": "fruity_delay_3"
        }
        
        for plugin in chain_config.plugins:
            if apply_safety_clamps:
                plugin = self.clamp_normalized_params(plugin)
            config_name = short_names.get(plugin.name, plugin.name)
            
            # Попытка загрузить VST3-плагин напрямую через Pedalboard, если настроен маппер
            vst_loaded = False
            try:
                mapper = VSTParameterMapper(config_name)
                if mapper.vst_type == "VST3":
                    vst_name = mapper.display_name
                    vst3_path = f"C:/Program Files/Common Files/VST3/{vst_name}.vst3"
                    
                    try:
                        from pedalboard import VST3Plugin
                        vst_plugin = VST3Plugin(vst3_path)
                        
                        # Применение нормализованных параметров напрямую на VST3
                        for p_name, val in plugin.params.items():
                            clean_p_name = p_name.replace("_hz", "").replace("_db", "").replace("_ms", "").replace("_seconds", "")
                            param_name = clean_p_name if clean_p_name in mapper.mappings else p_name
                            
                            param_idx = mapper.get_param_index(param_name)
                            if param_idx is not None:
                                vst_plugin.set_parameter(param_idx, val)
                                
                        effects.append(vst_plugin)
                        vst_loaded = True
                        print(f"[+] Успешно загружен VST3-плагин '{vst_name}' в цепь Pedalboard.")
                    except Exception as ex:
                        print(f"[!] Предупреждение: Не удалось загрузить VST3 '{vst_name}' из {vst3_path} ({ex}). Переход на встроенный аналог.")
            except Exception:
                pass
                
            if vst_loaded:
                continue

            # Откат к стандартной встроенной обработке Pedalboard
            phys = self.get_physical_params(plugin)
            
            if plugin.name == "eq" or (vst_loaded is False and plugin.name == "fabfilter_pro_q_3"):
                # Создаем цепочку из 7 фильтров EQ в качестве оффлайн-аналога
                effects.append(LowShelfFilter(
                    cutoff_frequency_hz=phys.get("band1_freq_hz", phys.get("band1_freq", 100.0)),
                    gain_db=phys.get("band1_gain_db", phys.get("band1_gain", 0.0)),
                    q=0.707
                ))
                effects.append(PeakFilter(
                    cutoff_frequency_hz=phys.get("band2_freq_hz", phys.get("band2_freq", 200.0)),
                    gain_db=phys.get("band2_gain_db", phys.get("band2_gain", 0.0)),
                    q=1.0
                ))
                effects.append(PeakFilter(
                    cutoff_frequency_hz=phys.get("band3_freq_hz", phys.get("band3_freq", 500.0)),
                    gain_db=phys.get("band3_gain_db", phys.get("band3_gain", 0.0)),
                    q=1.0
                ))
                effects.append(PeakFilter(
                    cutoff_frequency_hz=phys.get("band4_freq_hz", phys.get("band4_freq", 1200.0)),
                    gain_db=phys.get("band4_gain_db", phys.get("band4_gain", 0.0)),
                    q=1.0
                ))
                effects.append(PeakFilter(
                    cutoff_frequency_hz=phys.get("band5_freq_hz", phys.get("band5_freq", 3000.0)),
                    gain_db=phys.get("band5_gain_db", phys.get("band5_gain", 0.0)),
                    q=1.0
                ))
                effects.append(PeakFilter(
                    cutoff_frequency_hz=phys.get("band6_freq_hz", phys.get("band6_freq", 6000.0)),
                    gain_db=phys.get("band6_gain_db", phys.get("band6_gain", 0.0)),
                    q=1.0
                ))
                effects.append(HighShelfFilter(
                    cutoff_frequency_hz=phys.get("band7_freq_hz", phys.get("band7_freq", 12000.0)),
                    gain_db=phys.get("band7_gain_db", phys.get("band7_gain", 0.0)),
                    q=0.707
                ))
                
            elif plugin.name == "pitch_corrector":
                # Сначала рендерим предыдущие эффекты
                if effects:
                    board = Pedalboard(effects)
                    try:
                        y_working = board(y_working, sr)
                    except Exception as e:
                        print(f"[-] Ошибка обработки Pedalboard перед автотюном: {e}")
                    effects = []
                
                # Если тональность не передана, автодектим тональность исходника, чтобы настроить на самого себя
                from app.audio.vocal_enhancer import VocalEnhancer
                enhancer = VocalEnhancer(sr=sr)
                
                k_idx = target_key_index
                s_type = target_scale
                if k_idx is None or s_type is None:
                    k_idx, s_type = enhancer.detect_key_and_scale(y_working)
                    
                phys = self.get_physical_params(plugin)
                speed_val = phys.get("speed", 0.85)
                y_working = enhancer.apply_autotune(y_working, key_index=k_idx, scale=s_type, speed=speed_val)
                
            elif plugin.name == "compressor" or (vst_loaded is False and plugin.name in ("fabfilter_pro_c_2", "waves_cla_2a")):
                thresh_val = float(phys.get("threshold_db", phys.get("threshold", -20.0)))
                ratio_val = float(phys.get("ratio", 4.0))
                effects.append(Compressor(
                    threshold_db=thresh_val,
                    ratio=ratio_val,
                    attack_ms=phys.get("attack_ms", phys.get("attack", 2.0)),
                    release_ms=phys.get("release_ms", phys.get("release", 100.0))
                ))
                # Автоматический частичный Makeup Gain компенсирует потерю пикового уровня,
                # чтобы порог и степень сжатия однозначно уменьшали Crest Factor!
                auto_makeup_db = float(np.clip((-10.0 - thresh_val) * (1.0 - 1.0 / max(ratio_val, 1.0)) * 0.45, 0.0, 12.0))
                if auto_makeup_db > 0.1:
                    effects.append(Gain(gain_db=auto_makeup_db))
                
            elif plugin.name == "deesser":
                # Сначала рендерим предыдущие эффекты
                if effects:
                    board = Pedalboard(effects)
                    try:
                        y_working = board(y_working, sr)
                    except Exception as e:
                        print(f"[-] Ошибка обработки Pedalboard перед деэссером: {e}")
                    effects = []
                
                # Реализуем Split-Band De-esser (разделяем полосу на 4.2 кГц).
                # Учитываем, что энергия полосы >4.2 кГц на ~14 дБ ниже широкополосного сигнала (-23 LUFS),
                # поэтому подаем ВЧ-полосу в детектор с пред-усилением +14 дБ и обратным ослаблением -14 дБ!
                threshold_db = float(phys.get("threshold_db", -20.0))
                ratio = float(phys.get("ratio", 4.0))
                try:
                    low_board = Pedalboard([LowpassFilter(cutoff_frequency_hz=4200.0)])
                    y_low = low_board(y_working, sr)
                    
                    high_board = Pedalboard([
                        HighpassFilter(cutoff_frequency_hz=4200.0),
                        Gain(gain_db=14.0),
                        Compressor(threshold_db=threshold_db, ratio=ratio, attack_ms=1.5, release_ms=35.0),
                        Gain(gain_db=-14.0)
                    ])
                    y_high_compressed = high_board(y_working, sr)
                    
                    # Суммируем полосы обратно
                    y_working = y_low + y_high_compressed
                except Exception as e:
                    print(f"[-] Ошибка обработки De-esser: {e}")
                    
            elif plugin.name == "multiband_compressor":
                # Сначала рендерим предыдущие эффекты
                if effects:
                    board = Pedalboard(effects)
                    try:
                        y_working = board(y_working, sr)
                    except Exception as e:
                        print(f"[-] Ошибка обработки Pedalboard перед OTT: {e}")
                    effects = []
                
                # Применяем эмулированный OTT-компрессор
                from app.audio.vocal_enhancer import VocalEnhancer
                enhancer = VocalEnhancer(sr=sr)
                phys = self.get_physical_params(plugin)
                depth_val = phys.get("depth", 0.40)
                y_working = enhancer.apply_multiband_compressor(y_working, depth=depth_val)
                
            elif plugin.name == "distortion":
                # Сначала рендерим предыдущие эффекты, чтобы сохранить порядок цепи
                if effects:
                    board = Pedalboard(effects)
                    try:
                        y_working = board(y_working, sr)
                    except Exception as e:
                        print(f"[-] Ошибка обработки Pedalboard перед дисторшном: {e}")
                    effects = []
                
                # Применяем параллельную сатурацию/дисторшн с отчетливо слышимым балансом (22%..38% wet)
                # Отсекаем низ (до 150 Гц) и ультра-верх (выше 5500 Гц) на перегруженном сигнале,
                # чтобы убрать свистящий цифровой песок (fizz) и мутный гул.
                drive_db = phys.get("drive_db", 10.0)
                norm_drive = float(plugin.params.get("drive_db", 0.5))
                wet_mix = 0.22 + 0.16 * np.clip(norm_drive, 0.0, 1.0)
                try:
                    dist_board = Pedalboard([
                        Distortion(drive_db=drive_db),
                        HighpassFilter(cutoff_frequency_hz=150.0),
                        LowpassFilter(cutoff_frequency_hz=5500.0)
                    ])
                    y_dist = dist_board(y_working, sr)
                    y_working = (1.0 - wet_mix) * y_working + wet_mix * y_dist
                except Exception as e:
                    print(f"[-] Ошибка параллельного дисторшна: {e}")
                
            elif plugin.name == "chorus":
                effects.append(Chorus(
                    rate_hz=phys.get("rate_hz", 1.0),
                    depth=phys.get("depth", 0.25),
                    feedback=phys.get("feedback", 0.0),
                    mix=phys.get("mix", 0.5)
                ))
                
            elif plugin.name == "stereo_enhancer":
                # Сначала рендерим предыдущие эффекты, чтобы они применились в моно
                if effects:
                    board = Pedalboard(effects)
                    try:
                        y_working = board(y_working, sr)
                    except Exception as e:
                        print(f"[-] Ошибка обработки Pedalboard перед стерео-расширителем: {e}")
                    effects = []
                
                # Применяем стерео-расширитель Хааса
                from app.audio.vocal_enhancer import VocalEnhancer
                enhancer = VocalEnhancer(sr=sr)
                phys = self.get_physical_params(plugin)
                delay_val = phys.get("delay_ms", 18.0)
                width_val = phys.get("width", 1.0)
                y_working = enhancer.apply_stereo_enhancer(y_working, delay_ms=delay_val, width=width_val)
                
            elif plugin.name == "harmony":
                # Параллельная шина гармонии/дабла: сдвигается сигнал после инсертов лидера, стоящих выше
                # по цепочке, и сумма идет дальше (в реверб и дилей). Собственных инсертов у слоя нет.
                # Все слои строятся из лидера на месте первого harmony, иначе второй слой сдвигал бы и первый.
                if effects:
                    board = Pedalboard(effects)
                    try:
                        y_working = board(y_working, sr)
                    except Exception as e:
                        print(f"[-] Ошибка обработки Pedalboard перед гармонией: {e}")
                    effects = []
                if harmony_source is None:
                    harmony_source = y_working

                from app.audio.vocal_enhancer import VocalEnhancer
                enhancer = VocalEnhancer(sr=sr)
                y_working = enhancer.add_layer(y_working, enhancer.harmony_layer(
                    harmony_source,
                    semitones=float(round(phys.get("semitones", 3.0))),
                    level_db=phys.get("level_db", -8.0),
                    formant_shift=phys.get("formant_shift", 0.0),
                    delay_ms=phys.get("delay_ms", 0.0),
                    detune_cents=phys.get("detune_cents", 0.0),
                    width=phys.get("width", 0.0),
                    mix=phys.get("mix", 1.0),
                ))

            elif plugin.name == "reverb" or (vst_loaded is False and plugin.name == "valhalla_vintage_verb"):
                wet_val = phys.get("wet_level", phys.get("mix", 0.1))
                if wet_val > 1.0:
                    wet_val = wet_val / 100.0
                wet_val = float(np.clip(wet_val, 0.0, 1.0))
                if not any(k in phys for k in REVERB_SEND_PARAMS):
                    effects.append(Reverb(
                        room_size=phys.get("room_size", 0.5),
                        damping=phys.get("damping", 0.5),
                        wet_level=wet_val,
                        dry_level=phys.get("dry_level", 0.9)
                    ))
                else:
                    # Ревер как send-шина: predelay и фильтры стоят только на мокром сигнале, как в реальных
                    # миксах (low cut 400–700 Гц, темный хвост), сухой проходит мимо них.
                    if effects:
                        try:
                            y_working = Pedalboard(effects)(y_working, sr)
                        except Exception as e:
                            print(f"[-] Ошибка обработки Pedalboard перед ревером: {e}")
                        effects = []
                    # Срезы ниже Найквиста: на 22050 Гц фильтр на 20 кГц неустойчив и ломает хвост
                    nyquist_safe = 0.45 * sr
                    send = [HighpassFilter(cutoff_frequency_hz=min(phys.get("low_cut_hz", 20.0), nyquist_safe)),
                            LowpassFilter(cutoff_frequency_hz=min(phys.get("high_cut_hz", 20000.0), nyquist_safe))]
                    predelay_ms = phys.get("predelay_ms", 0.0)
                    if predelay_ms > 0.5:
                        send.insert(0, Delay(delay_seconds=predelay_ms / 1000.0, feedback=0.0, mix=1.0))
                    if "decay_s" in phys:
                        # Ревер с явным RT60 вместо Freeverb (room_size тогда не используется)
                        wet = decay_reverb(Pedalboard(send)(y_working, sr), sr, phys["decay_s"],
                                           phys.get("damping", 0.5))
                    else:
                        send.append(Reverb(room_size=phys.get("room_size", 0.5), damping=phys.get("damping", 0.5),
                                           wet_level=1.0, dry_level=0.0))
                        wet = Pedalboard(send)(y_working, sr)
                    # JUCE Reverb внутри умножает dry_level на 2 (wet уже учтен через wet_level=1.0) — держим тот же
                    # масштаб, чтобы send-шина без predelay и фильтров звучала как последовательный Reverb
                    y_working = (2.0 * phys.get("dry_level", 0.9) * y_working + wet_val * wet).astype(np.float32)
                
            elif plugin.name == "delay":
                effects.append(Delay(
                    delay_seconds=phys.get("delay_seconds", 0.25),
                    feedback=phys.get("feedback", 0.2),
                    mix=phys.get("mix", 0.1)
                ))
                
        # Лимитер в самом конце цепочки, чтобы предотвратить цифровой клиппинг
        if final_limiter:
            effects.append(Limiter(threshold_db=-1.0, release_ms=10.0))

        if effects:
            board = Pedalboard(effects)
            try:
                y_working = board(y_working, sr)
            except Exception as e:
                print(f"[-] Ошибка обработки Pedalboard на выходе: {e}")
            
        # Применение гармонического экситера (если плагин присутствует в цепочке)
        for plugin in chain_config.plugins:
            if plugin.name == "exciter":
                plugin_used = self.clamp_normalized_params(plugin) if apply_safety_clamps else plugin
                phys = self.get_physical_params(plugin_used)
                mix_val = phys.get("mix", 0.15)
                cutoff_hz_val = phys.get("cutoff_hz", 7000.0)
                from app.audio.vocal_enhancer import VocalEnhancer
                enhancer = VocalEnhancer(sr=sr)
                y_working = enhancer.apply_exciter(y_working, cutoff_hz=cutoff_hz_val, mix=mix_val)
                break
                
        return y_working
