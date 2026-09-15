import numpy as np
from pedalboard import Pedalboard, Compressor, Reverb, Delay, LowShelfFilter, PeakFilter, HighShelfFilter, Chorus, Distortion, HighpassFilter, LowpassFilter, NoiseGate, Limiter
from app.dataset.dataset_schema import ChainConfig, PluginConfig, DATAGEN_PLUGINS
from app.bridge.vst_parameter_mapper import VSTParameterMapper

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
                clamped_params["mix"] = min(clamped_params["mix"], 0.20)
            if "rate_hz" in clamped_params:
                clamped_params["rate_hz"] = min(clamped_params["rate_hz"], 0.25)
        elif name == "distortion":
            if "drive_db" in clamped_params:
                clamped_params["drive_db"] = min(clamped_params["drive_db"], 0.40)
        elif name == "reverb":
            if "wet_level" in clamped_params:
                clamped_params["wet_level"] = min(clamped_params["wet_level"], 0.22)
        elif name == "delay":
            if "mix" in clamped_params:
                clamped_params["mix"] = min(clamped_params["mix"], 0.20)
                
        return PluginConfig(name=name, params=clamped_params)

    def get_physical_params(self, plugin_config: PluginConfig) -> dict:
        """
        Возвращает физические параметры эффекта, преобразованные из нормализованной конфигурации [0..1].
        
        Args:
            plugin_config: Нормализованная конфигурация плагина.
            
        Returns:
            dict: Словарь с физическими параметрами плагина.
        """
        plugin_name = plugin_config.name
        if plugin_name == "eq":
            ranges = DATAGEN_PLUGINS["eq"]["params"]
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
            # Откат к жестко заданным диапазонам по умолчанию, если маппер недоступен
            if plugin_name not in DATAGEN_PLUGINS:
                raise ValueError(f"Плагин '{plugin_name}' не поддерживается DSPEngine или VSTParameterMapper: {e}")
            
            ranges = DATAGEN_PLUGINS[plugin_name]["params"]
            phys_params = {}
            for p_name, val in plugin_config.params.items():
                if p_name in ranges:
                    min_v, max_v = ranges[p_name]
                    phys_params[p_name] = self.denormalize_value(val, min_v, max_v)
                else:
                    phys_params[p_name] = val
            return phys_params

    def apply_chain(self, y: np.ndarray, sr: int, chain_config: ChainConfig, gate_threshold_db: float = None, target_key_index: int = None, target_scale: str = None) -> np.ndarray:
        """
        Применяет последовательную цепочку плагинов к аудиосигналу.
        
        Args:
            y: Входной аудиосигнал.
            sr: Частота дискретизации аудиосигнала.
            chain_config: Конфигурация цепочки плагинов и их параметров.
            gate_threshold_db: Порог Noise Gate в дБ. Если None, гейт не применяется.
            
        Returns:
            np.ndarray: Обработанный аудиосигнал.
        """
        effects = []
        if gate_threshold_db is not None:
            effects.append(NoiseGate(threshold_db=gate_threshold_db, attack_ms=2.0, release_ms=100.0))
            
        y_working = y.copy()
        
        # Применение подавления резонансов (если плагин присутствует в цепочке)
        for plugin in chain_config.plugins:
            if plugin.name == "resonance_suppressor":
                plugin_clamped = self.clamp_normalized_params(plugin)
                phys = self.get_physical_params(plugin_clamped)
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
                print(f"[*] Применение Pitch Corrector (Автотюна): {k_idx} ({s_type}), speed={speed_val:.2f}")
                y_working = enhancer.apply_autotune(y_working, key_index=k_idx, scale=s_type, speed=speed_val)
                
            elif plugin.name == "compressor" or (vst_loaded is False and plugin.name in ("fabfilter_pro_c_2", "waves_cla_2a")):
                effects.append(Compressor(
                    threshold_db=phys.get("threshold_db", phys.get("threshold", -20.0)),
                    ratio=phys.get("ratio", 4.0),
                    attack_ms=phys.get("attack_ms", phys.get("attack", 2.0)),
                    release_ms=phys.get("release_ms", phys.get("release", 100.0))
                ))
                
            elif plugin.name == "deesser":
                # Сначала рендерим предыдущие эффекты
                if effects:
                    board = Pedalboard(effects)
                    try:
                        y_working = board(y_working, sr)
                    except Exception as e:
                        print(f"[-] Ошибка обработки Pedalboard перед деэссером: {e}")
                    effects = []
                
                # Реализуем Split-Band De-esser (разделяем полосу на 4 кГц)
                threshold_db = phys.get("threshold_db", -20.0)
                ratio = phys.get("ratio", 4.0)
                try:
                    low_board = Pedalboard([LowpassFilter(cutoff_frequency_hz=4000.0)])
                    y_low = low_board(y_working, sr)
                    
                    high_board = Pedalboard([
                        HighpassFilter(cutoff_frequency_hz=4000.0),
                        Compressor(threshold_db=threshold_db, ratio=ratio, attack_ms=3.0, release_ms=45.0)
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
                
                # Применяем параллельный дисторшн (8% wet, 92% dry)
                # Отсекаем низ (до 150 Гц) и верх (выше 5000 Гц) на перегруженном сигнале,
                # чтобы убрать свистящий цифровой песок (fizz) и мутный гул.
                drive_db = phys.get("drive_db", 10.0)
                try:
                    dist_board = Pedalboard([
                        Distortion(drive_db=drive_db),
                        HighpassFilter(cutoff_frequency_hz=150.0),
                        LowpassFilter(cutoff_frequency_hz=5000.0)
                    ])
                    y_dist = dist_board(y_working, sr)
                    y_working = 0.92 * y_working + 0.08 * y_dist
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
                
            elif plugin.name == "reverb" or (vst_loaded is False and plugin.name == "valhalla_vintage_verb"):
                wet_val = phys.get("wet_level", phys.get("mix", 0.1))
                if wet_val > 1.0:
                    wet_val = wet_val / 100.0
                effects.append(Reverb(
                    room_size=phys.get("room_size", 0.5),
                    damping=phys.get("damping", 0.5),
                    wet_level=np.clip(wet_val, 0.0, 1.0),
                    dry_level=phys.get("dry_level", 0.9)
                ))
                
            elif plugin.name == "delay":
                effects.append(Delay(
                    delay_seconds=phys.get("delay_seconds", 0.25),
                    feedback=phys.get("feedback", 0.2),
                    mix=phys.get("mix", 0.1)
                ))
                
        # Всегда добавляем лимитер в самый конец цепочки,
        # чтобы предотвратить цифровой клиппинг
        effects.append(Limiter(threshold_db=-1.0, release_ms=10.0))
        
        board = Pedalboard(effects)
        try:
            y_working = board(y_working, sr)
        except Exception as e:
            print(f"[-] Ошибка обработки Pedalboard на выходе: {e}")
            
        # Применение гармонического экситера (если плагин присутствует в цепочке)
        for plugin in chain_config.plugins:
            if plugin.name == "exciter":
                plugin_clamped = self.clamp_normalized_params(plugin)
                phys = self.get_physical_params(plugin_clamped)
                mix_val = phys.get("mix", 0.12)
                cutoff_hz_val = phys.get("cutoff_hz", 7000.0)
                from app.audio.vocal_enhancer import VocalEnhancer
                enhancer = VocalEnhancer(sr=sr)
                y_working = enhancer.apply_exciter(y_working, cutoff_hz=cutoff_hz_val, mix=mix_val)
                break
                
        return y_working
