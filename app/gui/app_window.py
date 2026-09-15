import sys
import os
import time
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QFileDialog, 
                             QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QPointF, QMimeData
from PySide6.QtGui import QPainter, QColor, QPen, QDrag, QPixmap, QIcon, QPainterPath

# Добавление корня проекта в пути импорта Python для обеспечения правильного импорта
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

DEFAULT_MODEL_PATH = ROOT_DIR / "data" / "models" / "best_model.pth"

from app.audio.audio_processor import AudioProcessor
from app.audio.dsp_engine import DSPEngine
from app.audio.vocal_enhancer import VocalEnhancer
from app.ml.model import MixingAIModel
from app.presets.preset_generator import PresetGenerator
from app.presets.full_preset_generator import FullPresetGenerator
from app.dataset.dataset_schema import CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS

class EQVisualizer(QFrame):
    """
    Виджет визуализации кривой эквалайзера (EQ) на основе PySide6 QFrame.
    Отображает сетку частот и плавную кривую АЧХ, предсказанную моделью.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bands = []
        self.setMinimumHeight(180)
        self.setStyleSheet("background: #111116; border: 1px solid #2d2d35; border-radius: 6px;")

    def set_bands(self, bands: list):
        """Устанавливает полосы эквалайзера для отрисовки."""
        self.bands = bands
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        mid_y = h / 2

        # 1. Рассчитываем динамический масштаб по вертикали (Y limit)
        max_gain_abs = max([abs(g) for f, g in self.bands]) if self.bands else 0.0
        # Округляем до ближайшего кратного 6 дБ, минимум 12 дБ
        y_limit = max(12.0, np.ceil(max_gain_abs / 6.0) * 6.0)

        # 2. Отрисовка координатной сетки и подписей
        painter.setPen(QPen(QColor(40, 40, 50, 150), 1))
        
        # Настройка шрифта для сетки
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        
        # 0 дБ линия
        painter.drawLine(0, mid_y, w, mid_y)
        painter.drawText(w - 45, mid_y + 11, "0 дБ")
        
        # Линии верхнего и нижнего шагов
        for step in [0.5, -0.5]:
            y_val = mid_y - step * (h / 2)
            db_val = step * y_limit
            painter.drawLine(0, y_val, w, y_val)
            painter.drawText(w - 45, y_val + 11, f"{db_val:+.0f} дБ")

        # Вертикальные логарифмические линии частот (100Гц, 1кГц, 10кГц)
        grid_freqs = [100.0, 1000.0, 10000.0]
        for f in grid_freqs:
            x = (np.log10(f) - np.log10(20)) / (np.log10(20000) - np.log10(20)) * w
            painter.drawLine(x, 0, x, h)
            f_str = f"{int(f)} Гц" if f < 1000 else f"{int(f/1000)} кГц"
            painter.drawText(x + 5, h - 8, f_str)

        if not self.bands: 
            return

        # 3. Красивая верхняя панель параметров (сводный текст)
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(170, 170, 185), 1))
        
        text_parts = []
        for i, (f, g) in enumerate(self.bands):
            f_str = f"{int(f)}Гц" if f < 1000 else f"{f/1000:.1f}кГц"
            text_parts.append(f"B{i+1}: {f_str} ({g:+.1f}дБ)")
        full_text = "  |  ".join(text_parts)
        painter.drawText(15, 22, full_text)

        # 4. Рисование плавной кривой эквалайзера
        points = []
        for freq, gain in sorted(self.bands, key=lambda x: x[0]):
            x = (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20)) * w
            y = mid_y - (gain / y_limit) * (h / 2)
            points.append(QPointF(x, y))

        if len(points) >= 2:
            path = QPainterPath()
            path.moveTo(points[0])
            for p in points[1:]:
                path.lineTo(p)
            
            # Свечение кривой (светло-синий/бирюзовый ореол)
            painter.setPen(QPen(QColor(0, 191, 255, 75), 6))
            painter.drawPath(path)
            
            # Основная белая линия кривой (как в Parametric EQ 2)
            painter.setPen(QPen(QColor(255, 255, 255), 2.5))
            painter.drawPath(path)

        # 5. Отрисовка аккуратных круглых маркеров полос (как в Parametric EQ 2)
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        
        # 7 цветов полос Fruity Parametric EQ 2:
        # 1-Красный, 2-Оранжевый, 3-Желтый, 4-Зеленый, 5-Голубой, 6-Синий, 7-Фиолетовый
        eq2_colors = [
            QColor(244, 67, 54),   # Band 1: Красный
            QColor(255, 152, 0),   # Band 2: Оранжевый
            QColor(255, 235, 59),  # Band 3: Желтый
            QColor(76, 175, 80),   # Band 4: Зеленый
            QColor(0, 188, 212),   # Band 5: Бирюзовый
            QColor(63, 81, 181),   # Band 6: Синий
            QColor(156, 39, 176)   # Band 7: Фиолетовый
        ]
        
        for i, (freq, gain) in enumerate(self.bands):
            x = (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20)) * w
            y = mid_y - (gain / y_limit) * (h / 2)
            
            # Берем соответствующий цвет из палитры EQ 2
            color = eq2_colors[i % len(eq2_colors)]
            painter.setBrush(color)
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
            painter.drawEllipse(x - 9, y - 9, 18, 18)
            
            # Номер полосы по центру круга. 
            # Для желтой полосы (3) используем черный цвет шрифта, для остальных белый
            text_color = Qt.black if i == 2 else Qt.white
            painter.setPen(QPen(text_color))
            painter.drawText(x - 4, y + 5, str(i + 1))


class DropZoneWidget(QFrame):
    """
    Виджет загрузки файлов методом Drag-and-Drop или ручным выбором.
    """
    fileSelected = Signal(str)
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.file_path = None
        self.setAcceptDrops(True)
        self.setMinimumHeight(110)
        self.update_style(dragging=False)
        
        layout = QVBoxLayout()
        self.label_title = QLabel(title.upper())
        self.label_title.setStyleSheet("font-size: 11px; color: #888; font-weight: bold; font-family: Segoe UI;")
        self.label_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_title)
        
        self.icon_label = QLabel("📥")
        self.icon_label.setStyleSheet("font-size: 26px;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)
        
        self.label_status = QLabel("Перетащите аудио файл\nили кликните для выбора")
        self.label_status.setStyleSheet("font-size: 10px; color: #aaa; font-family: Segoe UI;")
        self.label_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_status)
        
        self.setLayout(layout)
        
    def update_style(self, dragging=False):
        if self.file_path:
            # Файл загружен (оранжевая рамка под FL Studio)
            self.setStyleSheet("""
                border: 2px solid #FF8A00;
                background: #251e18;
                border-radius: 8px;
            """)
        elif dragging:
            # Перетаскивание над зоной (голубая рамка)
            self.setStyleSheet("""
                border: 2px dashed #00E5FF;
                background: #102630;
                border-radius: 8px;
            """)
        else:
            # Дефолтное состояние (темная рамка)
            self.setStyleSheet("""
                border: 2px dashed #3a3a42;
                background: #15151a;
                border-radius: 8px;
            """)
            
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.update_style(dragging=True)
            
    def dragLeaveEvent(self, event):
        self.update_style(dragging=False)
        
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.wav', '.mp3', '.flac')):
                self.set_file(path)
                event.acceptProposedAction()
            else:
                self.update_style(dragging=False)
                
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_name, _ = QFileDialog.getOpenFileName(
                self, f"Выберите {self.title}", "", "Audio Files (*.wav *.mp3 *.flac)"
            )
            if file_name:
                self.set_file(file_name)
                
    def set_file(self, path: str):
        self.file_path = path
        p = Path(path)
        self.label_status.setText(p.name)
        self.icon_label.setText("🎵")
        self.update_style(dragging=False)
        self.fileSelected.emit(path)


class DragOutWidget(QFrame):
    """
    Виджет Drag-Out для перетаскивания сгенерированного пресета (.fst) 
    напрямую в FL Studio (микшер или рэк).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fst_path = None
        self.setAcceptDrops(False)
        self.setMinimumSize(130, 90)
        self.setStyleSheet("""
            background: #15151a; 
            border: 2px dashed #444; 
            border-radius: 8px;
            color: #777;
        """)
        
        layout = QVBoxLayout()
        self.icon_label = QLabel("📦")
        self.icon_label.setStyleSheet("font-size: 32px;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)
        
        self.text_label = QLabel("Экспорт в FL Studio\n(пока не готов)")
        self.text_label.setStyleSheet("font-size: 11px; font-weight: bold; font-family: Segoe UI;")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)
        
        self.setLayout(layout)
        self.setEnabled(False)
        
    def set_fst_path(self, path: str):
        self.fst_path = path
        self.setEnabled(True)
        self.icon_label.setText("🔥")
        self.text_label.setText("Перетащи пресет\nв FL Studio!")
        self.setStyleSheet("""
            background: #2b1f1a; 
            border: 2px solid #FF8A00; 
            border-radius: 8px;
            color: #FF8A00;
        """)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.fst_path:
            self.drag_start_position = event.position().toPoint()
            
    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if not self.fst_path:
            return
        if (event.position().toPoint() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return
            
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(self.fst_path)])
        drag.setMimeData(mime_data)
        
        # Создаем привлекательную иконку перетаскивания (Pixmap)
        pixmap = QPixmap(80, 80)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Рисуем оранжево-желтую иконку FL Studio
        painter.setBrush(QColor(255, 110, 0))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(10, 10, 60, 60)
        
        painter.setPen(QPen(Qt.white, 2))
        painter.setFont(self.font())
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "Mixing\nEQ")
        painter.end()
        
        drag.setPixmap(pixmap)
        drag.exec(Qt.CopyAction)


class AnalysisThread(QThread):
    """
    Фоновый рабочий поток PySide6 (QThread) для инференса PyTorch модели 
    и оффлайн Pedalboard DSP-рендеринга. Предотвращает фризы интерфейса.
    """
    finished = Signal(dict)
    progress = Signal(str)

    def __init__(self, source: str, ref: str, model_path: str = None):
        super().__init__()
        self.source = source
        self.ref = ref
        self.model_path = model_path if model_path else str(DEFAULT_MODEL_PATH)

    def run(self):
        try:
            self.progress.emit("Инициализация DSP движков...")
            import torch
            import librosa
            import pyloudnorm as pyln
            import soundfile as sf
            
            processor = AudioProcessor(sr=44100)
            dsp = DSPEngine()
            
            self.progress.emit("Загрузка и подготовка аудио...")
            y_src, _ = librosa.load(self.source, sr=44100)
            y_ref, _ = librosa.load(self.ref, sr=44100)
            
            # Нормализация громкости и удаление тишины для корректной АЧХ
            y_src_norm = processor.load_audio(self.source, target_lufs=-23.0, trim_silence=True)
            y_ref_norm = processor.load_audio(self.ref, target_lufs=-23.0, trim_silence=True)
            
            self.progress.emit("Анализ спектра (FFT)...")
            mel_src = processor.get_mel_spectrogram(y_src_norm)
            mel_ref = processor.get_mel_spectrogram(y_ref_norm, pitch_normalize=True)
            
            # Дополнение спектрограмм
            max_len = 700
            def pad_spec(mel):
                n_mels, t = mel.shape
                if t >= max_len:
                    return mel[:, :max_len]
                pad_val = mel.min()
                padded = np.full((n_mels, max_len), pad_val, dtype=np.float32)
                padded[:, :t] = mel
                return padded
                
            mel_src_padded = pad_spec(mel_src)
            mel_ref_padded = pad_spec(mel_ref)
            # Формирование входного тензора (1 канал: только референс)
            x = mel_ref_padded[np.newaxis, :, :] # (1, 128, max_len)
            x_tensor = torch.from_numpy(x).unsqueeze(0) # (1, 1, 128, max_len)
            
            self.progress.emit("Инференс нейросети (512 ResNet)...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MixingAIModel().to(device)
            
            # Загрузка весов с проверкой на размерность для предотвращения сбоев при несовпадении версий моделей
            checkpoint = torch.load(self.model_path, map_location=device)
            state_dict = checkpoint["model_state_dict"]
            model_dict = model.state_dict()
            
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    print(f"[!] Предупреждение: Несовпадение размерностей для слоя {k}, пропуск.")
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            model.eval()
            
            with torch.no_grad():
                x_tensor = x_tensor.to(device)
                logits, pred_params = model(x_tensor)
                
                # Классификация
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                active_plugins = (probs > 0.5).astype(int)
                
                # Регрессия параметров
                params = pred_params.squeeze(0).cpu().numpy()
                
            self.progress.emit("Сведение эквалайзера (Spectral Matcher)...")
            # Расчет кривой компенсации эквализации
            env_src = processor.get_spectral_envelope(y_src_norm)
            env_ref = processor.get_spectral_envelope(y_ref_norm)
            delta_eq = env_ref - env_src
            freqs = librosa.fft_frequencies(sr=44100, n_fft=2048)
            
            eq_gen = PresetGenerator()
            bands = eq_gen.extract_key_bands(freqs, delta_eq)
            
            self.progress.emit("Создание .fst шаблона эквалайзера...")
            fst_gen = FullPresetGenerator(template_path=str(ROOT_DIR / "FlProject" / "template.fst"))
            fst_file = fst_gen.generate_vocal_chain(bands, output_name="mixing_ai_eq.fst")
            
            self.progress.emit("Оффлайн-рендеринг Pedalboard...")
            # Построение структуры цепочки плагинов
            from app.dataset.dataset_schema import ChainConfig, PluginConfig
            plugins_list = [PluginConfig(name="eq", params={})] # Эквалайзер активен всегда
            for i, plugin_name in enumerate(CHAIN_ORDER):
                if plugin_name == "eq":
                    continue
                if active_plugins[i] == 1:
                    plugin_params = {}
                    if plugin_name in PARAM_OFFSETS:
                        offset = PARAM_OFFSETS[plugin_name]
                        layout = PARAM_LAYOUT[plugin_name]
                        for param_idx, param_name in enumerate(layout):
                            plugin_params[param_name] = float(params[offset + param_idx])
                    plugins_list.append(PluginConfig(name=plugin_name, params=plugin_params))
                    
            chain_config = ChainConfig(plugins=plugins_list)
            
            # Порог Noise Gate на основе шума
            gate_thresh = processor.estimate_noise_gate_threshold(y_src)
            
            # Детекция тональности референса для Pitch Corrector
            self.progress.emit("Определение тональности референса...")
            enhancer = VocalEnhancer(sr=44100)
            ref_key, ref_scale = enhancer.detect_key_and_scale(y_ref)
            print(f"[+] Детектор тональности референса: {ref_key} ({ref_scale})")
            
            self.progress.emit("Обработка аудио в DSPEngine...")
            y_processed = dsp.apply_chain(
                y_src, 44100, chain_config, 
                gate_threshold_db=gate_thresh,
                target_key_index=ref_key,
                target_scale=ref_scale
            )
            
            # Нормализация громкости на выходе
            meter = pyln.Meter(44100)
            try:
                # pyloudnorm ожидает форму (samples, channels) для стерео
                y_temp = y_processed.T if y_processed.ndim == 2 else y_processed
                processed_loudness = meter.integrated_loudness(y_temp)
                if not np.isnan(processed_loudness) and not np.isinf(processed_loudness):
                    y_temp = pyln.normalize.loudness(y_temp, processed_loudness, -23.0)
                    y_processed = y_temp.T if y_processed.ndim == 2 else y_temp
            except Exception:
                pass
                
            # Запись обработанного WAV
            out_audio_path = Path("data/processed/res_gui.wav")
            out_audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_audio_path, y_processed.T if y_processed.ndim == 2 else y_processed, 44100)
            
            one_liner = eq_gen.generate_one_liner(bands)
            
            results = {
                "success": True,
                "bands": bands,
                "active_plugins": active_plugins,
                "params": params,
                "fst_path": str(fst_file.absolute()) if fst_file else None,
                "audio_path": str(out_audio_path.absolute()),
                "script": one_liner,
                "plugins_list": plugins_list
            }
            self.finished.emit(results)
            
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class MixingAIApp(QMainWindow):
    """Главный графический интерфейс Mixing-AI с премиум темным дизайном."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixing-AI: Companion")
        self.setFixedSize(660, 920)
        
        # Настройка Frameless (безрамочного) окна
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint)
        self.setStyleSheet("""
            QMainWindow {
                background: #1a1a20;
                border: 2px solid #32323a;
            }
            QLabel {
                color: #e2e2e9;
                font-family: 'Segoe UI', 'Outfit', sans-serif;
            }
            QPushButton {
                background: #23232a;
                border: 1px solid #3c3c46;
                border-radius: 5px;
                color: #e2e2e9;
                padding: 10px;
                font-weight: bold;
                font-family: 'Segoe UI';
            }
            QPushButton:hover {
                background: #2d2d38;
                border-color: #FF8A00;
            }
            QPushButton:disabled {
                background: #141418;
                color: #555;
                border-color: #222;
            }
        """)

        # Переменные для перетаскивания окна
        self.drag_position = None

        # Инициализация макета
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 10, 15, 15)
        
        # 1. Custom Title Bar (Кастомный заголовок)
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(5, 5, 5, 10)
        
        self.title_logo = QLabel("🤖 Mixing-AI Companion")
        self.title_logo.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF8A00;")
        title_layout.addWidget(self.title_logo)
        
        title_layout.addStretch()
        
        # Кнопка Always-on-top
        self.chk_top = QPushButton("📌 Поверх окон")
        self.chk_top.setCheckable(True)
        self.chk_top.setStyleSheet("""
            QPushButton { padding: 4px 10px; font-size: 10px; border-radius: 3px; }
            QPushButton:checked { background: #FF8A00; color: black; border-color: #FF8A00; }
        """)
        self.chk_top.clicked.connect(self.toggle_always_on_top)
        title_layout.addWidget(self.chk_top)
        
        # Кнопка свернуть
        btn_min = QPushButton("—")
        btn_min.setStyleSheet("QPushButton { padding: 2px 8px; font-size: 11px; border: none; }")
        btn_min.clicked.connect(self.showMinimized)
        title_layout.addWidget(btn_min)
        
        # Кнопка закрыть
        btn_close = QPushButton("✕")
        btn_close.setStyleSheet("QPushButton { padding: 2px 8px; font-size: 11px; border: none; } QPushButton:hover { color: #f44336; }")
        btn_close.clicked.connect(self.close)
        title_layout.addWidget(btn_close)
        
        main_layout.addLayout(title_layout)

        # 2. Две Drop-зоны ввода файлов
        zones_layout = QHBoxLayout()
        self.zone_dry = DropZoneWidget("Исходный сухой вокал")
        self.zone_ref = DropZoneWidget("Песня-ориентир (Референс)")
        zones_layout.addWidget(self.zone_dry)
        zones_layout.addWidget(self.zone_ref)
        main_layout.addLayout(zones_layout)

        # 3. График-визуализатор эквалайзера
        self.visualizer = EQVisualizer()
        main_layout.addWidget(self.visualizer)

        # 4. Панель параметров плагинов (Инструменты сведения)
        self.param_widget = QFrame()
        self.param_widget.setStyleSheet("background: #17171d; border: 1px solid #23232c; border-radius: 8px;")
        param_layout = QVBoxLayout(self.param_widget)
        param_layout.setContentsMargins(15, 15, 15, 15)
        
        self.param_title = QLabel("ДИАГНОСТИКА ЦЕПОЧКИ ЭФФЕКТОВ (AI PREDICTIONS)")
        self.param_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #888; letter-spacing: 1px;")
        param_layout.addWidget(self.param_title)
        
        self.grid_params = QGridLayout()
        self.grid_params.setSpacing(10)
        param_layout.addLayout(self.grid_params)
        
        main_layout.addWidget(self.param_widget)
        
        # Инициализируем пустую сетку эффектов
        self.init_empty_params()

        # 5. Кнопка запуска и индикатор статуса
        self.status_label = QLabel("Ожидание загрузки файлов...")
        self.status_label.setStyleSheet("color: #777; font-size: 11px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.btn_run = QPushButton("✨ СВЕСТИ ВОКАЛ ПО СТИЛЮ РЕФЕРЕНСА")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF8A00, stop:1 #E65100);
                color: white; 
                padding: 15px; 
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF9E15, stop:1 #FF6E00);
            }
            QPushButton:disabled {
                background: #1c1c22;
                color: #555;
            }
        """)
        self.btn_run.setEnabled(False)
        main_layout.addWidget(self.btn_run)

        # 6. Панель экспорта (Drag-Out и Скрипт)
        self.export_panel = QHBoxLayout()
        self.export_panel.setContentsMargins(0, 10, 0, 0)
        
        self.drag_out = DragOutWidget()
        self.export_panel.addWidget(self.drag_out)
        
        right_export = QVBoxLayout()
        self.btn_copy_script = QPushButton("📋 Копировать скрипт FL Studio")
        self.btn_copy_script.clicked.connect(self.copy_script_to_clipboard)
        self.btn_copy_script.setEnabled(False)
        right_export.addWidget(self.btn_copy_script)
        
        self.btn_play_audio = QPushButton("🔊 Воспроизвести сведенный файл")
        self.btn_play_audio.clicked.connect(self.play_audio)
        self.btn_play_audio.setEnabled(False)
        right_export.addWidget(self.btn_play_audio)
        
        self.export_panel.addLayout(right_export)
        main_layout.addLayout(self.export_panel)

        # Связываем сигналы слотов
        self.zone_dry.fileSelected.connect(self.check_files)
        self.zone_ref.fileSelected.connect(self.check_files)
        
        self.script_data = ""
        self.processed_audio_path = ""

    def toggle_always_on_top(self):
        """Переключает флаг 'Поверх всех окон'."""
        if self.chk_top.isChecked():
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

    def mousePressEvent(self, event):
        """Обработчик нажатия мыши для перетаскивания безрамочного окна."""
        if event.button() == Qt.LeftButton:
            # Разрешаем тащить окно только за верхнюю часть (заголовок высотой 45 пикселей)
            if event.position().y() < 45:
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                
    def mouseMoveEvent(self, event):
        """Передвигает окно при зажатой левой кнопке мыши."""
        if event.buttons() & Qt.LeftButton and self.drag_position is not None:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
            
    def mouseReleaseEvent(self, event):
        self.drag_position = None

    def check_files(self):
        """Проверяет наличие файлов в drop-зонах для разблокировки кнопки запуска."""
        if self.zone_dry.file_path and self.zone_ref.file_path:
            self.btn_run.setEnabled(True)
            self.status_label.setText("Готов к анализу!")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
        else:
            self.btn_run.setEnabled(False)

    def init_empty_params(self):
        """Заполняет панель параметров эффектов дефолтными пустыми значениями."""
        # Очищаем сетку
        for i in reversed(range(self.grid_params.count())): 
            self.grid_params.itemAt(i).widget().setParent(None)
            
        # Список отображаемых плагинов с их маппингом из схемы
        display_map = [
            ("pitch_corrector", "Pitch Corrector"),
            ("compressor", "Compressor"),
            ("deesser", "De-esser"),
            ("multiband_compressor", "Multiband Comp (OTT)"),
            ("resonance_suppressor", "Resonance Suppressor"),
            ("distortion", "Distortion"),
            ("chorus", "Chorus"),
            ("stereo_enhancer", "Stereo Enhancer"),
            ("reverb", "Reverb"),
            ("delay", "Delay"),
            ("exciter", "Exciter")
        ]
        
        for idx, (code_name, display_name) in enumerate(display_map):
            r, c = idx // 2, idx % 2
            
            box = QFrame()
            box.setStyleSheet("background: #1b1b22; border: 1px solid #23232c; border-radius: 4px; padding: 5px;")
            l = QHBoxLayout(box)
            l.setContentsMargins(5, 5, 5, 5)
            
            fx_status = QLabel("⚪")
            fx_status.setStyleSheet("font-size: 10px;")
            l.addWidget(fx_status)
            
            fx_name = QLabel(display_name)
            fx_name.setStyleSheet("font-weight: bold; font-size: 11px; color: #888;")
            l.addWidget(fx_name)
            l.addStretch()
            
            fx_val = QLabel("Bypass")
            fx_val.setStyleSheet("font-size: 11px; color: #555;")
            l.addWidget(fx_val)
            
            self.grid_params.addWidget(box, r, c)

    def update_params_display(self, active_plugins, params, plugins_list):
        """Обновляет значения ручек и активность эффектов в интерфейсе."""
        # Очищаем
        for i in reversed(range(self.grid_params.count())): 
            self.grid_params.itemAt(i).widget().setParent(None)
            
        # Формируем словарь параметров для удобства
        dict_plugins = {p.name: p.params for p in plugins_list}
        
        # Список отображаемых плагинов с их маппингом из схемы
        display_map = [
            ("pitch_corrector", "Pitch Corrector", "speed", None, "%", ""),
            ("compressor", "Compressor", "threshold_db", "ratio", "дБ", "x"),
            ("deesser", "De-esser", "threshold_db", "ratio", "дБ", "x"),
            ("multiband_compressor", "Multiband Comp (OTT)", "depth", None, "%", ""),
            ("resonance_suppressor", "Resonance Suppressor", "threshold_db", "max_attenuation_db", "дБ", "дБ"),
            ("distortion", "Distortion", "drive_db", None, "дБ", ""),
            ("chorus", "Chorus", "rate_hz", "mix", "Гц", "%"),
            ("stereo_enhancer", "Stereo Enhancer", "width", "delay_ms", "x", "мс"),
            ("reverb", "Reverb", "room_size", "wet_level", "", "%"),
            ("delay", "Delay", "feedback", "mix", "", "%"),
            ("exciter", "Exciter", "mix", "cutoff_hz", "%", "Гц")
        ]
        
        for idx, (code_name, display_name, param1, param2, unit1, unit2) in enumerate(display_map):
            r, c = idx // 2, idx % 2
            
            box = QFrame()
            is_active = code_name in dict_plugins
            
            if is_active:
                box.setStyleSheet("background: #142518; border: 1px solid #4CAF50; border-radius: 4px; padding: 5px;")
                status_dot = "🟢"
                name_style = "font-weight: bold; font-size: 11px; color: #4CAF50;"
                
                # Достаем физические значения
                dsp = DSPEngine()
                dummy_plugin = plugins_list[[p.name for p in plugins_list].index(code_name)]
                phys = dsp.get_physical_params(dummy_plugin)
                
                val_text = ""
                if param1 in phys:
                    val1 = phys[param1]
                    if unit1 == "%" and val1 <= 1.0:
                        val_text += f"{val1*100:.0f}{unit1}"
                    else:
                        val_text += f"{val1:.1f}{unit1}"
                if param2 and param2 in phys:
                    val2 = phys[param2]
                    if unit2 == 'x':
                        val_text += f" | {val2:.2f}{unit2}"
                    elif unit2 == '%':
                        val_text += f" | {val2*100:.0f}{unit2}"
                    else:
                        val_text += f" | {val2:.1f}{unit2}"
                    
                val_style = "font-size: 11px; color: #fff; font-family: Consolas;"
            else:
                box.setStyleSheet("background: #1b1b22; border: 1px solid #23232c; border-radius: 4px; padding: 5px;")
                status_dot = "⚪"
                name_style = "font-weight: bold; font-size: 11px; color: #777;"
                val_text = "Bypass"
                val_style = "font-size: 11px; color: #555;"
                
            l = QHBoxLayout(box)
            l.setContentsMargins(5, 5, 5, 5)
            
            lbl_dot = QLabel(status_dot)
            l.addWidget(lbl_dot)
            
            lbl_name = QLabel(display_name)
            lbl_name.setStyleSheet(name_style)
            l.addWidget(lbl_name)
            l.addStretch()
            
            lbl_val = QLabel(val_text)
            lbl_val.setStyleSheet(val_style)
            l.addWidget(lbl_val)
            
            self.grid_params.addWidget(box, r, c)

    def run_analysis(self):
        """Запускает фоновый поток вычисления ИИ-сведения."""
        self.btn_run.setEnabled(False)
        self.status_label.setText("Запуск фонового процесса анализа...")
        self.status_label.setStyleSheet("color: #00E5FF; font-size: 11px;")
        
        self.thread = AnalysisThread(self.zone_dry.file_path, self.zone_ref.file_path)
        self.thread.progress.connect(self.on_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_progress(self, msg: str):
        """Отображает текущий шаг прогресса в интерфейсе."""
        self.status_label.setText(msg)

    def on_finished(self, results: dict):
        """Обработчик завершения процесса сведения."""
        self.btn_run.setEnabled(True)
        
        if not results.get("success", False):
            self.status_label.setText(f"Ошибка сведения: {results.get('error', 'Неизвестная ошибка')}")
            self.status_label.setStyleSheet("color: #f44336; font-size: 11px;")
            return
            
        self.status_label.setText("Сведение успешно завершено!")
        self.status_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
        
        # Обновляем визуализатор EQ
        self.visualizer.set_bands(results["bands"])
        
        # Обновляем панель параметров
        self.update_params_display(results["active_plugins"], results["params"], results["plugins_list"])
        
        # Разблокируем экспорт и Drag-Out
        self.script_data = results["script"]
        self.processed_audio_path = results["audio_path"]
        
        self.btn_copy_script.setEnabled(True)
        self.btn_play_audio.setEnabled(True)
        
        if results["fst_path"]:
            self.drag_out.set_fst_path(results["fst_path"])
        else:
            self.drag_out.text_label.setText("Ошибка генерации пресета")

    def copy_script_to_clipboard(self):
        """Копирует сгенерированный Python скрипт-однострочник в буфер обмена."""
        if self.script_data:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.script_data)
            self.status_label.setText("Скрипт скопирован! Вставьте его в Python консоль FL Studio.")
            self.status_label.setStyleSheet("color: #00E5FF; font-size: 11px;")

    def play_audio(self):
        """Воспроизводит сгенерированный WAV файл в системном плеере."""
        if self.processed_audio_path and os.path.exists(self.processed_audio_path):
            os.startfile(self.processed_audio_path)


def main():
    app = QApplication(sys.argv)
    window = MixingAIApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
