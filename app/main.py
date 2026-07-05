import sys
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QHBoxLayout, QFrame)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import librosa

from app.core.audio_processor import AudioProcessor
from app.core.preset_generator import PresetGenerator

class EQVisualizer(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bands = []
        self.setMinimumHeight(150)
        self.setStyleSheet("background: #000; border: 1px solid #444;")

    def set_bands(self, bands):
        self.bands = bands
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        mid_y = h / 2

        # Сетка
        painter.setPen(QPen(QColor(40, 40, 40), 1))
        for i in range(1, 10):
            x = (w / 10) * i
            painter.drawLine(x, 0, x, h)
        painter.drawLine(0, mid_y, w, mid_y)

        if not self.bands: return

        # Рисуем точки (шарики) EQ
        for i, (f, g) in enumerate(self.bands):
            # Логарифмическая частота для X
            x = (np.log10(f) - np.log10(10)) / (np.log10(20000) - np.log10(10)) * w
            # Гейн для Y (-18..+18дБ)
            y = mid_y - (g / 18.0) * (h / 2)
            
            painter.setBrush(QColor(76, 175, 80))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - 5, y - 5, 10, 10)
            painter.setPen(QPen(Qt.white, 1))
            painter.drawText(x + 8, y + 5, f"{i+1}")

class AnalysisThread(QThread):
    finished = Signal(list)
    progress = Signal(str)

    def __init__(self, source, ref):
        super().__init__()
        self.source = source
        self.ref = ref

    def run(self):
        processor = AudioProcessor(sr=44100)
        y_s = processor.load_audio(self.source)
        y_ref = processor.load_audio(self.ref)
        env_s = processor.get_spectral_envelope(y_s)
        env_ref = processor.get_spectral_envelope(y_ref)
        delta = env_ref - env_s
        freqs = librosa.fft_frequencies(sr=44100, n_fft=2048)
        
        generator = PresetGenerator()
        bands = generator.extract_key_bands(freqs, delta)
        self.finished.emit(bands)

class MixingAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixing-AI: Visual Guide")
        self.setFixedSize(600, 600)
        self.setStyleSheet("background: #1a1a1a; color: #ddd;")

        self.source_path = "data/raw/source/source.wav"
        self.ref_path = "data/raw/reference/reference.wav"

        layout = QVBoxLayout()
        
        self.title = QLabel("MIXING-AI VISUAL ASSISTANT")
        self.title.setStyleSheet("font-size: 20px; color: #4CAF50; font-weight: bold;")
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        self.visualizer = EQVisualizer()
        layout.addWidget(self.visualizer)

        self.info_label = QLabel("Нажми ANALYZE, чтобы увидеть кривую")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        self.btn_run = QPushButton("ANALYZE & GENERATE GUIDE")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setStyleSheet("background: #4CAF50; color: white; padding: 15px; font-weight: bold;")
        layout.addWidget(self.btn_run)

        self.results_area = QLabel("")
        self.results_area.setStyleSheet("background: #222; padding: 10px; font-family: Consolas;")
        layout.addWidget(self.results_area)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_analysis(self):
        self.btn_run.setEnabled(False)
        self.thread = AnalysisThread(self.source_path, self.ref_path)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, bands):
        self.btn_run.setEnabled(True)
        self.visualizer.set_bands(bands)
        
        text = "<b>ИНСТРУКЦИЯ ДЛЯ FL STUDIO:</b><br><br>"
        for i, (f, g) in enumerate(bands):
            f_p = (np.log10(f) - np.log10(10)) / (np.log10(20000) - np.log10(10))
            g_p = (g + 18) / 36
            text += f"Band {i+1}: <b>FREQ</b> {f_p:.3f} | <b>GAIN</b> {g_p:.3f}<br>"
        
        self.results_area.setText(text)
        self.info_label.setText("Выстави эти значения в Fruity Parametric EQ 2")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MixingAIApp()
    window.show()
    sys.exit(app.exec())
