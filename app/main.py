import sys
from pathlib import Path

# Добавляем корень проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from PySide6.QtWidgets import QApplication
from app.gui.app_window import MixingAIApp

def main():
    """
    Основная точка входа для запуска GUI-приложения Mixing-AI.
    """
    app = QApplication(sys.argv)
    window = MixingAIApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
