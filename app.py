
import sys
from PyQt6.QtWidgets import QApplication
from gui import DataCleanerApp

if __name__ == "__main__":
    """
    Main entry point for the Advanced Data Cleaning Tool.
    
    This script launches the PyQt6 GUI application.
    """
    app = QApplication(sys.argv)
    window = DataCleanerApp()
    window.show()
    sys.exit(app.exec())

