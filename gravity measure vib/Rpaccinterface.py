import sys
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

class WorkerThread(QThread):
    # Определяем сигнал для передачи данных в основной поток
    update_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        """Код, выполняемый в отдельном потоке"""
        while 1:
            print(1)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Многопоточность в PyQt5')

        self.start_button = QPushButton('start', self)
        self.start_button.clicked.connect(self.start_thread)

        self.stop_button = QPushButton('stop', self)
        self.stop_button.move(0, 20)
        self.stop_button.clicked.connect(self.stop_thread)

        self.thread = WorkerThread()



    def start_thread(self):
        self.thread.start()  # Запускаем поток

    def stop_thread(self):
        self.thread.quit()
        self.thread.wait()  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())