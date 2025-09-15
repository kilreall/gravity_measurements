import sys
import time
import io
from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton



class WorkerSignals(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)


class Worker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self._is_running = True

    @pyqtSlot()
    def run(self):
        i = 0
        print("Поток запущен")
        while self._is_running:
            print(f"Работаю: {i}")
            self.signals.progress.emit(i)
            i += 1
            time.sleep(0.1)  # чтобы не грузить CPU
        print("Поток завершён")
        self.signals.finished.emit()

    def stop(self):
        print("Остановка потока")
        self._is_running = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.threadpool = QThreadPool()
        self.worker = None

    def initUI(self):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('QRunnable + QThreadPool')

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_worker)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.move(0, 30)
        self.stop_button.clicked.connect(self.stop_worker)

    def start_worker(self):
        if self.worker is None:
            self.worker = Worker()
            self.worker.signals.finished.connect(self.worker_finished)
            self.worker.signals.progress.connect(self.worker_progress)
            self.threadpool.start(self.worker)
        else:
            print("Поток уже запущен")

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()

    def worker_progress(self, val):
        print(f"Прогресс: {val}")

    def worker_finished(self):
        print("Worker завершил выполнение")
        self.worker = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())