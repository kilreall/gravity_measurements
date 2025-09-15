from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import redpitaya_scpi as scpi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

name = 1

class WorkerSignals(QObject):
    finished = pyqtSignal()
    data = pyqtSignal(np.ndarray)  # Добавляем сигнал с данными

class Worker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self._is_running = True

    @pyqtSlot()
    def run(self):
        global name
        print("Поток запущен")
        IP = 'rp-f05e99.local'
        dec = 512
        trig_lvl = 0.1
        data_units = 'volts'
        data_format = 'ascii'
        acq_trig = 'CH1_PE'
        
        i = 1

        while self._is_running:
            
            rp = scpi.scpi(IP)

            rp.tx_txt('ACQ:RST')

            rp.tx_txt(f"ACQ:DEC:Factor {dec}")
            rp.tx_txt(f"ACQ:DATA:Units {data_units.upper()}")
            rp.tx_txt(f"ACQ:DATA:FORMAT {data_format.upper()}")

            rp.tx_txt(f"ACQ:TRig:LEV {trig_lvl}")

            rp.tx_txt('ACQ:START')
            rp.tx_txt(f"ACQ:TRig {acq_trig}")

            while self._is_running:
                rp.tx_txt('ACQ:TRig:STAT?')
                if rp.rx_txt() == 'TD':
                    break  
                time.sleep(0)
            
            if not self._is_running:
                rp.tx_txt('ACQ:RST')
                break

            rp.tx_txt('ACQ:SOUR1:DATA?')
            buff_string = rp.rx_txt()
            buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
            buff = np.array(buff_string).astype(np.float64)

            with open('stat.txt', 'r', encoding='utf-8') as file:
                content = file.read().strip()
                last_two = content[-2:] if len(content) >= 2 else content
                
            if last_two == '01':
                i = 0
                name += 1
                with open('file.txt', 'a', encoding='utf-8') as f:
                    f.write('1')  # добавляем символ '1' в конец файла

            np.savetxt('%d-%d.csv' % (name, i) , buff, delimiter=',')
            i += 1

            self.signals.data.emit(buff)  # Отправляем данные в основной поток

            with open('start.txt', 'w', encoding='utf-8') as file:
                file.write('0')

        time.sleep(0)  # чтобы не грузить CPU

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
        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('QRunnable + QThreadPool')

        self.start_button = QPushButton('Start trigger mode', self)
        self.start_button.clicked.connect(self.start_worker)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_worker)

        # Добавляем FigureCanvas для matplotlib графика
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Располагаем кнопки и график на окне
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

    def start_worker(self):
        if self.worker is None:
            self.worker = Worker()
            self.worker.signals.finished.connect(self.worker_finished)
            self.worker.signals.data.connect(self.update_plot)  # Подписываемся на данные
            self.threadpool.start(self.worker)
        else:
            print("Поток уже запущен")

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()

    def worker_finished(self):
        print("Worker завершил выполнение")
        self.worker = None

    @pyqtSlot(np.ndarray)
    def update_plot(self, buff):
        self.ax.clear()
        self.ax.plot(buff)
        self.ax.set_title("Данные с Red Pitaya")
        self.ax.set_xlabel("Отсчёты")
        self.ax.set_ylabel("Амплитуда")
        self.canvas.draw_idle()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())