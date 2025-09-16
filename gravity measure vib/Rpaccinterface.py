from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLineEdit, QLabel, QSpinBox, QDoubleSpinBox, QHBoxLayout, QFileDialog



import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import redpitaya_scpi as scpi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

name = 1

# непрерывный режим
class Worker2Signals(QObject):
    finished = pyqtSignal()
    data = pyqtSignal(np.ndarray)  # Добавляем сигнал с данными

class Worker2(QRunnable):
    def __init__(self, ip, dec):
        super().__init__()
        self.ip = ip
        self.dec = dec
        self.signals = Worker2Signals()
        self._is_running = True

    @pyqtSlot()
    def run(self):
        print("Continious mode was started")

        IP = self.ip
        dec = self.dec
        data_units = 'volts'
        data_format = 'ascii'
        rp = scpi.scpi(IP)
        i = 0
        while self._is_running:

            rp.tx_txt('ACQ:RST')

            rp.tx_txt(f"ACQ:DEC:Factor {dec}")
            rp.tx_txt(f"ACQ:DATA:Units {data_units.upper()}")
            rp.tx_txt(f"ACQ:DATA:FORMAT {data_format.upper()}")

            rp.tx_txt('ACQ:START')
            rp.tx_txt('ACQ:TRig NOW')

            while 1:
                rp.tx_txt('ACQ:TRig:STAT?')
                if rp.rx_txt() == 'TD':
                    break

            rp.tx_txt('ACQ:SOUR1:DATA?')
            buff_string = rp.rx_txt()
            buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
            buff = np.array(buff_string).astype(np.float64)

            self.signals.data.emit(buff)
            print(i)
            i += 1
            time.sleep(0)

        print("continious mode was stopped")
        self.signals.finished.emit()

    def stop(self):
        self._is_running = False

# trigger mode acquisition
class WorkerSignals(QObject):
    finished = pyqtSignal()
    data = pyqtSignal(np.ndarray)  # Добавляем сигнал с данными

class Worker(QRunnable):
    def __init__(self, ip, dec, tl, path, ttime):
        super().__init__()
        self.ip = ip
        self.ttime = ttime
        self.dec = dec
        self. tl = tl
        self.path = path
        self.signals = WorkerSignals()
        self._is_running = True

    @pyqtSlot()
    def run(self):
        global name
        print("Поток запущен")

        IP = self.ip
        dec = self.dec
        ttime = self.ttime
        ttime = ttime*1e-3
        trig_lvl = self.tl
        path = self.path

        data_units = 'volts'
        data_format = 'ascii'
        acq_trig = 'CH1_PE'
        
        i = 0

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
                with open('stat.txt', 'a', encoding='utf-8') as f:
                    f.write('1')  # добавляем символ '1' в конец файла

            np.savetxt('%s/%d/%d.csv' % (path, name, i) , buff, delimiter=',')
            i += 1

            self.signals.data.emit(buff)  # Отправляем данные в основной поток

        time.sleep(0)  # чтобы не грузить CPU

        print("Flow was finished")
        self.signals.finished.emit()

    def stop(self):
        print("Flow stop")
        self._is_running = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ip = '192.168.1.100'
        self.threadpool = QThreadPool()
        self.worker = None
        self.worker2 = None
        self.initUI()

    def initUI(self):

        # file path
        self.path_label = QLabel("path to folder:")
        self.path_input = QLineEdit()
        self.path_input.setFixedSize(110, 25) 
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setFixedSize(100, 35)
        self.browse_button.clicked.connect(self.browse_folder)


        # IP window
        self.ip_label = QLabel("Enter IP:")
        self.ip_input = QLineEdit(self)
        self.ip_input.setFixedSize(110, 25) 
        self.ip_input.setText(self.ip)
        self.ip_input.setPlaceholderText("For example, 192.168.1.100")
        self.ip_input.textChanged.connect(self.on_ip_changed)

        # decimation window
        self.int_label = QLabel("Enter dec:")
        self.int_input = QSpinBox()
        self.int_input.setFixedSize(60, 25)
        self.int_input.setRange(0, 100000)
        self.int_input.setValue(1024) 

        # trig_lvl window
        self.trig_label = QLabel("Enter trig_lvl:")
        self.trig_input = QDoubleSpinBox()
        self.trig_input.setFixedSize(60, 25)
        self.trig_input.setRange(0, 10)
        self.trig_input.setValue(0.1) 

        # time acq window
        self.ttime_label = QLabel("Enter acq time: ms")
        self.ttime_input = QDoubleSpinBox()
        self.ttime_input.setFixedSize(60, 25)
        self.ttime_input.setRange(0, 1000)
        self.ttime_input.setDecimals(3) 
        self.ttime_input.setValue(134.218) 


        # Main window
        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('Accelerometer controller')

        self.start_button = QPushButton('Trigger mode', self)
        self.start_button.setFixedSize(100, 35)
        self.start_button.clicked.connect(self.start_worker)

        self.start_worker2_button = QPushButton('Continious mode', self)
        self.start_worker2_button.setFixedSize(100, 35)
        self.start_worker2_button.clicked.connect(self.start_worker2)

        self.stop_workers_button = QPushButton('Stop', self)
        self.stop_workers_button.setFixedSize(100, 35)
        self.stop_workers_button.clicked.connect(self.stop_workers)

        # Добавляем FigureCanvas для matplotlib графика
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Создаём главный горизонтальный layout
        main_layout = QHBoxLayout()

        # Левая вертикальная колонка с кнопками и полями ввода
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.ip_label)
        left_layout.addWidget(self.ip_input)
        left_layout.addWidget(self.int_label)
        left_layout.addWidget(self.int_input)
        left_layout.addWidget(self.start_worker2_button)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.trig_label)
        left_layout.addWidget(self.trig_input)
        left_layout.addWidget(self.ttime_label)
        left_layout.addWidget(self.ttime_input)
        left_layout.addWidget(self.path_label)
        left_layout.addWidget(self.path_input)
        left_layout.addWidget(self.browse_button)
        left_layout.addWidget(self.stop_workers_button)

        # Правая колонка с рисунком
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)

        # Добавляем левую и правую части в главный горизонтальный layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Устанавливаем главный layout на окно или виджет
        self.setLayout(main_layout)

    def browse_folder(self):
        # Открываем диалог выбора папки
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку", "")
        if folder:
            self.path_input.setText(folder)  # Записываем выбранный путь в QLineEdit


    def on_ip_changed(self, text):
        self.ip = text
        print("IP changed on:", self.ip)

    def start_worker(self):
        self.stop_workers()
        if self.worker is None:
            ip = self.ip_input.text()
            dec = self.int_input.value()
            ttime = self.ttime_input.value()
            tl = self.trig_lvl.value()
            path = self.path_input.text()
            self.worker = Worker(ip=ip, dec=dec, tl=tl, path=path, ttime=ttime)
            self.worker.signals.finished.connect(self.worker_finished)
            self.worker.signals.data.connect(self.update_plot)  # Подписываемся на данные
            self.threadpool.start(self.worker)
        else:
            print("The flow have already started")

    def start_worker2(self):
        self.stop_workers()
        if self.worker2 is None:
            ip = self.ip_input.text()
            dec = self.int_input.value()
            self.worker2 = Worker2(ip=ip,dec=dec)
            self.worker2.signals.finished.connect(self.worker2_finished)
            self.worker2.signals.data.connect(self.update_plot)
            self.threadpool.start(self.worker2)
        else:
            print("Второй поток уже запущен")

    def stop_workers(self):
        if self.worker is not None:
            print("Останавливаем worker")
            self.worker.stop()
        if self.worker2 is not None:
            print("Останавливаем worker2")
            self.worker2.stop()
        if self.worker is None and self.worker2 is None:
            print("Оба потока не запущены")

    def worker_finished(self):
        print("Worker завершил выполнение")
        self.worker = None

    def worker2_finished(self):
        print("Worker2 завершил выполнение")
        self.worker2 = None

    @pyqtSlot(np.ndarray)
    def update_plot(self, buff):
        self.ax.clear()
        self.ax.plot(buff)
        self.ax.set_title("CH1 RP voltage")
        self.ax.set_xlabel("counts")
        self.ax.set_ylabel("volts")
        self.canvas.draw_idle()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())