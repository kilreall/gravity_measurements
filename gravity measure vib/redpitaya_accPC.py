import socket
import numpy as np
import matplotlib.pyplot as plt
import csv

HOST = "rp-f05e99"   # IP Red Pitaya
PORT = 5000
BUFFER_SIZE = 16384
BYTES_PER_BLOCK = BUFFER_SIZE * 4  # float32

SAVE_CSV = "capture_ch1.csv"


def live_mode(sock):
    """Постоянная отрисовка"""
    sock.sendall(b"LIVE")
    plt.ion()
    fig, ax = plt.subplots()
    try:
        while True:
            data = sock.recv(BYTES_PER_BLOCK)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.float32)
            ax.clear()
            ax.plot(samples)
            ax.set_title("Live mode (CH1)")
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Остановлено пользователем")
        sock.sendall(b"STOP")


def capture_mode(sock, decimation):
    """Сбор одного блока данных"""
    fs = 125e6 / decimation
    sock.sendall(b"START")
    print(f"Захват одного блока ({BUFFER_SIZE} отсчётов) @ {fs/1e6:.2f} МГц")

    data = sock.recv(BYTES_PER_BLOCK)
    sock.sendall(b"STOP")

    samples = np.frombuffer(data, dtype=np.float32)
    print("Получено:", len(samples), "отсчётов")

    # === сохраняем в CSV ===
    with open(SAVE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CH1"])
        writer.writerows(zip(samples))
    print("Сохранено в:", SAVE_CSV)

    # === строим итоговый график ===
    t = np.arange(len(samples)) / fs
    plt.figure()
    plt.plot(t, samples, label="CH1")
    plt.title(f"Захваченный сигнал (CH1), decimation={decimation}")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда (В)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dec = int(input("Укажи decimation (1, 8, 64, 1024...): "))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("Подключено к Red Pitaya")

    # отправляем decimation серверу
    sock.sendall(f"DEC {dec}".encode())

    mode = input("Выбери режим (live/capture): ").strip().lower()
    if mode == "live":
        live_mode(sock)
    elif mode == "capture":
        capture_mode(sock, dec)
    else:
        print("Неизвестный режим")

    sock.sendall(b"EXIT")
    sock.close()