import socket
import numpy as np
import rp

HOST = ''       # слушаем все интерфейсы
PORT = 5000
BUFFER_SIZE = 16384  # число сэмплов за блок

# === Инициализация RP API ===
rp.rp_Init()

# === TCP сервер ===
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

print(f"Ожидание подключения на порту {PORT}...")
conn, addr = s.accept()
print("Подключился клиент:", addr)

running = False
mode_live = False

try:
    while True:
        # ждём команду от клиента
        cmd = conn.recv(32).decode().strip()
        if not cmd:
            break

        if cmd.startswith("DEC "):
            dec_value = int(cmd.split()[1])
            print(f"⚙ Установка decimation = {dec_value}")
            rp.rp_AcqReset()
            rp.rp_AcqSetDecimation(dec_value)
            rp.rp_AcqSetTriggerSrc(rp.RP_TRIG_SRC_DISABLED)
            rp.rp_AcqStart()

        elif cmd == "LIVE":
            print("▶ Live режим")
            mode_live = True
            running = True

        elif cmd == "START":
            print("▶ Capture режим: один блок")
            mode_live = False
            running = True

        elif cmd in ["STOP", "EXIT"]:
            print("⏹ Остановка / выход")
            running = False
            if cmd == "EXIT":
                break

        # === поток данных ===
        while running:
            state = rp.bool_t()
            rp.rp_AcqGetBufferFillState(state)
            if state.value:
                size = BUFFER_SIZE
                data_ch1 = (rp.floatArray)(size)
                rp.rp_AcqGetOldestDataV(rp.RP_CH_1, size, data_ch1)

                arr = np.array([data_ch1[i] for i in range(size)], dtype=np.float32)
                conn.sendall(arr.tobytes())

                if not mode_live:  # capture → только один блок
                    running = False

except KeyboardInterrupt:
    print("Сервер остановлен")

finally:
    conn.close()
    s.close()
    rp.rp_Release()