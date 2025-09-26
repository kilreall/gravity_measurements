import sys
import redpitaya_scpi as scpi
import matplotlib.pyplot as plt
import time
import os


IP = 'rp-f05e99.local'

print("Enter decimation (example: 1024)")
dec = int(input())
dec = str("ACQ:DEC %d" % dec)

rp = scpi.scpi(IP)

# rp.tx_txt(dec)
rp.tx_txt(dec)

plt.ioff()
plt.ion()
fig, ax = plt.subplots()

while 1:
    command = ""
    if os.path.exists('commandRPacc.txt'):
        with open('commandRPacc.txt', 'r') as f:
            command = f.read().strip()
        os.remove('commandRPacc.txt')
    if command == "start" :
        rp.tx_txt('ACQ:RST')
        rp.tx_txt('ACQ:START')
        rp.tx_txt('ACQ:TRig NOW')


        while 1:
            rp.tx_txt('ACQ:TRig:STAT?')
            if rp.rx_txt() == 'TD':
                break

        rp.tx_txt('ACQ:SOUR1:DATA?')
        buff_string = rp.rx_txt()
        buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
        buff = list(map(float, buff_string))

        ax.clear()

        ax.plot(buff)
        ax.set_title("RF1 input")
        ax.set_xlabel("Counter")
        ax.set_ylabel("Voltage")
        # Обновляем отображение
        fig.canvas.draw()
        fig.canvas.flush_events()

        # time.sleep(140e-3)
    elif command == "break":
        break

plt.ioff()    