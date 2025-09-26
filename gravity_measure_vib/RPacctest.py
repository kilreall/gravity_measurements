import sys
import redpitaya_scpi as scpi
import matplotlib.pyplot as plot
import time



IP = 'rp-f05e99.local'
#IP = '192.168.54.171'
acq_time = 134.218e-3 # s

rp = scpi.scpi(IP)

start_time = time.time()


rp.tx_txt('ACQ:RST')

rp.tx_txt('ACQ:DEC 512')
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

end_time = time.time()
execution_time = end_time - start_time
execution_time *= 1e3
print(f"Время выполнения: {execution_time:.4f} миллисекунд")

plot.plot(buff)
plot.ylabel('Voltage')


plot.show()