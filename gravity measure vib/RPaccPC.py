import sys
import redpitaya_scpi as scpi
import matplotlib.pyplot as plot

IP = 'rp-f066c8.local'

rp = scpi.scpi(IP)

rp.tx_txt()с
