import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.size'] = 16
plt.rcParams["font.family"] = "Century Gothic"
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 1.5
#rcParams["legend.frameon"] = False

fig = plt.figure(1, (8,7))
a = np.linspace(0.8,1.6, 500)-1.2
p = (np.cos(a*2*np.pi/0.4)+1)/2
p1 = (np.cos(a*2*np.pi/0.2)+1)/2
plt.plot(a, p, label="T = 1 мс", color="b")
plt.plot(a, p1, label="T = 2 мс", color="r")
plt.xlabel("Отстройка скорости чирпирования α-α")
plt.ylabel("Населённость")
plt.legend()
plt.savefig("demonstr")
plt.show()