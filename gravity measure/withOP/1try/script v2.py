import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib
import matplotlib.ticker as ticker
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False


# constants and variables declaration
k = 2 * 2 * np.pi / 780.24e-9
T = 0.012                               # msec


# data import
_, roi1 = np.loadtxt('roi1_T_12_ms.csv', delimiter=',', skiprows=1, unpack=True)
_, roi2 = np.loadtxt('roi2_T_12_ms.csv', delimiter=',', skiprows=1, unpack=True)
_, norm = np.loadtxt('norm_T_12_ms.csv', delimiter=',', skiprows=1, unpack=True)
scan, _ = np.loadtxt('aver_T_12_ms.csv', delimiter=',', skiprows=1, unpack=True)

# scan data processing
points = 401
t_step = 0.5                            # sec
con = 5460 / 70
sweep_start = 87629900 * con            # Hz
sweep_stop = 87639900 * con             # Hz
sweep_time = 0.03                       # sec
scan = (scan[:points] * con - sweep_start) / sweep_time      # Hz/sec
scans = len(roi1) // points
print('Num of Scans =', scans)

# data array resizing
roi1 = roi1[:points*scans]
roi2 = roi2[:points*scans]

# norm data recalculation
roi1_back = 0
roi2_back = 0
norm = (roi1 - roi1_back) / (roi1 + roi2 - roi1_back - roi2_back)

# aver data recalculation
aver = np.reshape(norm, (scans, points))
aver = np.sum(aver, axis=0)

# fitting
init_guess = [np.amax(aver) - np.mean(aver),
              T,
              0,
              np.mean(aver)]


# fit function
def fit_func(alpha, a, T, phi, offset):
    return a * np.cos(2 * np.pi * alpha * T ** 2 + phi) + offset


# del_g calculation
deci = np.geomspace(23, 1, 15, dtype=int)
# print(np.geomspace(50, 1, 6, dtype=int))
print(deci)
time = np.zeros(len(deci))
del_g = np.zeros(len(deci))

for i in range(len(deci)):
    step = deci[i]
    popt, pcov = curve_fit(fit_func, scan[::step], aver[::step], p0=init_guess)
    std = np.sqrt(np.diag(pcov))
    snr = popt[0] / std[0]
    # print('delta g =', abs(1 / (k * T ** 2 * snr)))
    del_g[i] = abs(1 / (k * T ** 2 * snr))
    time[i] = len(scan[::step]) * t_step * scans


matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
fig, ax = plt.subplots(figsize=(10, 6))
ax.ticklabel_format(axis='both', style='plain')
ax.scatter(scan*1e-6, aver, color="blue", label="эксперимент")
ax.plot(scan*1e-6, aver, color="blue", lw =0.5)
ax.plot(scan*1e-6, fit_func(scan, *popt), ls="--", lw=2.5, color="red", label="аппроксимация")
print("signal snr" , snr)#plt.plot(time, del_g)
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel("Скорость чирпирования, [МГц/с]", fontsize=14)
plt.ylabel("Сигнал флюоресценции, [отн.ед]", fontsize=14)
plt.legend(fontsize=14)

plt.savefig("AfalpT12")
plt.show()

