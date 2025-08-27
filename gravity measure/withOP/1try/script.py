import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

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


# fit function
def fit_func(alpha, a, T, phi, offset):
    return a * np.cos(2 * np.pi * alpha * T**2 + phi) + offset


init_guess = [np.amax(aver)-np.mean(aver),
              T,
              0,
              np.mean(aver)]
popt, pcov = curve_fit(fit_func, scan, aver, p0=init_guess)
std = np.sqrt(np.diag(pcov))
snr = popt[0] / std[0]
print('delta g =', abs(1 / (k * T**2 * snr)))

fig = plt.figure()
plt.plot(scan, aver)
plt.plot(scan, fit_func(scan, *popt))
plt.show()

