import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rcParams

# plot settings
rcParams['font.size'] = 14
plt.rcParams["font.family"] = "Century Gothic"
plt.rcParams['savefig.dpi'] = 300
rcParams["legend.frameon"] = False

# constants and variables declaration
k = 2 * 2 * np.pi / 780.24e-9
T = 0.014                               # msec


# data import
_, roi1 = np.loadtxt('roi1_T_14_ms.csv', delimiter=',', skiprows=1, unpack=True)
_, roi2 = np.loadtxt('roi2_T_14_ms.csv', delimiter=',', skiprows=1, unpack=True)
_, norm = np.loadtxt('norm_T_14_ms.csv', delimiter=',', skiprows=1, unpack=True)
scan, _ = np.loadtxt('aver_T_14_ms.csv', delimiter=',', skiprows=1, unpack=True)

# scan data processing
points = 401
con = 5460 / 70
sweep_start = 87629900 * con            # Hz
sweep_stop = 87639900 * con             # Hz
sweep_time = 0.03                       # sec
scan = (scan[:points] * con - sweep_start) / sweep_time      # Hz/sec
scans = len(roi1) // points - 1
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
aver = np.sum(aver, axis=0) / scans


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
fit_scan = np.linspace(scan[0], scan[-1], 1000)

# data plotting
"""fig = plt.figure()
plt.plot(scan, aver)
plt.plot(scan, fit_func(scan, *popt))
plt.show()"""

# plotting
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot((scan - 26000944) * 1e-3, aver, linewidth=0.5, color='tab:red')
ax.scatter((scan - 26000944) * 1e-3, aver, s=5, color='tab:red', label=r'experiment: $2T=28$ ms')
ax.plot((fit_scan - 26000944) * 1e-3, fit_func(fit_scan, *popt), linewidth=2, color='tab:blue', label='fit')

ax.set_xlim([-(scan[-1] - 26000944) * 1e-3, (scan[-1] - 26000944) * 1e-3])
ax.set_ylim([0.095, 0.140])

ax.legend(loc='upper left', ncol=2)

plt.xlabel(r'Chirp rate detuning $\alpha - \alpha_0$, [kHz/s]')
plt.ylabel(r'Interference signal, [arb.un.]')

plt.tight_layout()
plt.savefig('interference_signal.png')
plt.show()

