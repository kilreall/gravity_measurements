import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from matplotlib import rcParams

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
t_step = 0.5                            # sec
con = 5460 / 70
sweep_start = 87629900 * con            # Hz
sweep_stop = 87639900 * con             # Hz
sweep_time = 0.03                       # sec
scan = (scan * con - sweep_start) / sweep_time      # Hz/sec
scans = len(roi1) // points
size = len(roi1)
scan = np.tile(scan, scans+1)
print('Num of Scans =', scans)

# data array resizing
roi1 = roi1[:size]
roi2 = roi2[:size]
scan = scan[:size]

# norm data recalculation
norm = roi1 / (roi1 + roi2)

# fitting
init_guess = [np.amax(norm) - np.mean(norm),
              T,
              0,
              np.mean(norm)]


# fit function
def fit_func(alpha, a, T, phi, offset):
    return a * np.cos(2 * np.pi * alpha * T ** 2 + phi) + offset


# del_g calculation
deci = np.geomspace(24, 1, 10, dtype=int)
time = np.zeros(len(deci))
del_g = np.zeros(len(deci))

for i in range(len(deci)):
    step = deci[i]
    popt, pcov = curve_fit(fit_func, scan[::step], norm[::step], p0=init_guess)
    std = np.sqrt(np.diag(pcov))
    snr = popt[0] / std[0]
    # print('delta g =', abs(1 / (k * T ** 2 * snr)))
    del_g[i] = abs(1 / (k * T ** 2 * snr))
    time[i] = len(scan[::step]) * t_step


def fit_func_2(t, a):
    return a / np.sqrt(t)


# fit
popt2, pcov2 = curve_fit(fit_func_2, time, del_g)     # , p0=init_guess
fit_time = np.linspace(time[0], 900, 1000)
print(popt2)

# plotting
fig, ax = plt.subplots(figsize=(5.5, 5))
# plt.scatter(scan, aver)
# plt.plot(scan, fit_func(scan, *popt))
ax.plot(time, del_g, linewidth=1.8, color='tab:red', label='gravimeter data')
ax.scatter(time, del_g, s=25, color='tab:red')
ax.plot(fit_time, fit_func_2(fit_time, *popt2), linewidth=1.8, linestyle='--', color='tab:orange', label=r'$1/\sqrt{t}$ fit')

formatter = ticker.FuncFormatter(lambda x, _: f"{x:.1E}")  # Custom scientific notation formatter
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_minor_formatter(formatter)

ax.text(300, 4.05e-5, r"$\sigma_g(t)=3.9 \cdot 10^{-4} (m/s^2)/t^{1/2}$", fontsize=12, color="tab:orange")

# ax.set_xlim([70, 550])
# ax.set_ylim([1.65e-5, 1.75e-4])

ax.legend()

plt.xlabel(r'Integration Time, [s]')
plt.ylabel(r'$\sigma_g$, [$m/s^2$]')

plt.tight_layout()
plt.savefig('std.png')
plt.show()

