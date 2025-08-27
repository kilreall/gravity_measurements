import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sinss(x, A1, A2, A3, w1, w2, w3, ph1, ph2, ph3, ss):
    return A1*np.sin(w1*x+ph1) + A2*np.sin(w2*x+ph2) + A3*np.sin(w3*x+ph3) + ss


filepath = "data\llt1.csv"
data = np.genfromtxt(filepath, delimiter=',', skip_header=1) 
t = data[:,0]
a = data[:,1]/150
t, indices = np.unique(t, return_inverse=True)
a = np.bincount(indices, weights=a) / np.bincount(indices)
a = np.interp(np.linspace(t[0], t[-1], len(t)*10), t, a)
a = a - np.mean(a)
t = np.linspace(t[0], t[-1], len(t)*10)

# Вычисляем FFT
N = len(t)
dt = (t[-1]-t[0])/(N-1)
fft_a = np.fft.fft(a)  # Комплексные коэффициенты Фурье
freqs = np.fft.fftfreq(N, dt)



# 3. Преобразуем ускорение в скорость (V = A / (i * 2πf))
omega = 2 * np.pi * freqs
epsilon = 1e-10  # Чтобы избежать деления на 0
fft_v = np.zeros_like(fft_a, dtype=complex)
fft_v[1:] = fft_a[1:] / (1j * omega[1:])  # Игнорируем нулевую частоту (постоянная составляющая)

# 4. Обратное FFT → v(t)
v_t = np.fft.ifft(fft_v).real  # Отбрасываем мнимую часть (погрешности вычислений)



#print(t)
plt.plot(t, a/3)
plt.plot(t, v_t, 'r', label='Скорость v(t)')
#plt.plot(freqs[:N//2], np.abs(fft_a[:N//2]), label='Спектр |A(ω)|')

# initial_guess = [1.54, 0.947, 0.9842, 2*np.pi, 42.5*np.pi,  72.6*np.pi, 0, 0, 0, 0] 
# par, cov = curve_fit(sinss, t, a, p0=initial_guess)
# A1, A2, A3, w1, w2, w3, ph1, ph2, ph3, ss = par
# plt.plot(t, A1*np.sin(w1*t+ph1) + A2*np.sin(w2*t+ph2) + A3*np.sin(w3*t+ph3) + ss)

print(np.std(a))
print(np.std(v_t))
plt.show()

