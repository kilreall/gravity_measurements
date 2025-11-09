import numpy as np
import matplotlib.pyplot as plt

T = 2e-3
a1 = 25.0e6
a2 = 25.225e6 
k = 2*2*np.pi/780e-9
g = 9.8125

a = np.linspace(a1, a2, 15)
ph = (2*np.pi*a - k*g) * T**2
P = (1 - np.cos(ph)) / 2

# Нормируем 'a'
a_norm = (a - np.mean(a)) / 50000

# Сохраняем исходные значения
a_orig = a_norm.copy()

# Добавляем шум
noise = np.random.normal(0, 1, len(a))
a_noisy = a_norm + noise

plt.figure(1)
# График исходных данных
plt.plot(a_norm, P, color="blue")
#plt.scatter(a_norm, P, color="blue")

# График зашумленных данных
plt.scatter(a_noisy, P, color="red")

plt.figure(2)
noise = noise/4
a_noisy = a_norm + noise
plt.plot(a_norm, P, color="blue")
plt.scatter(a_noisy, P, color="red")

# # Параметры отступа
# offset_frac = 0.25  # доля от длины стрелки, которую "обрезаем" с обоих концов

# # Соединяем стрелками с отступом
# for x0, x1, y in zip(a_orig[4:8], a_noisy[4:8], P[4:8]):
#     dx = x1 - x0
#     # уменьшаем длину стрелки на 2 * offset_frac и сдвигаем начало
#     x_start = x0 + dx * offset_frac
#     dx_short = dx * (1 - 2 * offset_frac)
#     plt.arrow(
#         x_start, y, dx_short, 0,
#         length_includes_head=True,
#         head_width=0.03,
#         head_length=0.05,
#         color='black',
#         alpha=1
#     )



plt.show()