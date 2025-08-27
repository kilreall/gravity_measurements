import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import stats
import matplotlib

from scipy.integrate import simps

def save_numpy_array(array, filename):
    """Сохраняет массив NumPy в файл"""
    np.save(filename, array)
    print(f"Массив сохранён в файл: {filename}.npy")

def load_numpy_array(filename):
    """Загружает массив NumPy из файла"""
    array = np.load(f"{filename}.npy")
    print(f"Массив загружен из файла: {filename}.npy")
    return array

def sins(x, A, w, ph, s):
    return A*np.sin(w*x+ph) + s

def sign(A):
    if A > 0:
        return 1
    else:
        return -1

def mm(w, ph):
    return round((w*gR+ph-sg*np.pi/2)/2/np.pi)

def gm(m, w, ph):
    return (sg*np.pi/2 + 2*np.pi*m-ph)/w

def fa(t):
    if 0 < t <= ty:
        return 2/OR*(1-np.cos(OR*t/2))
    elif ty < t <= ty+T:
        return t + 2/OR -ty
    elif ty+T < t <= 3*ty+T:
        return T + 2/OR*(1-np.cos(2/OR*(t-T)))     
    elif 3*ty+T < t <= 3*ty+2*T:
        return 2*T + 2/OR+3*ty-t       
    elif 3*ty+2*T < t <= 4*ty+2*T:
        return 2/OR*(1-np.cos(2/OR*(t-2*T)))    
    else:
        return 0            
    
c = 3*1e8
k =  (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9)/c + (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9 - 6.83468261090429*1e9)/c
k = k*2*np.pi
#print(k)
start_freq = 90582400/70*5282
dt = 30e-3 # s для чирпирования
n = 101 # количество точек
T = 12e-3 # s временной интервал между пи импульсами
M = 0
Tg = 0.0027# T1:0.4;T2:0.089;T4:0.0226;T6:0.0109;T8:0.0061;T10:0.00357;T12:0.0027; # пристрелка периода для fitа
gR = 9.68
Tf = 29210*1e-6 # полное време подготовки атомов
TF = Tf+2*T
ty = 40e-6 # s длительность pi/2 импульса
OR = np.pi/2/ty
r = 100000


# чтение csv P(a)
file_path = 'gravity measure vib\data\data_T_12_ms.csv' 
data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None)
data = np.array(data.tolist())

## создание многомерного массива данных
mdata = data[:n,0]/70*5282 # конечная частота рамановского излучения
n = 4*n # тестовое для модели
mdata = np.linspace(mdata[0], mdata[-1], n) # тестовое для модели
mdata = np.vstack((mdata, (mdata-start_freq)/dt)) # скорости чирпирования
tidata = np.sin(T*T*(k*gR-2*np.pi*mdata[1]))

filepath = "gravity measure vib\data\kkt11.csv"
adata = np.genfromtxt(filepath, delimiter=',', skip_header=1) 
ta = adata[:,0]
ta = ta-ta[0]
a = adata[:,1]/150
ta, indices = np.unique(ta, return_inverse=True)
a = np.bincount(indices, weights=a) / np.bincount(indices)

plt.figure(1)


a = np.interp(np.linspace(ta[0], ta[-1], len(ta)*10), ta, a)
ta = np.linspace(ta[0], ta[-1], len(ta)*10)

plt.plot(ta, a)

# Вычисляем FFT
N = len(ta)
dt = (ta[-1]-ta[0])/(N-1)
fft_a = np.fft.fft(a)  # Комплексные коэффициенты Фурье
freqs = np.fft.fftfreq(N, dt)



# 3. Преобразуем ускорение в скорость (V = A / (i * 2πf))
omega = 2 * np.pi * freqs
epsilon = 1e-10  # Чтобы избежать деления на 0
fft_v = np.zeros_like(fft_a, dtype=complex)
fft_v[1:] = fft_a[1:] / (1j * omega[1:])  # Игнорируем нулевую частоту (постоянная составляющая)

# 4. Обратное FFT → v(t)
v = np.fft.ifft(fft_v).real  # Отбрасываем мнимую часть (погрешности вычислений)
v = v - np.mean(v)
plt.plot(ta, v)


#print(ta)
plt.figure(2)
plt.plot(mdata[1], tidata)
vfunc = np.vectorize(fa)
fvibm = []

dl = int((4*ty+2*T)/dt)
t1, t2, t3, t4, t5, t6 = np.argmin(np.abs(0-ta)), np.argmin(np.abs(ty-ta))+1, np.argmin(np.abs(ty+T-ta)), np.argmin(np.abs(3*ty+T-ta))+1, np.argmin(np.abs(3*ty+2*T-ta)), np.argmin(np.abs(4*ty+2*T-ta))+1
for i in range(len(mdata[1])): # коррекция скорости чирпирования
    #print(t1, t2, t3, t4, t5, t6)
    t1, t2, t3, t4, t5, t6 = t1%(len(ta)-1), t2%(len(ta)-1), t3%(len(ta)-1), t4%(len(ta)-1), t5%(len(ta)-1), t6%(len(ta)-1)
    fvibtest = k*(simps(v[t1:t2], ta[t1:t2])-2*simps(v[t3:t4],ta[t3:t4])+simps(v[t5:t6],ta[t5:t6]))
    fvibm.append(fvibtest)
    tidata[i] = np.sin((k*gR-2*np.pi*mdata[1, i])*T**2+fvibtest)
    print(ta)
    print(t1, t6)
    print(len(ta))
    fat = vfunc(ta[t1:t6]-ta[t1])
    # intvib = fat*a[t1:t6]
    # fvib = k*simps(intvib, ta[t1:t6])
    t1, t2, t3, t4, t5, t6 = t1+dl, t2+dl, t3+dl, t4+dl, t5+dl, t6+dl
    # mdata[1,i] = mdata[1,i] - 2*np.pi*fvib/T**2

#print(t1, t2, t3, t4, t5, t6)

print(np.std(np.array(fvibm))/k/T/T*1e8)
initial_guess = [1, 2*np.pi*T*T, 0, 0] 
par, cov = curve_fit(sins, mdata[1], tidata, p0=initial_guess)
A, w, ph, s = par
sg = sign(A)
dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
dg = 1/k/T**2/(A/dA)
print(dg*1e8*np.sqrt(TF*n))


plt.scatter(mdata[1], tidata, color="orange")
plt.plot(mdata[1], tidata, color="orange")
plt.plot(mdata[1], A*np.sin(w*mdata[1]+ph) + s, color="green")
plt.show()