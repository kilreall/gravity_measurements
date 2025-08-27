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
dt = 30e-3 # s
n = 101 # количество точек
T = 12e-3 # s временной интервал между пи импульсами
M = 0
Tg = 0.0027# T1:0.4;T2:0.089;T4:0.0226;T6:0.0109;T8:0.0061;T10:0.00357;T12:0.0027; # пристрелка периода для fitа
gR = 9.68
Tf = 29210*1e-6 # время прохода без
TF = Tf+2*T
ty = 10e-6 # s длительность pi/2 импульса
OR = np.pi/2/ty
r = 100000


# чтение csv P(a)
file_path = 'data\data_T_12_ms.csv' 
data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None)
data = np.array(data.tolist())

## создание многомерного массива данных
mdata = data[:n,0]/70*5282 # конечная частота рамановского излучения
mdata = np.vstack((mdata, (mdata-start_freq)/dt)) # скорости чирпирования
tidata = np.sin(k*T**2*(gR-2*np.pi*mdata[1]))

plt.plot(mdata[1], tidata)
vfunc = np.vectorize(fa)
for i in range(len(mdata[1])): # коррекция скорости чирпирования
    ta = np.linspace(0, TF, 54000)
    a = np.random.normal(loc=0.0, scale=1e-3/5, size=54000)
    v0 = float(np.random.normal(loc=0.0, scale=1e-3/5, size=1))
    v = [v0]
    for j in range(len(a)):
        v.append(v[-1]+a[j]*TF/54000)
    v = np.array(v) 
    t1, t2, t3, t4, t5, t6 = np.argmin(np.abs(ta-25080*1e-6)), np.argmin(np.abs(ta-25095*1e-6)), np.argmin(np.abs(ta-37095*1e-6)), np.argmin(np.abs(ta-37125*1e-6)), np.argmin(np.abs(ta-49125*1e-6)), np.argmin(np.abs(ta-49140*1e-6))
    fvibtest = k*(simps(v[t1:t2], ta[t1:t2])-2*simps(v[t3:t4],ta[t3:t4])+simps(v[t5:t6],ta[t5:t6]))
    tidata[i] = np.sin(k*(gR-2*np.pi*mdata[1, i])*T**2+fvibtest)
    # fat = vfunc(ta)
    # intvib = fat*a
    # fvib = k*simps(intvib, ta)
    # mdata[1,i] = mdata[1,i] - 2*np.pi*fvib/T**2

plt.plot(mdata[1], tidata)
plt.show()