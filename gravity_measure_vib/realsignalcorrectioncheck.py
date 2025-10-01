import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import stats
import matplotlib
import os
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.integrate import simps

def sins(x, A, w, ph, s):
    return A*np.sin(w*x+ph) + s

def csv_np(folder_path):
    # Получаем список файлов и сортируем по номеру
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')],
                    key=lambda x: int(x.split('.')[0]))

    # Считываем первый столбец из каждого CSV
    columns = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, header=None)  # Без заголовков
        column = df.iloc[:, 0].to_numpy()  # Первый столбец
        columns.append(column)

    return np.column_stack(columns)


def fitcoef(StI, TF):
    for i in range(len(chirp_rate)):
        acc_data = acc_mx[i]*TF
        fat = vfunc(ta[StI:StI+iTAI+1])
        intvib = fat*acc_data[StI:StI+iTAI+1]
        fvib = k*simps(intvib, ta[StI:StI+iTAI+1])
        chirp_rate[i] = chirp_rate[i] - fvib/T**2/(2*np.pi) # + or -, 2pi?
        initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, np.min(intensity)] 
        par, cov = curve_fit(sins, chirp_rate, intensity, p0=initial_guess)
        A, w, ph, s = par
        dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
        dg = 1/k/T**2/(A/dA)
        return dg*1e5

def aver(a):
    # усреднение
    trimmed_len = (len(a) // n) * n
    a = a[:trimmed_len]
    a = a.reshape(-1, n) # Переформатируем в двумерный массив (группы по n)
    a = a.mean(axis=0) # Считаем среднее по столбцам (ось 0)
    return a

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

vfunc = np.vectorize(fa)


c = 3*1e8
k =  (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9)/c + (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9 - 6.83468261090429*1e9)/c
k = k*2*np.pi
#print(k)
# start_freq = 90582400/70*5282
# dt = 30e-3 # s для чирпирования
n = 101-1 # количество точек
T = 10200e-6 # s временной интервал между пи импульсами
M = 0
Tg = 0.00357# T1:0.4;T2:0.089;T4:0.0226;T6:0.0109;T8:0.0061;T10:0.00357;T12:0.0027; # пристрелка периода для fitа
gR = 9.68
Tf = 30528*1e-6 # полное време подготовки атомов
ty = 25e-6 # s длительность pi/2 импульса
Tpause = 500e-3
TF = Tf+2*T+Tpause+4*ty # point time
TAI = 2*T+4*ty
OR = np.pi/2/ty
TRP = 33.556e-3 # время сбора данных red pitaya'ей
dt = TRP/16383 # Red Pitaya time step
ta = np.arange(0, 16384)*dt
iTAI = int(np.floor(TAI/dt))
r = 100000

# чтение csv P(a)
file_path = r'gravity_measure_vib/testdata/37290925191200/interference_signal.csv' 
data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None, skip_header=1)
data = np.array(data.tolist())

chirp_rate = data[:,0]
intensity = data[:,1]

plt.scatter(chirp_rate, intensity, color="black")

# start average data
chirp0 = chirp_rate[:n]
intensity0 = aver(intensity)

# plt.plot(chirp0, intensity0, color="orange")
# plt.scatter(chirp0, intensity0, color="orange")

# fit start data
initial_guess = [(np.max(intensity0) - np.min(intensity0))/2, 2*np.pi*T*T, 0, np.min(intensity0)] 
par, cov = curve_fit(sins, chirp0, intensity0, p0=initial_guess)
A, w, ph, s = par
dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
dg = 1/k/T**2/(A/dA)
print("sensitivity for noisy data =",dg*1e5*np.sqrt(TF*n), "mGal/.")
#plt.plot(chirp0, A*np.sin(w*chirp0+ph) + s, color="red")

TF = 1#4.098300562505257
StI = 0#44

# correct data
for i in range(len(chirp_rate)):
    acc_file = f'gravity_measure_vib/testdata/37290925191200/{i}.csv'
    acc_data = np.loadtxt(acc_file)/50/150*TF
    fat = vfunc(ta[StI:StI+iTAI+1])
    intvib = fat*acc_data[StI:StI+iTAI+1]
    fvib = k*simps(intvib, ta[StI:StI+iTAI+1])
    chirp_rate[i] = chirp_rate[i] - fvib/T**2/(2*np.pi) # + or -, 2pi?

#plt.plot(chirp_rate, intensity, color="green")
plt.scatter(chirp_rate, intensity, color="green")

# fit corrected data
# initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, np.min(intensity)] 
# par, cov = curve_fit(sins, chirp_rate, intensity, p0=initial_guess)
# A, w, ph, s = par
# dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
# dg = 1/k/T**2/(A/dA)
# print("sensetivity for correct data =", dg*1e5*np.sqrt(TF*n), 'mGal/.')
# chirp_rate = np.sort(chirp_rate)
# plt.plot(chirp_rate, A*np.sin(w*chirp_rate+ph) + s, color="blue")

# find coef for acc
# Диапазон возможных целых значений StI

sti_range = range(0, 100)#16384-iTAI)  # можно изменить

best_result = float('inf')
best_sti = None
best_tf = None


acc_mx = csv_np('gravity_measure_vib/testdata/37290925191200')/150/50
for sti in sti_range:
    # Минимизируем по TF для текущего sti
    res = minimize_scalar(lambda tf: fitcoef(sti, tf), bounds=(-5.0, 5.0), method='bounded')
    
    if res.fun < best_result:
        best_result = res.fun
        best_sti = sti
        best_tf = res.x

print("Minimal sensetivity:", best_result*np.sqrt(TF*n))
print("Optimal StI:", best_sti)
print("Optimal TF:", best_tf)

# plt.show()