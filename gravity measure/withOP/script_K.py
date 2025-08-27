import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import stats

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

c = 3*1e8
k =  (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9)/c + (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9 - 6.83468261090429*1e9)/c
k = k*2*np.pi
#print(k)
con = 5460 / 70
dt = 30e-3 # s
n = 401
T = 12e-3 # ms
M = 0
Tg = 0.0027# T1:0.4;T2:0.089;T4:0.0226;T6:0.0109;T8:0.0061;T10:0.00357;T12:0.0027;
gR = 9.68
Tf = 29210*1e-6
TF = Tf+2*T

r = 100000


# чтение csv
file_path = 'aver_T_12_ms.csv' 
data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None)
data = np.array(data.tolist())




# создание многомерного массива данных
mdata = data[:,0][:101]/70*5460 # частота рамановского излучения
mdata = np.vstack((mdata, (mdata-start_freq)/dt)) # скорости чирпирования
mdata = np.vstack((mdata, mdata[1]*2*np.pi/k)) # ускорение geff
#print("границы g",mdata[2][-1], mdata[2][0]) # нахождение границ выборки данных по g
#save_numpy_array(mdata[2], "data\gT1") # накопление g для нахождение g
N = []
dgM = []
G = []
for i in range(data.shape[0]//101):
    mdata = np.vstack((mdata, data[:,1][101*i:101*(i+1)])) # присоединение прохода

    # fit каждого прохода
    initial_guess = [3000, 2*np.pi/Tg, 0, np.mean(mdata[i+3])] 
    par, cov = curve_fit(sins, mdata[2], mdata[i+3], p0=initial_guess)
    A, w, ph, s = par
    sg = sign(A)
    dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
    if i != 0:
        M = 0
    m = mm(w, ph) + M
    #print(m)
    gcs = gm(m, w, ph)
    if G ==[]:
        G.append(gcs)
    while abs(gcs - np.mean(np.array(G))) > abs(gm(m+1, w, ph) - np.mean(np.array(G))) or abs(gcs - np.mean(np.array(G))) > abs(gm(m-1, w, ph) - np.mean(np.array(G))):
        if abs(gcs - np.mean(np.array(G))) > abs(gm(m+1, w, ph) - np.mean(np.array(G))):
            m += 1
            gcs = gm(m, w, ph)
        else: 
            m -= 1
            gcs = gm(m, w, ph)
    G.append(gcs)
    #print("номер минимума", m)
    
    # sigma от кол-ва усреднений
    N.append((i+1)*TF*1e3)
    GG = np.array(G)
    gcc = np.mean(GG)
    gcc_std_dev = np.std(GG, ddof=1)
    dgcc = gcc_std_dev / np.sqrt(len(GG))
    dgM.append(dgcc)

    # debug отрисовка
    if i == -1:
        plt.figure(3)
        plt.plot(mdata[2], mdata[3+i])
        plt.plot(mdata[2], A*np.sin(w*mdata[2]+ph) + s)
        plt.xlabel("acceleration, m/s2")
        plt.ylabel("signal")

    # выбрасывание не подходящих
    if i == -1 or i == -1: # Для T=2 i=5 и i=9
        print("выкинутый i", i)
        #if np.mean(np.array(dgM[-2])) + np.mean(np.array(G[:-2])) < G[-1] or np.mean(np.array(dgM[-2])) + np.mean(np.array(G[:-2])) > G[-1]:
        G.pop()
        N.pop()
        dgM.pop()
        np.delete(mdata, -1)

# нахождение g и dg по усреднению g
G = np.array(G)
print("полученные для усреднений g:", G*r)
gcs = np.mean(G)
gcs_std_dev = np.std(G, ddof=1)
dgcs = gcs_std_dev / np.sqrt(len(G))
print("g[усреднение g] =", gcs*r, "mGal")
print("dg[усреднение g] =",abs(dgcs*r), "mGal")


# усреднение данных
ndata = np.sum(mdata[3:], axis=0)/len(mdata[3:])
#save_numpy_array(ndata, "data\sT1") # накопление величины сигнала для нахождение g
initial_guess = [3000, 2*np.pi/Tg, 0, np.mean(ndata)] 
par, cov = curve_fit(sins, mdata[2], ndata, p0=initial_guess)
A, w, ph, s = par
print("параметры fit для совмещения данных",A, w, ph, s) # параметры для усреднения
print("SNR",abs(A/dA)) #SNR от T
dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
sg = sign(A)
m = mm(w, ph) + M
gca = gm(m, w, ph)
dgT = gca*dw/w
print("g[усреднение данных] =", gca*r, "mGal")
print("dg[усреднение данных по T] =", abs(dgT)*r, "mGal")
dgca = (-dph-gca*dw)/w
print("dg[усреднение данных по стандартной формуле] =", abs(dgca)*r, "mGal")
dgfa = 1/k/T**2/(A/dA)
print("dg[усреднение данных по специальное формуле] =", abs(dgfa*r), "mGal")

## тестовый образец
#tdata = data[:,0][:101]/70*5282
#tdata = np.vstack((tdata, (tdata-start_freq)/dt))
#tdata = np.vstack((tdata, tdata[1]*2*np.pi/k))
#tdata = np.vstack((tdata, data[:,1][:101]))
#print(tdata)


## Для единичного прохода
#initial_guess = [3000, 2*np.pi/Tg, 0, np.mean(tdata[3])] 
#par, cov = curve_fit(sins, tdata[2], tdata[3], p0=initial_guess)
#A, w, ph, s = par
#print(A, w, ph, s)
#dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
#m = int((w*gR+ph+np.pi/2)/2/np.pi) + M
#g = (-np.pi/2 + 2*np.pi*m-ph)/w
#print("g =", g*r, "mGal")

#dg1 = (-dph-g*dw)/w
#print("dg1 =", abs(dg1)*r, "mGal")

#dg2 = 1/k/T**2/(A/dA)
#print("dg2 =",dg2*r, "mGal")

##отрисовка усредненных данных и их fit
plt.figure(1)
plt.plot(mdata[1], ndata)
plt.plot(mdata[1], A*np.sin(w*mdata[2]+ph) + s)
print("signal std" ,np.std(ndata-A*np.sin(w*mdata[2]+ph) + s, ddof=1)/A)
plt.xlabel("сhirprate")
plt.ylabel("signal")


## зависимость dg от количества усреднений
#plt.figure(2)
#plt.plot(np.array(N), np.array(dgM)*r)
#plt.xlabel("время интегрирования, мс")
#plt.ylabel("стандартное отклонение среднего, mGal")


# совмещенные данных для разных T
# gs = 9.5# 9.328301458448493 для T=1
# gf = 9.9 # 10.31022792775102 для T=1
# gls = np.linspace(gs, gf, 1500)

# plt.figure(4)

# A1, w1, ph1, s1 = 2801.37776081057, 16.98437130137184, -12.130723180112293, 29275.530883687716
# plt.plot(gls, np.sin(w1*gls+ph1)+1, label="T1")

# A2, w2, ph2, s2 = -1571.7705404116696, 65.9439409927218, 44.61724247057551, 16625.20469983374
# plt.plot(gls, -np.sin(w2*gls+ph2)+1, label="T2")

# A3, w3, ph3, s3 = 2064.93534398503, 259.41673520183184, 254.17921081097612, 20832.64583135989
# plt.plot(gls, np.sin(w3*gls+ph3)+1, label="T4")

# A4, w4, ph4, s4 = 2684.1445628545357, 585.3098740484558, -90.47259630643168, 18944.044856421962
# plt.plot(gls, np.sin(w4*gls+ph4) + 1, label="T6")

# A5, w5, ph5, s5 = 2972.2696830264385, 1036.837430207476, -66.0338609061255, 17763.330046877283
# plt.plot(gls, np.sin(w5*gls+ph5) + 1, label="T8")

# A6, w6, ph6, s6 = 2167.581148124417, 1618.8990492653938, 1373.286707391665, 15580.991489705288
# plt.plot(gls, np.sin(w6*gls+ph6) + 1, label="T10")

# A7, w7, ph7, s7 = -2516.7290380427253, 2327.558886759566, -3.766630371594933, 15461.50759575267
# plt.plot(gls, -np.sin(w7*gls+ph7) + 1, label="T12")

# plt.xlabel("acceleration, mGal")
# plt.ylabel("signal")
# plt.legend()

## зависимость SNR от T
#plt.figure(5)
#T = [1, 2, 4, 6, 8, 10, 12]
#SNR = [16.991818930743687,8.146747823691864, 10.065942386994363, 12.99529758216232, 18.780507096049675, 8.794436183350287, 14.502494217851535]
#plt.plot(T, SNR)
#plt.xlabel("T, мс")
#plt.ylabel("SNR")

# plt.figure(6)
# T = [1, 2, 4, 6, 8, 10, 12]
# g = [968454, 968452, 968320, 967905.7, 968234, 968223.82, 968125.6]
# dg = [140, 47, 14, 3.8, 1.8,0.86, 1.1]
# plt.plot(T, g)
# plt.errorbar(T, g, yerr=dg, fmt='o', color='b', markersize=5, capsize=3)
# plt.xlabel("T, ms")
# plt.ylabel("g, mGal")
# plt.legend()

plt.show()