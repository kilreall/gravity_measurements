import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

def sins(x, A, w, ph, s):
    return A*np.sin(w*x+ph) + s


c = 3*1e8
k =  (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9)/c + (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9 - 6.83468261090429*1e9)/c
#print(k)
start_freq = 90582400
dt = 2.5e-3 # s
n = 101
T = 1 # ms


# чтение csv
file_path = 'data\data_T_1_ms.csv'
data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None)
data = np.array(data.tolist())


# создание многомерного массива данных
mdata = data[:,0][:101]
mdata = np.vstack((mdata, (mdata-start_freq)/dt))
for i in range(data.shape[0]//101):
    mdata = np.vstack((mdata, data[:,1][101*i:101*(i+1)]))

tdata = data[:,0][:101]
tdata = np.vstack((tdata, (tdata-start_freq)/dt))
tdata = np.vstack((tdata, tdata[1]*2*np.pi/k))
tdata = np.vstack((tdata, data[:,1][:101]))
#print(tdata)

initial_guess = [3000, 2*np.pi/0.4, 0, np.mean(tdata[3])] 
par, cov = curve_fit(sins, tdata[2], tdata[3], p0=initial_guess)
A, w, ph, s = par
dw = cov[2,2]

dg = 2*np.pi*dw/w**2

print(A, w, ph, s, dg)
plt.scatter(tdata[2], tdata[3])
plt.plot(tdata[2], A*np.sin(w*tdata[2]+ph) + s)

plt.show()
