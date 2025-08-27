import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import stats

def sins(x, A, w, ph, s):
    return A*np.sin(w*x+ph) + s


c = 3*1e8
k =  (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9)/c + (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9 - 6.83468261090429*1e9)/c
k = k*2*np.pi
#print(k)
start_freq = 90582400/70*5282
dt = 30e-3 # s
n = 101
T = 1e-3 # ms
m = 25

# чтение csv
file_path = 'data\data_T_1_ms.csv'
data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None)
data = np.array(data.tolist())




# создание многомерного массива данных
mdata = data[:,0][:101]/70*5282
mdata = np.vstack((mdata, (mdata-start_freq)/dt))
mdata = np.vstack((mdata, mdata[1]*2*np.pi/k))
G = []
for i in range(data.shape[0]//101):
    mdata = np.vstack((mdata, data[:,1][101*i:101*(i+1)]))

    initial_guess = [3000, 2*np.pi/0.4, 0, np.mean(mdata[i+3])] 
    par, cov = curve_fit(sins, mdata[2], mdata[i+3], p0=initial_guess)
    A, w, ph, s = par
    dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
    gcs = (-np.pi/2 + 2*np.pi*m-ph)/w
    G.append(gcs)



G = np.array(G)
gcs = np.mean(G)
gcs_std_dev = np.std(G, ddof=1)
dgcs = gcs_std_dev / np.sqrt(len(G))
print("gcs =", gcs*100000)
print("dgcs =", dgcs*100000)

ndata = np.sum(mdata[3:], axis=0)/len(mdata[3:])
initial_guess = [3000, 2*np.pi/0.4, 0, np.mean(ndata)] 
par, cov = curve_fit(sins, mdata[2], ndata, p0=initial_guess)
A, w, ph, s = par
dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])

gca = (-np.pi/2 + 2*np.pi*m-ph)/w
print("x", dph, gca*dw)
dgT = gca*dw/w
print("dgT =", abs(dgT)*100000)
print("gca =", gca*100000)
dgca = (-dph-gca*dw)/w
print("dgca =", abs(dgca)*100000)
dgfa = 1/k/T**2/(A/dA)
print("dgfa =",dgfa*100000)


tdata = data[:,0][:101]/70*5282
tdata = np.vstack((tdata, (tdata-start_freq)/dt))
tdata = np.vstack((tdata, tdata[1]*2*np.pi/k))
tdata = np.vstack((tdata, data[:,1][:101]))
#print(tdata)



initial_guess = [3000, 2*np.pi/0.4, 0, np.mean(tdata[3])] 
par, cov = curve_fit(sins, tdata[2], tdata[3], p0=initial_guess)
A, w, ph, s = par
#print(A, w, ph, s)
dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])

g = (-np.pi/2 + 2*np.pi*m-ph)/w
print("g =", g*100000)

dg1 = (-dph-g*dw)/w
print("dg1 =", abs(dg1)*100000)

dg2 = 1/k/T**2/(A/dA)
print("dg2 =",dg2*100000)

plt.scatter(tdata[2], tdata[3])
plt.plot(tdata[2], A*np.sin(w*tdata[2]+ph) + s)

plt.show()
