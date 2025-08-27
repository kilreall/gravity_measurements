import numpy as np
import matplotlib.pyplot as plt

filepath = "data\yt0.csv"
data = np.genfromtxt(filepath, delimiter=',', skip_header=1) 
t = data[:,0]*1e3
a = data[:,1]/150*1e2*1e3
t, indices = np.unique(t, return_inverse=True)
a = np.bincount(indices, weights=a) / np.bincount(indices)

print(t)
plt.plot(t, a)

print(np.mean(a))
print(np.std(a))

plt.show()

