import numpy as np
import matplotlib.pyplot as plt

# accSensFunc_var2
def fa(t):
    if 0 <= t < ty:
        return (1+np.cos(OR*(t-2*ty)))/OR
    elif ty <= t < ty+T:
        return t + 1/OR -ty
    elif ty+T <= t < 3*ty+T:
        return T + (1+np.cos(OR*(t-T-2*ty)))/OR     
    elif 3*ty+T <= t < 3*ty+2*T:
        return 2*T + 1/OR+3*ty-t       
    elif 3*ty+2*T <= t < 4*ty+2*T:
        return 1/OR*(1+np.cos(OR*(t-2*T-2*ty)))    
    else:
        return 0  
    

T = 10e-3
ty = 20e-6
OR = np.pi/2/ty
t = np.linspace(0, 2*T+4*ty, 10000)

vfunc = np.vectorize(fa)

plt.plot(t, vfunc(t))

plt.show()