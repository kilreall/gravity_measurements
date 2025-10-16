import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import stats
import matplotlib
import os
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson
from joblib import Parallel, delayed
from scipy.signal import find_peaks


def aver(a):
    # усреднение
    trimmed_len = (len(a) // n) * n
    a = a[:trimmed_len]
    a = a.reshape(-1, n) # Переформатируем в двумерный массив (группы по n)
    a = a.mean(axis=0) # Считаем среднее по столбцам (ось 0)
    return a


def sins(x, A, w, ph, s):
    return A*np.sin(w*x+ph) + s

def csv_np(folder_path):
    file_list = [
        f for f in os.listdir(folder_path)
        if f.endswith('.csv') and f.split('.')[0].isdigit()
    ]
    file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))

    columns = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, header=None)
        column = df.iloc[:, 0].to_numpy()
        columns.append(column)

    return np.column_stack(columns).transpose()

# # coef finding var1

# def fitcoef(StI, TFF):
#     # Используем глобальные: chirp_rate, intensity, acc_mx, ta, etc.
#     try:
#         local_chirp = chirp_rate.copy()
#         for i in range(len(local_chirp)):
#             acc_data = acc_mx[i] * TFF
#             intvib = fat * acc_data[StI:StI+iTAI+1]
#             fvib = k * simpson(intvib,x=tan)
#             local_chirp[i] = local_chirp[i] - fvib / T**2 / (2*np.pi)

#         initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, np.min(intensity)] 
#         par, cov = curve_fit(sins, local_chirp, intensity, p0=initial_guess, maxfev=10000)
#         A, w, ph, s = par
#         dA = np.sqrt(cov[0,0])
#         dg = 1/k/T**2/(A/dA)
#         return dg * 1e5
#     except Exception as e:
#         return np.inf  # На случай, если curve_fit не сойдется

# def optimize_for_sti(StI):
#     res = minimize_scalar(lambda TFF: fitcoef(StI, TFF), bounds=(-5.0, 5.0), method='bounded')
#     return (StI, res.x, res.fun)

# def optimalFind(): # find coef for acc
#     # Диапазон возможных целых значений StI

#     sti_range = range(0, 10)  # Диапазон значений StI
#     results = Parallel(n_jobs=-1)(delayed(optimize_for_sti)(sti) for sti in sti_range)

#     # Найдём лучший результат
#     best_sti, best_tf, best_result = min(results, key=lambda x: x[2])

#     print("Minimal sensitivity:", best_result * np.sqrt(TF * n))
#     print("Optimal StI:", best_sti)
#     print("Optimal TF:", best_tf)

# # coef finding var2

# def fitcoef(TFF, fvib, A, w, ph, s):

#     fvib = fvib*TFF
#     local_chirp = chirp_rate.copy()
#     local_chirp = local_chirp - fvib / T**2 / (2*np.pi)
#     fit_intensity = A*np.sin(w*local_chirp+ph)+s
#     intensity_diff = intensity-fit_intensity

#     return np.std(intensity_diff)

# def optimize_for_sti(A, w, ph, s, StI, a, b):

#     intvib = acc_mx[:, StI:StI+iTAI+1]*fat
#     fvib = k*simpson(y=intvib, x=tan, axis=-1)

#     res = minimize_scalar(lambda TFF: fitcoef(TFF, fvib, A, w, ph, s), bounds=(a, b), method='bounded')
#     return (StI, res.x, res.fun)

# def optimalFind(delay, a , b): # find coef for acc
    
#     # starting fit finder
#     initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, np.min(intensity)] 
#     par, cov = curve_fit(sins, chirp_rate, intensity, p0=initial_guess)
#     A, w, ph, s = par

#     # Диапазон возможных целых значений StI
#     StI_range = range(0, delay)  # Диапазон значений StI
#     results = Parallel(n_jobs=-1)(delayed(optimize_for_sti)(A, w, ph, s, StI, a, b) for StI in StI_range)

#     # Найдём лучший результат
#     best_sti, best_tf, best_result = min(results, key=lambda x: x[2])

#     # print("Minimal sensitivity:", best_result * np.sqrt(TF * n))
#     print("Optimal StI:", best_sti)
#     print("Optimal TF:", best_tf)

# coef finding var3

def fitcoef(TFF, fvib):

    try:
        fvib = fvib*TFF
        local_chirp = chirp_rate.copy()
        local_chirp = local_chirp - fvib / T**2 / (2*np.pi)

        initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, np.min(intensity)] 
        par, cov = curve_fit(sins, local_chirp, intensity, p0=initial_guess, maxfev=10000)
        A, w, ph, s = par
        dA = np.sqrt(cov[0,0])
        dg = 1/k/T**2/(A/dA*sk)
        dg = abs(dg)
        # dg = 1/k/T**2/(A/dA)**2
        # noise = intensity - (A*np.sin(w*local_chirp+ph)+s)
        # SNR = np.mean(intensity**2)/np.mean(noise**2)
        # dg = 1/k/T**2/SNR
        return dg * r
    
    except Exception as e:
       return np.inf  # На случай, если curve_fit не сойдется

def optimize_for_sti(StI, a, b):

    intvib = acc_mx[:, StI:StI+iTAI+1]*fat
    fvib = k*simpson(y=intvib, x=tan, axis=-1)

    res = minimize_scalar(lambda TFF: fitcoef(TFF, fvib), bounds=(a, b), method='bounded')
    return (StI, res.x, res.fun)

def optimalFind(delay, a , b): # find coef for acc
    

    # Диапазон возможных целых значений StI
    StI_range = range(0, delay)  # Диапазон значений StI
    results = Parallel(n_jobs=-1)(delayed(optimize_for_sti)(StI, a, b) for StI in StI_range)

    # Найдём лучший результат
    best_sti, best_tf, best_result = min(results, key=lambda x: x[2])

    print("Minimal sensitivity:", best_result * np.sqrt(TF * n))
    print("Optimal StI:", best_sti)
    print("Optimal TF:", best_tf)



def singleWork(StI, TFF):
    global chirp_rate, intensity, TF, n

    chirp0 = chirp_rate[:n]
    intensity0 = aver(intensity)
    #plt.plot(chirp0, intensity, color="black")



    # fit start data
    initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, np.min(intensity)] 
    par, cov = curve_fit(sins, chirp_rate, intensity, p0=initial_guess)
    A, w, ph, s = par
    dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
    dgE = 1/k/T**2/(A/dA*sk)
    dgE = abs(dgE)
    # noise = intensity - (A*np.sin(w*chirp_rate+ph)+s)

    # # Производная фита
    # ddt_fit = np.gradient(A*np.sin(w*chirp_rate+ph)+s, chirp_rate)

    # # Нахождение всех локальных максимумов абсолютного значения производной
    # # Порог для пиков: height=0.1 (минимальная высота, адаптируйте), prominence загружаем
    # peaks, _ = find_peaks(np.abs(ddt_fit), height=0.1, prominence=None)  # height — минимальная амплитуда пика

    # # Если пиков слишком мало/много, отрегулируйте параметры find_peaks
    # if len(peaks) == 0:
    #     print("Пики не найдены; проверьте параметры или данные.")
    #     peaks = [np.argmax(np.abs(ddt_fit))]  # Fallback к одному максимуму

    # # Окрестность вокруг каждого пика (например, ±5 индексов)
    # window = 5
    # selected_indices = set()  # Unique индексы
    # for peak_idx in peaks:
    #     start_idx = max(0, peak_idx - window)
    #     end_idx = min(len(ddt_fit), peak_idx + window + 1)
    #     for idx in range(start_idx, end_idx):
    #         selected_indices.add(idx)

    # # Преобразование в список для сортировки
    # selected_list = sorted(list(selected_indices))

    # # sigmaP: стандартное отклонение шума в окрестностях всех пиков
    # sigmaP = np.std(noise[selected_list])
    # SNR = abs(A)/sigmaP
    # dgE = 1/k/T**2/SNR

    print("sensitivity for experimental data =", dgE*np.sqrt(TF*n)*r, "mGal/.")

    plt.figure(1)
    plt.scatter(chirp0, intensity0, color="black", label="averaged", s=10)
    plt.scatter(chirp_rate, intensity, color="orange", label="experimental", s=10)
    plt.xlabel('chirp rate')
    plt.ylabel('signal')
    plt.plot(chirp0, A*np.sin(w*chirp0+ph) + s, color="orange")

    # fit average data
    initial_guess = [(np.max(intensity0) - np.min(intensity0))/2, 2*np.pi*T*T, 0, np.min(intensity0)] 
    par, cov = curve_fit(sins, chirp0, intensity0, p0=initial_guess)
    Aa, wa, pha, sa = par
    dwa, dpha, dAa = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
    #dgC = 1/k/T**2/(A/dA)
    #dgC = 1/k/T**2/(A/dA)**2
    #noise = intensity0 - (Aa*np.sin(wa*chirp0+pha)+sa)
    #SNR = np.mean(intensity0**2)/np.mean(noise**2)
    dgAv = 1/k/T**2/(Aa/dAa*sk)
    dgAv = abs(dgAv)
    print("sensetivity for averaged data =", dgAv*np.sqrt(TF*n)*r, 'mGal/.')
    plt.plot(chirp0, Aa*np.sin(wa*chirp0+pha) + sa, color="black")

    # # sensetivity for averaged g
    # dgMX = []
    # g0 = 9.955
    # for j in range(len(intensity)//n):
    #     initial_guess = [(np.max(intensity[j*n:j*n+n]) - np.min(intensity[j*n:(j+1)*n]))/2, 2*np.pi*T*T, 0, np.min(intensity[j*n:(j+1)*n])] 
    #     par, cov = curve_fit(sins, chirp0, intensity[j*n:n*(j+1)], p0=initial_guess)
    #     Aa, wa, pha, sa = par

    #     if j == 0:
    #         m = round((wa*k*g0/2/np.pi+pha-np.pi/2)/np.pi)
    #         gj = (np.pi/2+np.pi*m-pha)/wa*2*np.pi/k
    #         dgMX.append(gj)
    #     else:
    #         np_buff = np.array(gj)
    #         mean_buff = np.mean(np_buff)
    #         m = round((wa*k*g0/2/np.pi+pha-np.pi/2)/np.pi)
    #         gj0 = (np.pi/2+np.pi*m-pha)/wa*2*np.pi/k
    #         gj1 = (np.pi/2+np.pi*(m+1)-pha)/wa*2*np.pi/k
    #         gj_1 = (np.pi/2+np.pi*(m-1)-pha)/wa*2*np.pi/k
    #         if abs(gj1-mean_buff) < abs(gj0-mean_buff):
    #             m += 1
    #             gj0 = gj1
    #             gj1 = (np.pi/2+np.pi*(m+1)-pha)/wa*2*np.pi/k
    #             while abs(gj1-mean_buff) < abs(gj0-mean_buff):
    #                 m += 1
    #                 gj0 = gj1
    #                 gj1 = (np.pi/2+np.pi*(m+1)-pha)/wa*2*np.pi/k
    #         elif abs(gj_1-mean_buff) < abs(gj0-mean_buff):
    #             m -= 1
    #             gj0 = gj_1
    #             gj_1 = (np.pi/2+np.pi*(m-1)-pha)/wa*2*np.pi/k
    #             while abs(gj_1-mean_buff) < abs(gj0-mean_buff):
    #                 m -= 1
    #                 gj0 = gj_1
    #                 gj_1 = (np.pi/2+np.pi*(m-1)-pha)/wa*2*np.pi/k
    #         dgMX.append(gj0)    
    # dgMX = np.array(dgMX)
    # print("list of g", dgMX)
    # print("sensetivity for averaged g =",np.std(dgMX)/np.sqrt(len(dgMX))*np.sqrt(TF*n)*r, 'mGal/.')

    # evaluate vibration influence block1
    intensity_clear = A*np.sin(w*chirp_rate+ph)+s
    chirp_copy = chirp_rate
    t1, t2, t3, t4, t5, t6 = np.argmin(np.abs(0-ta)), np.argmin(np.abs(ty-ta))+1, np.argmin(np.abs(ty+T-ta)), np.argmin(np.abs(3*ty+T-ta))+1, np.argmin(np.abs(3*ty+2*T-ta)), np.argmin(np.abs(4*ty+2*T-ta))+1
    NF = 16384

    # # correct data ver1
    # for i in range(len(chirp_rate)):

    #     acc_data = acc_mx[i] * TFF

    #     # data correction
    #     intvib = fat*acc_data[StI:StI+iTAI+1]
    #     fvib = k*simpson(y=intvib, x=tan)
    #     chirp_rate[i] = chirp_rate[i] - fvib/T**2/(2*np.pi) # + or -, 2pi?

    #     # evaluate vibration influence block3

    #     # # find FFT
    #     # fft_a = np.fft.fft(acc_data)  # Комплексные коэффициенты Фурье
    #     # freqs = np.fft.fftfreq(NF, dt)

    #     # # 3. Преобразуем ускорение в скорость (V = A / (i * 2πf))
    #     # omega = 2 * np.pi * freqs
    #     # epsilon = 1e-10  # Чтобы избежать деления на 0
    #     # fft_v = np.zeros_like(fft_a, dtype=complex)
    #     # fft_v[1:] = fft_a[1:] / (1j * omega[1:])  # Игнорируем нулевую частоту (постоянная составляющая)

    #     # # 4. Обратное FFT → v(t)
    #     # v = np.fft.ifft(fft_v).real  # Отбрасываем мнимую часть (погрешности вычислений)
    #     # # v = v - np.mean(v)
    #     # fVibEval = k*(simpson(v[t1:t2], x=ta[t1:t2])-2*simpson(v[t3:t4], x=ta[t3:t4])+simpson(v[t5:t6], x=ta[t5:t6]))
    #     # chirp_copy[i] = chirp_copy[i]-fVibEval/2*np.pi/T**2

    # correct data
    intvib = acc_mx[:, StI:StI+iTAI+1]*fat
    fvib = k*simpson(y=intvib, x=tan, axis=-1)*TFF
    chirp_rate = chirp_rate - fvib/T**2/(2*np.pi) # + or -, 2pi?

    #plt.plot(chirp_rate, intensity, color="green")
    plt.scatter(chirp_rate, intensity, color="green", label="corrected", s=10)



    # evaluate vibration influence block3
    
    # From ga

    initial_guess = [(np.max(intensity_clear) - np.min(intensity_clear))/2, w, ph, np.min(intensity_clear)] 
    par, cov = curve_fit(sins, chirp_rate, intensity_clear, p0=initial_guess)
    dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
    dgA = 1/k/T**2/(A/dA*sk)
    dgA = abs(dgA)
    #dgA = 1/k/T**2/(A/dA)**2
    #noise = intensity - (A*np.sin(w*chirp_rate+ph)+s)
    #SNR = np.mean(intensity**2)/np.mean(noise**2)
    #dgA = 1/k/T**2/SNR
    print("vibration sensetivity influence ga =", dgA*np.sqrt(TF*n)*r, 'mGal/.')

    # from v

    # initial_guess = [(np.max(intensity_clear) - np.min(intensity_clear))/2, w, ph, np.min(intensity_clear)] 
    # par, cov = curve_fit(sins, chirp_copy, intensity, p0=initial_guess)
    # dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
    # dgV = 1/k/T**2/(A/dA)
    # print("vibration sensetivity influence v =", dgV*1e5*np.sqrt(TF*n), 'mGal/.')


    # fit corrected data
    initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, np.min(intensity)] 
    par, cov = curve_fit(sins, chirp_rate, intensity, p0=initial_guess)
    A, w, ph, s = par
    dw, dph, dA = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0])
    dgC = 1/k/T**2/(A/dA*sk)
    dgC = abs(dgC)
    #dgC = 1/k/T**2/(A/dA)**2
    noise = intensity - (A*np.sin(w*chirp_rate+ph)+s)
    #SNR = np.mean(intensity**2)/np.mean(noise**2)
    #dgC = 1/k/T**2/SNR

    print("sensetivity for corrected data =", dgC*np.sqrt(TF*n)*r, 'mGal/.')
    chirp_rate = np.sort(chirp_rate)
    plt.plot(chirp_rate, A*np.sin(w*chirp_rate+ph) + s, color="green")

    # # sensetivity for averaged g afer correction
    # dgMX = []
    # g0 = 9.955
    # for j in range(len(intensity)//n):
    #     initial_guess = [(np.max(intensity[j*n:j*n+n]) - np.min(intensity[j*n:(j+1)*n]))/2, 2*np.pi*T*T, 0, np.min(intensity[j*n:(j+1)*n])] 
    #     par, cov = curve_fit(sins, chirp_rate[j*n:n*(j+1)], intensity[j*n:n*(j+1)], p0=initial_guess)
    #     A, wa, ph, s = par

    #     if j == 0:
    #         m = round((w*k*g0/2/np.pi+ph-np.pi/2)/np.pi)
    #         gj = (np.pi/2+np.pi*m-ph)/w*2*np.pi/k
    #         dgMX.append(gj)
    #     else:
    #         np_buff = np.array(gj)
    #         mean_buff = np.mean(np_buff)
    #         m = round((w*k*g0/2/np.pi+ph-np.pi/2)/np.pi)
    #         gj0 = (np.pi/2+np.pi*m-ph)/w*2*np.pi/k
    #         gj1 = (np.pi/2+np.pi*(m+1)-ph)/w*2*np.pi/k
    #         gj_1 = (np.pi/2+np.pi*(m-1)-ph)/w*2*np.pi/k
    #         if abs(gj1-mean_buff) < abs(gj0-mean_buff):
    #             m += 1
    #             gj0 = gj1
    #             gj1 = (np.pi/2+np.pi*(m+1)-ph)/w*2*np.pi/k
    #             while abs(gj1-mean_buff) < abs(gj0-mean_buff):
    #                 m += 1
    #                 gj0 = gj1
    #                 gj1 = (np.pi/2+np.pi*(m+1)-pha)/w*2*np.pi/k
    #         elif abs(gj_1-mean_buff) < abs(gj0-mean_buff):
    #             m -= 1
    #             gj0 = gj_1
    #             gj_1 = (np.pi/2+np.pi*(m-1)-ph)/w*2*np.pi/k
    #             while abs(gj_1-mean_buff) < abs(gj0-mean_buff):
    #                 m -= 1
    #                 gj0 = gj_1
    #                 gj_1 = (np.pi/2+np.pi*(m-1)-ph)/w*2*np.pi/k
    #         dgMX.append(gj0)    
    # dgMX = np.array(dgMX)
    # print("list of g after correction", dgMX)
    # print("sensetivity for averaged g after correction =",np.std(dgMX)/np.sqrt(len(dgMX))*np.sqrt(TF*n)*r, 'mGal/.')


    print("correction efficiency ga",(dgE-dgC)/dgA*100, "%")
    # print("correction efficiency V",(dgE-dgC)/dgV*100, "%")
    plt.legend()

    plt.figure(2)
    plt.scatter(abs(noise), abs(fvib))
    plt.xlabel("noise")
    plt.ylabel("fvib")

    plt.figure(3)
    plt.scatter(chirp_rate, abs(noise))
    plt.xlabel("chirp rate")
    plt.ylabel("noise")

    #plt.legend()
    plt.show()

def phaseCheck(i, StI, TFF):

    # fit start data
    initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, np.min(intensity)] 
    par, cov = curve_fit(sins, chirp_rate[:n], intensity[:n], p0=initial_guess)
    A, w, ph, s = par


    chirp_copy = chirp_rate.copy()  # Используем copy(), чтобы не изменять оригинал
    #plt.ion()  # Включаем интерактивный режим
    
    # correct data
    intvib = acc_mx[:, StI:StI+iTAI+1]*fat
    fvib = k*simpson(y=intvib, x=tan, axis=-1)*TFF
    chirp_copy = chirp_copy - fvib/T**2/(2*np.pi) # + or -, 2pi?
    
        
    shft = chirp_rate[0]
    # Создаём новую фигуру (новое окно) на каждой итерации
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 3 субплота в одном окне (1 ряд, 3 столбца)
    fig.suptitle(f'Iteration {i}')  # Заголовок для окна
        
 
    


    # Первый график: current_data vs ta
    axes[0].plot(ta*1e3, acc_mx[i])
    axes[0].set_title(f'acc data, correction = {fvib[i]/T**2/(2*np.pi)} Hz')
    axes[0].set_xlabel('ta')
    axes[0].set_ylabel('current_data')
        
        
    # Второй график: chirp_rate vs intensity (до i+1)
    axes[1].plot(chirp_rate[:100]-shft, intensity[:100], color='blue')
    axes[1].scatter(chirp_rate[:100]-shft, intensity[:100], color='blue')
    axes[1].plot(chirp_rate[:100]-shft, A*np.sin(w*chirp_rate[:100]+ph) + s, color='blue')
    axes[1].scatter(chirp_copy[i]-shft, intensity[i], color='red')
    axes[1].set_title('Original Chirp')
    axes[1].set_xlabel('chirp_rate')
    axes[1].set_ylabel('intensity')
    
        
    # # Третий график: chirp_copy vs intensity (скорректированный)
    # axes[2].plot(chirp_copy[:i+1]-shft, intensity[:i+1], color='red')
    # axes[2].scatter(chirp_copy[:i+1]-shft, intensity[:i+1], color='red')
    # axes[2].set_title('Corrected Chirp')
    # axes[2].set_xlabel('chirp_copy')
    # axes[2].set_ylabel('intensity')
        
    # # Обновляем окно
    # plt.tight_layout()  # Улучшаем layout
    # plt.pause(0.1)  # Короткая пауза для обновления графика
        
    # Остановка для проверки
    # stop_input = input("Press Enter to continue or 'q' to quit: ")
    # if stop_input.lower() == 'q':
    #     break
    
    #plt.ioff()  # Выключаем интерактивный режим в конце
    plt.show()  # Показываем финальное состояние, если нужно

def sensCount():
    global chirp_rate, intensity, TF, n

    chirp0 = chirp_rate[:n]
    intensity0 = aver(intensity)


    # sensetivity for averaged g
    dgMX = []
    g0 = 9.955
    for j in range(len(intensity)//n):
        initial_guess = [(np.max(intensity[j*n:j*n+n]) - np.min(intensity[j*n:(j+1)*n]))/2, 2*np.pi*T*T, 0, np.min(intensity[j*n:(j+1)*n])] 
        par, cov = curve_fit(sins, chirp0, intensity[j*n:n*(j+1)], p0=initial_guess)
        Aa, wa, pha, sa = par
        plt.plot(chirp0*2*np.pi/k, Aa*np.sin(wa*chirp0+pha)+sa)
        if j == 0:
            m = round((wa*k*g0/2/np.pi+pha-np.pi/2)/np.pi)
            gj = (np.pi/2+np.pi*m-pha)/wa*2*np.pi/k
            dgMX.append(gj)
        else:
            np_buff = np.array(gj)
            mean_buff = np.mean(np_buff)
            m = round((wa*k*g0/2/np.pi+pha-np.pi/2)/np.pi)
            gj0 = (np.pi/2+np.pi*m-pha)/wa*2*np.pi/k
            gj1 = (np.pi/2+np.pi*(m+1)-pha)/wa*2*np.pi/k
            gj_1 = (np.pi/2+np.pi*(m-1)-pha)/wa*2*np.pi/k
            if abs(gj1-mean_buff) < abs(gj0-mean_buff):
                m += 1
                gj0 = gj1
                gj1 = (np.pi/2+np.pi*(m+1)-pha)/wa*2*np.pi/k
                while abs(gj1-mean_buff) < abs(gj0-mean_buff):
                    m += 1
                    gj0 = gj1
                    gj1 = (np.pi/2+np.pi*(m+1)-pha)/wa*2*np.pi/k
            elif abs(gj_1-mean_buff) < abs(gj0-mean_buff):
                m -= 1
                gj0 = gj_1
                gj_1 = (np.pi/2+np.pi*(m-1)-pha)/wa*2*np.pi/k
                while abs(gj_1-mean_buff) < abs(gj0-mean_buff):
                    m -= 1
                    gj0 = gj_1
                    gj_1 = (np.pi/2+np.pi*(m-1)-pha)/wa*2*np.pi/k
            dgMX.append(gj0)    
    dgMX = np.array(dgMX)
    print("list of g", dgMX)
    print("sensetivity for averaged g =",np.std(dgMX)/np.sqrt(len(dgMX))*np.sqrt(TF*n)*r, 'mGal/.')
    plt.show()

# # accSensFunc_var1
# def fa(t):
#     if 0 <= t <= ty:
#         return 2*(1-np.cos(OR*t/2))/OR
#     elif ty <= t <= ty+T:
#         return t - ty + 2/OR
#     elif ty+T <= t <= 3*ty+T:
#         return T + 2*(1-np.cos(OR/2*(t-T)))/OR    
#     elif 3*ty+T <= t <= 3*ty+2*T:
#         return 2*T +3*ty - t + 2/OR     
#     elif 3*ty+2*T <= t <= 4*ty+2*T:
#         return 2*(1-np.cos(OR/2*(t-2*T)))/OR
#     else:
#         return 0   


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

# # accSensFunc_var3
# def fa(t):
#     if 0 < t <= T+2*ty:
#         return t/(T+2*ty)**2
#     elif T+2*ty < t <= 2*T+4*ty:
#         return (2*(T+2*ty)-t)/(T+2*ty)**2
#     else:
#         return 0
    
vfunc = np.vectorize(fa)


c = 3*1e8
k =  (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9)/c + (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9 - 6.83468261090429*1e9)/c
k = k*2*np.pi
#print(k)
# start_freq = 90582400/70*5282
# dt = 30e-3 # s для чирпирования
n = 101 # количество точек
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
Ampl = 50
dt = TRP/16383 # Red Pitaya time step
iTAI = int(np.floor(TAI/dt))
ta = np.arange(0, 16384)*dt
tan = ta[:iTAI+1]



fat = vfunc(tan)

r = 100000 # коэф единиц измерения
sk = 1 # коэф поправки для оценки погрешности

# чтение csv P(a)
file_path = r'gravity_measure_vib/testdata/37290925191200/interference_signal.csv' 
data = np.genfromtxt(file_path, delimiter=',', dtype=None, skip_header=1)
data = np.array(data.tolist())

chirp_rate = data[:,0]
intensity = data[:,1]

# acc data read
acc_mx = csv_np('gravity_measure_vib/testdata/37290925191200')/150/Ampl
acc_mx = acc_mx - np.mean(acc_mx)
#acc_mx = acc_mx[:-1]

StI = 0
TFF =  1.81
#singleWork(StI, TFF)

delay = 150
a = -10.
b = 10.
optimalFind(delay+1, a, b)
print(delay, a, b)

# i = 3
# phaseCheck(i, StI, TFF)

#sensCount()