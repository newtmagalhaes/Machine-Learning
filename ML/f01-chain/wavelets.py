# -*- coding: utf-8 -*-

# %% [markdown]
# # Wavelets
#
# ## Importando bibliotecas

# %%
import os
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.robust.scale as robust
from scipy import signal
from scipy import stats as sc

# %%


def getListOfFiles(dirName):
    '''
    create a list of file and sub directories names in the given directory
    '''
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

# %%


def EnergyPercent(Ac, Dc, Original):

    E = np.empty((1, np.size(Dc, 1)+1), float)
    Ep = np.zeros((1, np.size(Dc, 1)+1), float)

    Et = np.sum(np.square(Original))

    for i in range(0, np.size(Dc, 1)):
        E[0, i] = np.sum(np.square(Dc[0, i]))

    E[0, i+1] = np.sum(np.square(Ac))

    Ep = E/Et

    return Ep


# %%
# Ler amostra do arquivo
path = 'C:\Renomeado2'  # Confgurar caminho para pasta contendo as amostras

Files = getListOfFiles(path)

amostra = np.empty((100000, 4), float)

N_decomp_levels = 5  # Configurar níveis de decomposição aqui

n = N_decomp_levels+2

nCols = (n-1)*6+3

DataBase = np.zeros((len(Files), nCols), float)

# Configurar quantas amostras devem ser "saltadas" para subamostragem
x_array = [1]

for x_idx in range(0, len(x_array)):

    for i in range(0, len(Files)):
        fp = open(Files[i])

        content = fp.readlines()
        x = np.array(content[21:])

        for j in range(0, 100000):
            amostra_strings = x[j].split(',')
            amostra[j] = np.array(amostra_strings)

        #  Remover a media
        mean = np.mean(amostra[:, 1:5], 0)
        amostra[:, 1:5] = amostra[:, 1:5]-mean

        x = x_array[x_idx]
        # Filtragem digital
        fs = 10000
        fc = fs/(2*x)-300  # Cut-off frequency of the filter
        w = fc / (fs / 2)  # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        output = signal.filtfilt(b, a, amostra[:, 3])
        # output = amostra[:,3]

        # Subamostragem
        t = 10  # Configurar tempo total em segundos da amostra cortada

        n_pontos = int(t*10000)

        fluxo = output[0:n_pontos:x]

        input = fluxo

        Desvio = np.zeros((1, n-1), float)
        MeanAD = np.zeros((1, n-1), float)
        MedianAD = np.zeros((1, n-1), float)
        Energia = np.zeros((1, n-1), float)
        Kurtosis = np.zeros((1, n-1), float)
        Skewness = np.zeros((1, n-1), float)
        cD = np.empty((1, n-2), object)

        for m in range(0, n-2):
            cA, cD[0, m] = pywt.dwt(input, 'db2')
            Desvio[0, m] = np.std(cD[0, m])
            MeanAD[0, m] = pd.DataFrame(cD[0, m]).mad()
            MedianAD[0, m] = robust.mad(cD[0, m])
            Kurtosis[0, m] = sc.kurtosis(cD[0, m])
            Skewness[0, m] = sc.skew(cD[0, m])
            input = cA

        Desvio[0, m+1] = np.std(cA)
        MeanAD[0, m+1] = pd.DataFrame(cA).mad()
        MedianAD[0, m+1] = robust.mad(cA)
        Kurtosis[0, m+1] = sc.kurtosis(cA)
        Skewness[0, m+1] = sc.skew(cA)

        Energia = EnergyPercent(cA, cD, fluxo)

        # sObject=slice(31,33);

        # Freq=int(Files[i][sObject]);
        Freq = 0

        # sObject=slice(34,37);

        # Load=int(Files[i][sObject]);
        Load = 0

        # sObject=slice(23,24);

        # Classe=int(Files[i][sObject]);
        Classe = 7

        # sObject=slice(26,30);

        # Index=int(Files[i][sObject]);

        Pattern = np.zeros((1, nCols-3), float)
        np.concatenate(
            (Desvio, MeanAD, MedianAD, Kurtosis, Skewness, Energia),
            out=Pattern,
            axis=1
        )

        DataBase[i, 0:nCols-3] = Pattern

        DataBase[i, nCols-3:] = [Freq, Classe, Load]

        fp.close()
print(i)

# %%
# FFT=np.fft.fft(fluxo);
#
# FFT_abs=abs(FFT)[0:50000]
#
# f1=np.linspace(0,5000,50000);
#
# FFT_a2=np.fft.fft(cA)
#
# FFT_abs_a2=abs(FFT_a2)[0:12500];
#
# f2=np.linspace(0,1250,12500);
#
#
# FFT_d2=np.fft.fft(cD[0,1])
#
# FFT_abs_d2=abs(FFT_d2)[0:12500];
#
# f3=np.linspace(1251,2500,12500);
#
# FFT_d1=np.fft.fft(cD[0,0])
#
# FFT_abs_d1=abs(FFT_d1)[0:25000];
#
# f4=np.linspace(2500,5000,25000);
#
#
#
#
# plt.figure()
# plt.plot(f1,FFT_abs)
#
# plt.figure()
# plt.plot(f2,FFT_abs_a2)
#
# plt.figure()
# plt.plot(f3,FFT_abs_d2)
#
# plt.figure()
# plt.plot(f4,FFT_abs_d1)
#
# new_freq=0.001*fs/x
#np.savetxt('Estudo Redução da frequencia 3//'+str(new_freq)+"kHz-10s-db2-5niveis.csv", DataBase, delimiter=",")

#
#
# fig=plt.figure();
#    #plt.plot(fluxo)
#ax = fig.add_subplot(n, 1, 1)
# ax.plot(fluxo)
#ax.set_title('Original', fontsize=10)
#ax = fig.add_subplot(n, 1, 2)
# ax.plot(cD[0,0])
#ax.set_title('Detalhe 1', fontsize=10)
#ax = fig.add_subplot(n, 1, 3)
# ax.plot(cD[0,1])
#ax.set_title('Detalhe 2', fontsize=10)
#ax = fig.add_subplot(n, 1, 4)
#    ax.plot(cD3)
#    ax.set_title('Detalhe 3', fontsize=10)
#    ax = fig.add_subplot(n, 1, 5)
#    ax.plot(cD4)
#    ax.set_title('Detalhe 4', fontsize=10)
#    ax = fig.add_subplot(n, 1, 6)
#    ax.plot(cD5)
#    ax.set_title('Detalhe 5', fontsize=10)
#    ax = fig.add_subplot(n, 1, 7)
#    ax.plot(cD6)
#    ax.set_title('Detalhe 6', fontsize=10)
#    ax = fig.add_subplot(n, 1, 8)
# ax.plot(cA)
#ax.set_title('Aproximação 2', fontsize=10)
#
#    plt.show()
