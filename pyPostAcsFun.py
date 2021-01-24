'''
Acoustic Post Processing Functions
By Daniel Weitsman
1/23/21
'''

import os
from shutil import rmtree
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
#%%
fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%

def apply_fun_to_folder(dir):
    '''
    This function applies a given function to all folders containing a acs_data.h5 file
    :param dir:
    :return:
    '''
    for item in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, item)):
            apply_fun_to_folder(os.path.join(dir, item))
    if os.path.exists(os.path.join(dir, 'acs_data.h5')):
        print('h5 file exists in' + dir)
        if os.path.exists(os.path.join(dir, 'Figures')) is False:
            os.mkdir(os.path.join(dir, 'Figures'))
    else:
        print('h5 file does not exist in' + dir)


def msPSD(xn, fs, df = 5, win = '', ovr = 0):
    '''
    This function computes the single and double sided mean-square averaged PSD for a given time series
    :param xn: time series
    :param fs: sampling frequency [Hz]
    :param N: number of points per averaging segment
    :param win: applies Hanning window if set equal to 1
    :param ovr: percentage of overlap between adjacent bins
    :return:
    :param f: frequency vector
    :param Gxx_avg: mean-square averaged single-sided PSD
    '''
    if len(np.shape(xn)) ==1:
        xn = np.expand_dims(xn,axis = 1)
    #   points per record
    N = int((df*fs**-1)**-1)
    #   constructs frequency array
    f = np.arange(int(N/2)) * df
    #   returns rms normalized Hanning window if no window is applied an array of ones is returned
    if win == 'hann':
        W = hann(N, fs)
    else:
        W = np.ones(N)
    if ovr == 0:
        #   number of records
        Nfft = int(np.floor(len(xn) / N))
        #   ensures that the length of the time series can accommodate the desired number of averages
        assert (N * Nfft <= len(xn), 'Desired number of averages exceeds the length of the time series')
        #   reshapes time series into a matrix of dimensions Nfft x N
        xn = np.reshape(xn[:int(N * Nfft),:], (int(N), Nfft,np.shape(xn)[1]), order='F')
        #   applies window and computes linear spectrum
        Xm = (fft(xn.transpose()) * fs ** -1).transpose()
    else:
        #   number of records
        Nfft = int(np.floor((len(xn) - N) / ((1 - ovr) * N)))
        #   initiates matrix to store linear spectrum
        Xm = np.zeros((N,Nfft + 1,np.shape(xn)[1]), complex)
        #   computes linear spectrum for each record and populates matrix
        for i in range(Nfft + 1):
            Xm[:,i,:] = (fft(xn[int(i * (1 - ovr) * N):int(i * (1 - ovr) * N + N),:].transpose() * W) * fs**-1).transpose()
    #   computes double-sided PSD
    Sxx = (fs**-1*N)**-1 * abs(Xm) ** 2
    #   computes single-sided PSD
    Gxx = Sxx[:int(N / 2),:,:]
    Gxx[1:-1,:,:] = 2 * Gxx[1:-1,:,:]
    #   averages the single-sided PSD of all segments
    Gxx_avg = 1 / Nfft * np.sum(Gxx, axis=1)
    return f, Gxx_avg

def hann(N, fs):
    '''
    This function returns the rms normalized hanning window consisting of N points. This normalization ensures that after the window function is applied, when the spectral density is integrated it would still yield the mean square of the time series.
    :param N: Number of points
    :param fs: sampling frequency [Hz]
    :return:
    :param W: rms normalized window function
    '''
    dt = 1 / fs
    hann = 1 - np.cos(2 * np.pi * np.arange(N) * dt / (N * dt))
    W = hann / np.sqrt(1 / N * np.sum(hann ** 2))
    return W

def spectrogram(xn, fs, N, win, ovr):
    '''
    This function computes the spectrogram of a given time series
    :param xn: time array
    :param fs: sampling frequency [S/sec]
    :param N: number of points in each record
    :param win: window function, defaults to none (only the hann window is supported at this time)
    :param ovr: percentage of overlap between subsequent records
    :return:
    :param tspec: resultant time vector corresponding to the midpoint of each record [s]
    :param f: frequency vector [Hz]
    :param Gxx: resultant matrix of single-sided spectral densities [N x Nfft] [WU^2/Hz]
    '''

    #   temporal resolution
    dt = 1 / fs
    #   number of records
    Nfft = int(np.floor((len(xn) - N) / ((1 - ovr) * N)))
    #   frequency resolution
    df = 1 / (N * dt)
    #   creates frequency array
    f = np.arange(0, N / 2) * df
    #   crates time array
    t = np.arange(0, len(xn)) * dt
    #   initiates matrix to store linear spectrum
    Xm = np.zeros((Nfft + 1, N), complex)
    #   returns rms normalized hanning window
    if win == 'hann':
        W = hann(N, fs)
    #   computes linear spectrum for each record and populates matrix
    for i in range(0, Nfft + 1):
        if win == 'hann':
            Xm[i] = fft(xn[int(i * (1 - ovr) * N):int(i * (1 - ovr) * N + N)] * W) * dt
        else:
            Xm[i] = fft(xn[int(i * (1 - ovr) * N):int(i * (1 - ovr) * N + N)]) * dt
    #   computes the double sided spectral density
    Sxx = 1 / (N * dt) * abs(Xm) ** 2
    #   extracts single sided spectral density
    Gxx = Sxx[:, :int(N / 2)]
    Gxx[:, 1:-1] = 2 * Gxx[:, 1:-1]
    #   extracts the time corresponding to the midpoint of each record
    tspec = t[int(N / 2):-int(N / 2)][::int((1 - ovr) * N)]
    return tspec, f, np.transpose(Gxx)