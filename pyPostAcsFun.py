'''
Acoustic Post Processing Functions
By Daniel Weitsman
1/23/21
'''

import os
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import h5py
import re

#%%
fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%

def apply_fun_to_h5(dir, function, append_perf = False):

    '''
    This function finds all the acs_data.h5 files contained in a directory and all of its subdirectories and applies
    the specified functions to the data set contained in each h5 file.
    :param dir: The parent directory within which to search for the acs_data.h5 files
    to search for the h5
    :param function: A list of functions which to run on each of the detected acs_data.h5 files.
    :param append_perf: Boolean (true/false) if you want to write the load contained in full.txt to the h5 file.

    :return:
    '''

    #   Checks to ensure that functions are provided in a list
    assert isinstance(function, list), 'Functions must be inputed to this function in a list'

    #   Loops through each subfolder in a directory until arriving at the base folder
    for item in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, item)):
            apply_fun_to_h5(os.path.join(dir, item), function, append_perf)

    #   Checks whether the acs_data.h5 exists in current directory
    if os.path.exists(os.path.join(dir, 'acs_data.h5')):
        print('h5 file exists in' + dir)
    #   Makes a new 'Figures' folder in the current directory
        if os.path.exists(os.path.join(dir, 'Figures')) is False:
            os.mkdir(os.path.join(dir, 'Figures'))
    #   Opens the acs_data.h5 file and executes all the specified functions
        with h5py.File(os.path.join(dir, 'acs_data.h5'),'r+') as dat_file:
            if append_perf is True:
                append_perf_dat(dat_file, dir)
            for f in function:
                f(dat_file,os.path.join(dir, 'Figures'))

    else:
        print('h5 file does not exist in ' + dir)
        # uncomment the following lines if a case was run without acquiring acoustic data (acs_data.h5 does not exist in folder)
        # with h5py.File(os.path.join(dir, 'acs_data.h5'),'a') as dat_file:
        #     if append_perf is True:
        #         append_perf_dat(dat_file,dir)



def append_perf_dat(dat_file, dir,col = 27):
    '''
    This function appends the performance and all the other data contained in Full.txt, which is exported by default
    from the LabVIEW data acquisition vi (UAV Control V4.vi) to the acs_data.h5 file.
    :param dat_file: opened acs_dat.h5 file to which to append the performance data
    :param b: prefix to the path containing the Full.txt file
    :param col: number of columns in the Full.txt file
    :return:
    '''

    #   opens and reads contents of Full.txt file. Any slashes ('/') are replaced with underscores since dataset names cannot contain '/'.
    with open(os.path.join(dir, 'Full.txt'), 'r') as f_txt:
        data = f_txt.read().replace('/','_')
    #   splits the Full.txt file with \t and \n as delimiters
    data_split = re.split('\t|\n', data)
    #   reshapes a single-dimensional list into the right dimensional array
    data_split = np.reshape(data_split[:-1], (int(len(data_split[:-1]) / col), col))
    #   extracts header of each data set in Full.txt
    header = data_split[0]
    #   changes numerical data from type str to float64
    data_split = data_split[1:].astype(float)
    #   loops through each dataset contained in Full.txt
    for i, dat in enumerate(data_split.transpose()):
        #   writes data to a new dataset titled with each header
        dat_file.create_dataset(header[i], data=dat, shape=np.shape(dat))

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

def tseries(xn, fs,t_lim = [0,1], levels = [-0.5,0.5],save_fig = True, save_path = ''):
    '''
    This function generates a figure of the pressure time history
    :param xn: time series [Pa]
    :param fs: sampling frequency [Hz]
    :param t_lim: extents of time axis, supplied as a list [s]
    :param levels: limits of vertical axis, supplied as a list [Pa]
    :param save_fig: set to true to save the figure
    :param save_path: path where to save the figure
    :param plot: set to true in order to generate the time series plot
    :return:
    '''

    t = np.arange(len(xn))*fs**-1
    for i in range(np.shape(xn)[1]):
        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(bottom=0.15)
        ax.plot(t[int(fs*t_lim[0]):int(fs*t_lim[1])], xn[int(fs*t_lim[0]):int(fs*t_lim[1]),i])
        ax.axis([t_lim[0],t_lim[1],levels[0],levels[1]])
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('$x_n [Pa]$')
        ax.grid()
        ax.set_title('Mic: '+str(i+1))

        if save_fig is True:
            plt.savefig(os.path.join(save_path,'tseries_'+str(i+1)+'.png'),format='png')
            plt.close()


def msPSD(xn, fs, df = 5, win = True, ovr = 0, f_lim =[10,5e3], levels = [0,100],save_fig = True, save_path = '',plot = True):
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
    if win is True:
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

    if plot is True:
        for i in range(np.shape(Gxx_avg)[1]):
            fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
            plt.subplots_adjust(bottom=0.15)
            ax.plot(f, 10 * np.log10(Gxx_avg[:,i]*df / 20e-6 ** 2))
            ax.set_xscale('log')
            ax.axis([f_lim[0],f_lim[1],levels[0],levels[1]])
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
            ax.grid()
            ax.set_title('Mic: '+str(i+1))

            if save_fig is True:
                plt.savefig(os.path.join(save_path,'spectra_m'+str(i+1)+'.png'),format='png')
                plt.close()

    return f,Gxx,Gxx_avg


def spectrogram(xn, fs, df, win = True, ovr= 0,t_lim = '', f_lim = [0,10e3],levels = [0,100],save_fig = True, save_path = '' ,plot = True):
    '''
    This function computes the spectrogram of a given time series
    :param xn: time array
    :param fs: sampling frequency [Hz]
    :param df: frequency resolution [hz]
    :param win: window function (T/F) (only the Hanning window is supported at this time)
    :param ovr: percentage of overlap between subsequent records
    :return:
    :param t: resultant time vector corresponding to the midpoint of each record [s]
    :param f: frequency vector [Hz]
    :param Gxx: resultant matrix of single-sided spectral densities [N x Nfft] [V^2/Hz]
    '''

    N = (fs**-1*df)**-1
    t = np.arange(len(xn)) * fs**-1
    f ,Gxx,Gxx_avg = msPSD(xn,fs,df =df,ovr = ovr,save_fig=0,win = win,plot = False)
    t = t[int(N / 2):-int(N / 2)][::int((1 - ovr) * N)]

    if plot is True:
        levels = np.arange(levels[0], levels[1], 5)
        for i in range(np.shape(Gxx)[2]):
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
            plt.subplots_adjust(bottom=0.15)
            spec = ax.contourf(t, f, 10 * np.log10(np.squeeze(Gxx[:,:-1,i])*df / 20e-6 ** 2), cmap='hot', levels=levels)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (sec)')
            if t_lim is list:
                ax.set_xlim([t_lim[0], t_lim[-1]])
            else:
                ax.set_xlim([t[0], t[-1]])
            ax.set_ylim(f_lim[0], f_lim[1])
            ax.set_title('Mic: '+str(i+1))
            cbar = fig.colorbar(spec)
            cbar.set_label('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

            if save_fig is True:
                plt.savefig(os.path.join(save_path, 'spectrogram_m' + str(i + 1) + '.png'),format='png')
                plt.close()

    return t, f, np.transpose(Gxx)