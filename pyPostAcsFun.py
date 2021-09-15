'''
Acoustic Post Processing Functions
By Daniel Weitsman
1/23/21
'''

import os
import numpy as np
from scipy.fft import fft,ifft
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import h5py
import re
from bisect import bisect

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
    if np.shape(xn) == (len(xn),1):
        xn = np.expand_dims(xn,axis = 1)

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


def PSD(xn, fs):
    '''
    This function computes the single and double sided PSD from a given time series
    :param xn: time series
    :param fs: sampling frequency [Hz]
    :return:
    :param f: frequency vector [Hz]
    :param Sxx: double-sided spectral density [WU^2/Hz]
    :param Gxx: single-sided spectral densities [WU^2/Hz]
    '''

    dt = fs**-1
    df = (len(xn)*dt)**-1

    #   Frequency vector
    f = np.arange(int(len(xn) /2)) * df
    #   Computes the linear spectrum
    Xm = fft(xn,axis = 0) * dt
    #   Computes double-sided PSD
    Sxx = 1/(len(xn)*dt) * np.conj(Xm) * Xm
    #   Computes single-sided PSD
    Gxx = Sxx[:int(len(xn) / 2)]
    Gxx[1:-1] = 2 * Gxx[1:-1]

    return f, Xm, Sxx, Gxx

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

    if len(np.shape(xn)) == 1:
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

    return f,Xm,Sxx,Gxx,Gxx_avg


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
        # levels = np.arange(levels[0], levels[1], 2)
        for i in range(np.shape(Gxx)[2]):
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
            plt.subplots_adjust(bottom=0.15)
            spec = ax.contourf(t, f, 10 * np.log10(np.squeeze(Gxx[:,:-1,i])*df / 20e-6 ** 2), cmap='hot', levels=levels)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (sec)')


            if isinstance(t_lim, list):
                ax.set_xlim([t_lim[0], t_lim[-1]])
            else:
                ax.set_xlim([t[0], t[-1]])

            ax.set_ylim(f_lim[0], f_lim[1])
            # ax.set_title('Mic: '+str(i+1))
            cbar = fig.colorbar(spec)
            cbar.set_ticks(np.arange(levels[0],levels[-1]+1,5))
            cbar.set_ticklabels(np.arange(levels[0],levels[-1]+1,5))

            cbar.set_label('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')


            if save_fig is True:
                plt.savefig(os.path.join(save_path, 'spectrogram_m' + str(i + 1) + '.png'),format='png')
                plt.close()

    return t, f, np.transpose(Gxx)

def filt_response(bb,aa,fs,N,plot=True):
    '''
    This function returns the frequency response of a moving average filter by computing the linear spectrum of the impulse response.
    :param bb: output (numerator) coefficients of the frequency response, multiplied by dt
    :param aa: input (denominator) coefficients of the frequency response
    :param fs: sampling frequency [Hz]
    :param N: length of the impulse time series [points]
    :return:
    :param f: frequency vector [Hz]
    :param y: impulse time series
    :param h: frequency response
    :param phase: phase [deg]

    '''
    impulse = np.zeros(int(N))
    impulse[0] = fs
    y = lfilter(bb, aa, impulse)
    h = (fft(y)*fs**-1)[:int(N/2)]
    phase = np.angle(h) * 180 / np.pi
    f = np.arange(N/2)*(N*fs**-1)**-1

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.5))
        ax[0].plot(f, abs(h))
        ax[0].set_ylabel('Magnitude')
        ax[0].tick_params(axis='x', labelsize=0)
        ax[0].grid()
        ax[0].set_xscale('log')
        # ax[0].set_xlim(10, 5e3)

        ax[1].plot(f, phase)
        ax[1].set_ylabel('Phase [$\circ$]')
        ax[1].set_xlabel('Frequency [Hz]')
        ax[1].grid()
        ax[1].set_xscale('log')
        # ax[1].set_xlim(10, 5e3)

    return f,y,h,phase

def xCorr(xn, yn, xfs, yfs):
    '''
    This function computes the circular cross correlation between two time series in the frequency domain. If a
    simple cross correlation is required, zero-pad both time series doubling their lengths.
    :param xn: first time series
    :param yn: second time series
    :param xfs: sampling rate of first time series
    :param yfs: sampling rate of second time series
    :return:
    :param Rxy: cross correlation (simple if the time series are zero padded)
    '''

    # assert len(xn) * xfs ** -1 == len(yn) * yfs ** -1, 'Ensure that the record lengths of both time series are equal'
    Xm = fft(xn)*xfs**-1
    Ym = fft(yn)*yfs**-1
    Sxy = 1 / (len(xn) * xfs**-1) * np.conj(Xm) * Ym
    Rxy = ifft(Sxy) * xfs
    return Rxy


def hilbert(xn,fs):
    '''
    This function applies the hilbert transform to a given time series and returns the envelope and instantaneous phase and frequency.
    :param xn: time series
    :param fs: sampling frequency [Hz]
    :return:
    :param envelope: envelope of the time series
    :param phi: instantaneous phase [degrees]
    :param f: instantaneous frequency

    '''

    #   temporal resolution
    dt = fs**-1
    #   computes linear spectrum
    Xm = fft(xn) * dt

    #   generates window of the Hilbert transform
    W_hilbert = np.concatenate((np.ones(int(len(xn) / 2)) * 2, np.zeros(int(len(xn) / 2))), axis=0)
    W_hilbert[0] = 1
    W_hilbert[int(len(xn) / 2)] = 1

    # computes the inverse fft of the product between the linear spectrum and the Hilbert transform window (
    # equivalent to a time shift in the time domain)
    zh = ifft(Xm * W_hilbert) * dt ** -1

    envelope = abs(zh)
    phi = np.unwrap(np.angle(zh))
    f = 1 / (2 * np.pi) * np.diff(phi) * 1 / dt

    return f,phi,envelope

def rpm_eval(ttl,fs,start_t,end_t):
    '''
    This function evaluates the average rotational rate for a segment of data based on the tac pulse signal.
    :param ttl: raw tac time series
    :param fs: sampling rate of the tac signal [Hz]
    :param start_t: start time of the segment under consideration [s]
    :param end_t: end time of the segment under consideration [s]
    :return:
    '''

    # identifies the indices corresponding to the leading edge of each pulse
    LE_ind = np.squeeze(np.where(np.diff(ttl) == 1))
    # returns a list containing the starting and ending index
    lim_ind = list(map(lambda x: bisect(LE_ind, x),[start_t * fs,end_t * fs]))
    # computes the rotational rate over the time interval
    rpm = (np.diff(LE_ind[lim_ind[0]:lim_ind[1]]) / fs / 60) ** -1
    # averages the rpm
    rpm_nom = np.mean(rpm)
    # confidence interval of the the rpm (95% level of certainty)
    u_rpm = 1.94*np.std(rpm)/np.sqrt(len(LE_ind)-1)

    return LE_ind, lim_ind, rpm_nom, u_rpm

def upsample(xn, fs, N):
    '''
    This function upsamples a time series that was sampled at a sampling rate of fs to a length of N points.
    :param xn: times series
    :param fs: sampling rate [Hz]
    :param N: number of points to upsample to
    :return:
    '''

    T = len(xn)*fs**-1
    fs1 = N/T
    Xm = (fft(xn.transpose())*fs**-1).transpose()
    if len(Xm) != N:
        if len(Xm)%2 ==0:
            Xm = np.concatenate((Xm[:int(len(Xm)/2)], np.zeros((N-len(Xm),np.shape(Xm)[1])), Xm[int(len(Xm)/2):]))
        else:
            Xm = np.concatenate((Xm[:int(len(Xm)/2)],np.ones((1,np.shape(Xm)[1]))*Xm[int(len(Xm)/2)]/2, np.zeros((N-len(Xm),np.shape(Xm)[1])),np.ones((1,np.shape(Xm)[1]))*Xm[int(len(Xm)/2)]/2, Xm[int(len(Xm) / 2)+1:]))
    xn = ifft(Xm.transpose()).transpose()*fs1

    return xn,Xm

def SD(Xm,fs):
    '''
    This function computes the single-sided spectral density (Gxx) from a linear spectrum.
    :param Xm: complex two-sided linear spectrum [Pa]
    :param fs: sampling rate [Hz]
    :return:
    '''

    # number of points in record
    N = len(Xm)
    # temporal resolution
    dt = fs ** -1
    # single sided power spectral density [Pa^2/Hz]
    Sxx = (dt * N) ** -1 * abs(Xm) ** 2
    Gxx = Sxx[:int(N / 2)]
    Gxx[1:-1] = 2 * Gxx[1:-1]
    return Gxx

def harm_extract(xn, tac_ind, fs, rev_skip, harm_filt,filt_shaft_harm,Nb):
    '''
    This function extracts te tonal and broadband components of a signal. The noise component separation is done in
    the frequency domain. The signal is first parsed on a rev-to-rev basis. The linear spectrum of each rev is then
    computed and upsampled by appending zeros to the linear spectrum of each record so that their lengths are equal.
    To determine the tonal noise components the linear spectrum is averaged across all revs. The averaged linear
    spectrum is then subtracted from that of each rev to determine the broadband contributions.

    The averaged linear spectrum can be filtered via the harm_filt BPF_harm parameters.

    :param xn: time series
    :param tac_ind: tachometer indices by which to parse the time series, ensure that the sample rates of the time series and TAC are equivalent or interpolated.
    :param fs: sampling rate of time series [Hz]
    :param rev_skip: number of intermediate revs to skip
    :param harm_filt: the BPF harmonic to retain, specified as a list [lowest BPF harmonic, highest BPF harmonic]
    :param filt_shaft_harm: boolean if set to True the shaft order harmonics will be filtered from the signal
    :param Nb:  number of blades, only needs to be specified when filt_shaft_harm is set to True
    :return:
    '''


    if len(np.shape(xn)) == 1:
        xn = np.expand_dims(xn,axis = 1)

    rev_skip = rev_skip+1
    dtac_ind = np.diff(tac_ind[::rev_skip])
    N = np.max(dtac_ind)
    N_avg = np.mean(dtac_ind)

    fs1 = N / N_avg * fs
    dt = fs1 ** -1
    df = (N * dt) ** -1

    xn_list = [xn[tac_ind[i]:tac_ind[i + 1]] for i in range(len(tac_ind[::rev_skip]) - 1)]

    out = np.array(list(map(lambda x: upsample(x, fs, N), xn_list)))
    Xn_avg = np.mean(out[:, 0, :, :], axis=0)
    Xm_avg = np.mean(out[:, 1, :, :], axis=0)
    u = 1.94*np.std(out[:, 1, :, :], axis=0)/np.sqrt(len(tac_ind)-1)

    Xm_bb = out[:, 1, :, :]-Xm_avg
    Xn_bb = (ifft(Xm_bb.transpose(),axis = 1).transpose()*fs1).reshape(np.shape(Xm_bb)[0]*np.shape(Xm_bb)[1],np.shape(Xm_bb)[2])

    if isinstance(harm_filt, list):

        Xm_avg[:harm_filt[0]] = 0
        Xm_avg[-harm_filt[0]+1:] = 0
        Xm_avg[harm_filt[1]+1:-harm_filt[1]] = 0

        u[:harm_filt[0]] = 0
        u[-harm_filt[0]+1:] = 0
        u[harm_filt[1]+1:-harm_filt[1]] = 0

        if filt_shaft_harm:
            Xm_avg[1::Nb] = 0

    f = np.arange(int(N / 2)) * df

    Xn_avg_filt = ifft(Xm_avg,axis = 0)*fs
    out = list(map(lambda x: SD(x,fs1),[Xm_avg,abs(Xm_avg)-u,abs(Xm_avg)+u]))
    spl = 10 * np.log10(out[0]*df / 20e-6 ** 2)

    u_low = spl-10 * np.log10(out[1]*df / 20e-6 ** 2)
    u_high = 10 * np.log10(out[2]*df / 20e-6 ** 2)-spl

    return f,fs1,spl,u_low, u_high, Xn_avg, Xm_avg,Xn_avg_filt, Xn_bb

def ffilter(xn,fs, btype, fc,filt_shaft_harm,Nb):
    '''
    This function filters the input signal in the frequency domain.
    :param xn: time series
    :param fs: sampling rate [Hz]
    :param btype: type of filter (lowpass (lp), highpass (hp), bandpass (bp))
    :param fc: cutoff frequency (-3dB), must be specified as a list if a bandpass filter is used.
    :return:
    '''
    df = (len(xn)*fs**-1)**-1
    f, Xm, Sxx, Gxx = PSD(xn, fs)

    if btype == 'lp':
        Xm[-int(fc/df):] = 0
    elif btype == 'hp':
        Xm[:int(fc/df)] = 0
    elif btype == 'bp':
        assert isinstance(fc,list), 'If a bandpass filter is being applied to the time series the low/high cutoff frequencies should be specified as a list.'
        Xm[:int(fc[0]/df)] = 0
        Xm[int(fc[-1]/df)+1:-int(fc[-1]/df)] = 0
        Xm[-int(fc[0]/df)+1:] = 0

    if filt_shaft_harm:
        Xm[1::Nb] = 0

    xn_filt = ifft(Xm,axis =0) * fs

    return Xm,xn_filt



