import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import bisect
from scipy.signal import lfilter, butter
from timeit import default_timer
#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})


#%%
def filt_response(bb,aa,fs,N):
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
    return f,y,h,phase

#%% Design Butterworth filter and check frequency response by applying the filter to an impulse response

fs=80e3
f0 = 131.5
f1 = f0-2
f2 = f0+2

f0 = np.tan(np.pi*f0/fs)/(np.pi/fs)
f1 = np.tan(np.pi*f1/fs)/(np.pi/fs)
f2 = np.tan(np.pi*f2/fs)/(np.pi/fs)

Q = f0/(f2-f1)
a  = fs/(np.pi*f0)
b = [a/Q,0,-a/Q]
a = [a**2+1+a/Q,2*(1-a**2),a**2+1-a/Q]
# b,a = butter(4,[(131.5-width)/(fs/2),(131.5+width)/(fs/2)], 'bp')
f,y,h,phase = filt_response(b,a,fs = fs,N = fs/8)

fig,ax = plt.subplots(2,1,figsize = (6.4,4.5))
ax[0].plot(f,10*np.log10(abs(h)))
ax[0].set_ylabel('Magnitude')
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].grid()
ax[0].set_xscale('log')
# ax[0].set_xlim(f[1],f[-1])

ax[1].plot(f,phase)
ax[1].set_ylabel('Phase [$\circ$]')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].grid()
ax[1].set_xscale('log')
# ax[1].set_xlim(f[1],f[-1])

#%%
#   Parent directory where all of the data files are contained.
dir ='/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/lat_cyc_sweep/highM/d2b/d2b51'

mic = 5
#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 15

#   Opens and reads in the acoustic data in the h5 file
with h5py.File(os.path.join(dir, 'acs_data.h5'), 'r') as dat_file:
    fs_acs = dat_file['Sampling Rate'][()]
    acs_data = (dat_file['Acoustic Data'][:].transpose()/(dat_file['Sensitivities'][:]*1e-3))[:,mic]
    ttl = dat_file['Motor1 RPM'][()]
    dt = np.diff(dat_file['Time (s)'])


fs_ttl = int(np.mean(dt) ** -1)
t = np.arange(len(ttl)) * fs_ttl ** -1
rpm = (np.diff(np.squeeze(np.where(np.diff(ttl) == 1))) / fs_ttl / 60) ** -1
t_rpm = t[np.squeeze(np.where(np.diff(ttl) == 1))]
t_rpm = t_rpm[bisect.bisect(t_rpm,start_t):bisect.bisect(t_rpm,end_t)]
t_acs = np.arange(len(acs_data))/fs_acs
Nb = 2

#%%
ind = list(map(lambda x: bisect.bisect(t_acs,x),t_rpm))
acs_data_list = [acs_data[ind[i]:ind[i+1]] for i in range(len(ind)-1)]
BPF = Nb*np.diff(t_rpm)**-1
Nharm = 1













# #%%
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
# ax.plot(ttl)
# ax.scatter(np.squeeze(np.where(np.diff(ttl) == 1)),np.ones(len(np.squeeze(np.where(np.diff(ttl) == 1))))*0,c = 'r')
# ax.set_ylabel('RPM')
# ax.set_xlabel('Time')
# # ax.set_xlim([-.15, .07])
# # ax.set_ylim([86, 89])
# ax.grid()


#%%


