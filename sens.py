import os
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import h5py
from scipy.signal import butter,lfilter
import pyPostAcsFun
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
#%%
# fs = 80e3
# b,a = butter(4,[100/(fs/2),1e3/(fs/2)], 'bp')
# f,y,h,phase = filt_response(b,a,fs = fs,N = fs*5)
#
# fig,ax = plt.subplots(2,1,figsize = (6.4,4.5))
# ax[0].plot(f,20*np.log10(abs(h)))
# ax[0].set_ylabel('Magnitude')
# ax[0].tick_params(axis = 'x', labelsize=0)
# ax[0].grid()
# ax[0].set_xscale('log')
# ax[0].set_xlim(f[0],f[-1])
#
# ax[1].plot(f,phase)
# ax[1].set_ylabel('Phase [$\circ$]')
# ax[1].set_xlabel('Frequency [Hz]')
# ax[1].grid()
# ax[1].set_xscale('log')
# ax[1].set_xlim(f[0],f[-1])


#%%
dir ='/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/January2021TestCampaign/CalibrationMicrophones/3_15_20'
mics = 10
S_spec = np.zeros(mics)
S_ts = np.zeros(mics)

#%%
for i in range(mics):
    with h5py.File(os.path.join(dir,'mic'+str(i+1),'acs_data.h5'),'r') as f:
        fs = f['Sampling Rate'][()]
        # fs = 40e3
        data = f['Acoustic Data'][()]

    #%%
    N = np.shape(data)[1]
    df = (fs**-1*N)**-1
    f = np.arange(N/2)*df
    Xm = fft(data)*fs**-1
    Sxx = (fs**-1*N)**-1 * abs(Xm) ** 2
    Gxx = (Sxx[:,:int(N/2)]).transpose()
    Gxx[1:-1] = 2*Gxx[1:-1]

    # plt.plot(f[:int(1500/df)], Gxx[:int(1500/df)])
    S_spec[i] = np.sqrt(np.trapz(Gxx[int((1000-80*df)/df):int((1000+80*df)/df)], dx = df,axis = 0))/1e-3

    b, a = butter(4, 250 / (fs / 2), 'hp')
    data_filt = lfilter(b, a, data)
    S_ts[i] = np.sqrt(np.mean(data_filt**2,axis = 1))/1e-3

    # print('Sens Spectra: '+str(np.squeeze(S_spec))+' ; Sens Tseries: '+str(np.squeeze(S_ts)))

#%%
dt = fs**-1
# df = (dt*len(data))**-1
# f = np.arange(len(data) / 2)*df
#
# Xm = fft(data/S*1e-3)*dt
# Sxx = (dt * len(data)) ** -1 * abs(Xm) ** 2
# Gxx = Sxx[:int(len(data) / 2)]
# Gxx[1:-1] = 2 * Gxx[1:-1]
#
# # #%%
# # plt.plot(f,Gxx)
# # plt.xlim([0,2000])