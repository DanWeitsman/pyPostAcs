import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt

#%%
fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
dir ='/Users/danielweitsman/OneDrive - The Pennsylvania State University/January2021TestCampaign/PhasedArrayCalibration/22Jan2021'
caseName  = ['Sweep_Cal2']

mic = 5
df = 5

fig,ax = plt.subplots(1,1,figsize = (6.4,4.5))

for i,case in enumerate(caseName):

    with h5py.File(os.path.join(dir,caseName[i], 'acs_data.h5'), 'r') as dat_file:
        data = dat_file['Acoustic Data (mV)'][:].transpose() / dat_file['Sensitivities (mV_Pa)']

        f, Gxx = fun.msPSD(data[:,mic], dat_file.attrs['Sampling Rate (Hz)'], df=df, win='hann', ovr=0.5)
        ax.plot(f,10*np.log10(Gxx/20e-6**2),label = caseName[i])

    ax.set_xscale('log')
    # ax.axis([10,10e3,-20,100])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('$PSD \:dB\: (re:\: 20 \:\mu Pa/Hz)$')
    ax.legend()
    ax.grid()
    ax.set_title('Mic ' + str(mic))