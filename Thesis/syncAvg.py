import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import bisect
from timeit import default_timer
#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})


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

#%%
fs_ttl = int(np.mean(dt) ** -1)
t = np.arange(len(ttl)) * fs_ttl ** -1
rpm = (np.diff(np.squeeze(np.where(np.diff(ttl) == 1))) / fs_ttl / 60) ** -1
t_rpm = t[np.squeeze(np.where(np.diff(ttl) == 1))]
t_acs = np.arange(len(acs_data))/fs_acs

#%%
N_rev = 10
ind = list(map(lambda x: bisect.bisect(t_acs,x),t_rpm[bisect.bisect(t_rpm,start_t):bisect.bisect(t_rpm,end_t)]))
N = np.max(np.diff(ind[::N_rev]))
df = fs_acs/N
dt = (df*N)**-1
acs_data_list = [acs_data[ind[i]:ind[i+1]] for i in range(len(ind[::N_rev])-1)]

#%%

def sync_avg(xn):
    Xm = fft(xn)*(len(xn)*df)**-1
    if len(Xm) != N:
        if len(Xm)%2:
            Xm = np.concatenate((Xm[:int(len(Xm)/2)], np.zeros(N-len(Xm)), Xm[int(len(Xm)/2):]))
        else:
            Xm = np.concatenate((Xm[:int(len(Xm) / 2)],np.expand_dims(Xm[int(len(Xm)/2)]/2,axis = 0), np.zeros(N - len(Xm)),np.expand_dims(Xm[int(len(Xm)/2)]/2,axis = 0), Xm[int(len(Xm) / 2)+1:]))
    xn_up = ifft(Xm)/dt
    return xn_up

xn = list(map(sync_avg,acs_data_list))
xn_avg = np.mean(xn,axis = 0)

#%%
i = 10
df = fs_acs/len(acs_data[ind[0]:ind[i]])
f =np.arange(int(len(acs_data[ind[0]:ind[i]])/2))*df
Xm_avg = fft(acs_data[ind[0]:ind[i]])*fs_acs**-1
Sxx = (fs_acs**-1*len(Xm_avg))**-1 * abs(Xm_avg) ** 2
Gxx = Sxx[:int(len(Xm_avg)/2)]
Gxx[1:-1]= 2*Gxx[1:-1]

#%%
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.plot(f,10*np.log10(Gxx*df/20e-6**2))
ax.set_xscale('log')
# ax.scatter(t_rpm,np.zeros(len(np.squeeze(np.where(np.diff(ttl) == 1)))),c = 'r')
ax.set_ylabel('$\overline{X_m}$')
# ax.set_xlabel('Time (s)')
ax.set_xlim([30, 15e3])
# ax.set_ylim([86, 89])
ax.grid()

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


