import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from bisect import bisect


#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%

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


def SPL(Xm,fs):
    '''
    This function computes the sound pressure level (SPL) from a linear spectrum.
    :param Xm: complex two-sided linear spectrum [Pa]
    :param fs: sampling rate [Hz]
    :return:
    '''

    # number of points in record
    N = len(Xm)
    # temporal resolution
    dt = fs ** -1
    # frequency resolution
    df = (N * dt) ** -1

    # single sided power spectral density [Pa^2/Hz]
    Sxx = (dt * N) ** -1 * abs(Xm) ** 2
    Gxx = Sxx[:int(N / 2)]
    Gxx[1:-1] = 2 * Gxx[1:-1]
    # converts to SPL [dB]
    spl = 10 * np.log10(Gxx*df / 20e-6 ** 2)

    return spl

def harm_extract(xn, tac_ind, fs,rev_skip,harm_filt):


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
        Xm_avg[-harm_filt[0]:] = 0
        Xm_avg[harm_filt[1]:-harm_filt[1] - 1] = 0

        u[:harm_filt[0]] = 0
        u[-harm_filt[0]:] = 0
        u[harm_filt[1]:-harm_filt[1] - 1] = 0

    f = np.arange(int(N / 2)) * df

    out = list(map(lambda x: SPL(x,fs1),[Xm_avg,abs(Xm_avg)-u,abs(Xm_avg)+u]))
    spl = out[0]
    u_low = spl-out[1]
    u_high = out[2]-spl

    return f,fs1,spl,u_low, u_high, Xn_avg, Xm_avg, Xn_bb

#%%
#   Directory of data files.
dir ='/Users/danielweitsman/Downloads/NASA_data/ddhcs10'

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]
#   Frequency resolution of spectra [Hz]
df = 5
# min and max shaft order harmonics to exclude from the analysis (set max to -1 to include all upper order harmonics)
harm_filt = [3, 8]
#   Spectra axis limits specified as: [xmin,xmax,ymin,ymax]
spec_ax_lim = [100, 2.5e3, 0, 85]
#   Time series yaxis limits specified as (xlim set automatically to range from 0-1): [ymin,ymax]
ts_ax_ylim = [-0.35, 0.35]

linestyle =['-','-.','--','-']

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 15

#%%
#   Opens and reads in the acoustic and TAC data from the h5 files
with h5py.File(os.path.join(dir, 'acs_data.h5'), 'r') as dat_file:
    acs_data = (dat_file['Acoustic Data'][:].transpose() / (dat_file['Sensitivities'][:] * 1e-3))
    ttl1 = dat_file['Motor2 RPM'][()]
    ttl2 = dat_file['Motor2 RPM'][()]
    fs_acs = dat_file['Sampling Rate'][()]
    fs_ttl = round((np.mean(np.diff(dat_file['Time (s)'])))**-1)

#%%
# generates time vectors for the tac and acoustic data
t = np.arange(len(ttl1)) / fs_ttl
t_acs = np.arange(len(acs_data)) / fs_acs

#%%

LE_ind1, lim_ind1, rpm_nom1, u_rpm1 = fun.rpm_eval(ttl1,fs_ttl,start_t,end_t)

ind = list(map(lambda x: bisect(t_acs,x),t[LE_ind1[lim_ind1[0]:lim_ind1[1]]]))
f,fs1,spl,u_low, u_high, Xn_avg,Xm_avg,Xn_bb = harm_extract(acs_data, ind, fs_acs, 0, harm_filt)

f_bb,Gxx_bb,Gxx_avg_bb = fun.msPSD(Xn_bb, fs = fs1, df = df, save_fig = False, plot = False)

# construct time vector for the averaged rpm waveform
t_rev = (np.arange(len(Xn_avg))*(rpm_nom1/60)**-1/len(Xn_avg))

#%%

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
#   Loops through each mic
for i,m in enumerate(mics):
#   Computes the mean-square PSD spectra for each mic
    f_ms,Gxx_ms,Gxx_avg_ms = fun.msPSD(acs_data[int(fs_acs * start_t):int(fs_acs * end_t),m-1], fs_acs, df=df, ovr=0.75,plot = False,save_fig = False)
#   Plots the resulting spectra in dB
    if len(mics)>1:
        ax[i].plot(f_ms,10*np.log10(Gxx_avg_ms*df/20e-6**2),linestyle=linestyle[i])
        ax[i].plot(f_bb,10*np.log10(Gxx_avg_bb[:,m-1]*df/20e-6**2),linestyle=linestyle[i])

        ax[i].errorbar(f,spl[:,m-1],yerr=np.array([u_low[:,m-1],u_high[:,m-1]]),fmt='o',capsize=5,capthick=1.5,elinewidth = 1.5)

    else:
        ax.plot(f,10*np.log10(Gxx_avg_ms*df/20e-6**2),linestyle=linestyle[i])
        ax.errorbar(f,spl[:,m-1],yerr=np.array([u_low[:,m-1],u_high[:,m-1]]),fmt='o',capsize=5,capthick=1.5,elinewidth = 1.5)

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for ii, m in enumerate(mics):
        ax[ii].set_title('Mic: '+str(m))
        if ii!=len(mics)-1:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[ii].set_xscale('log')
        ax[ii].set_yticks(np.arange(0, spec_ax_lim[-1], 20))
        ax[ii].axis(spec_ax_lim)
        ax[ii].grid('on')
    ax[len(mics) - 1].set_xlabel('Frequency (Hz)')
    ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title('Mic: ' + str(mics[0]))
    ax.set_xscale('log')
    ax.axis(spec_ax_lim)
    ax.set_yticks(np.arange(0, spec_ax_lim[-1], 20))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax.grid('on')

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
#   Loops through each mic
for i,m in enumerate(mics):
#   Plots the resulting spectra in dB
    if len(mics)>1:
        ax[i].plot(t_rev/(rpm_nom1/60)**-1,Xn_avg[:,m-1]-np.mean(Xn_avg[:,m-1]),linestyle=linestyle[i])

    else:
        ax.plot(t_rev/(rpm_nom1/60)**-1,Xn_avg[:,m-1]-np.mean(Xn_avg[:,m-1]),linestyle=linestyle[i])

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for ii, m in enumerate(mics):
        ax[ii].set_title('Mic: '+str(m))
        if ii!=len(mics)-1:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[ii].set_xlim([0,1])
        ax[ii].set_ylim(ts_ax_ylim)
        ax[ii].grid('on')
    ax[len(mics) - 1].set_xlabel('Revolution')
    ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title('Mic: ' + str(mics[0]))
    ax.set_xlim(0,1)
    ax.set_ylim(ts_ax_ylim)
    # ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Revolution')
    ax.set_ylabel('Pressure [Pa]')
    ax.grid('on')

