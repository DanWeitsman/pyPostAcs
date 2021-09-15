import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft
import sys
from bisect import bisect

#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
#   Parent directory where all of the data files are contained.
exp_dir ='/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/h2b69'

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]

# min and max shaft order harmonics to exclude from the analysis (set max to -1 to include all upper order harmonics)
harm_filt = [2,4]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t =12.5

#%% imports experimental data
with h5py.File(os.path.join(exp_dir, 'acs_data.h5'), 'r') as dat_file:
    exp = dat_file['Acoustic Data'][:].transpose() / (dat_file['Sensitivities'][:] * 1e-3)
    fs_exp = dat_file['Sampling Rate'][()]
    ttl = dat_file['Motor1 RPM'][()]
    fs_ttl = round((np.mean(np.diff(dat_file['Time (s)']))) ** -1)

#%% Spherical spreading correction

micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
exp = exp * micR / micR[4]

#%%

Nb = 2
t = np.arange(len(ttl)) / fs_ttl
t_acs = np.arange(len(exp)) / fs_exp
rpm = (np.diff(np.squeeze(np.where(np.diff(ttl) == 1))) / fs_ttl / 60) ** -1

LE_ind, lim_ind, rpm_nom, u_rpm = fun.rpm_eval(ttl,fs_ttl,start_t,end_t)
tac_ind = list(map(lambda x: bisect(t_acs,x),t[LE_ind[lim_ind[0]:lim_ind[1]]]))
# tac_rpm = (np.diff(tac_ind)*fs_exp**-1)**-1*60
tac_rpm = rpm[lim_ind[0]:lim_ind[1]]
t_rpm = t[LE_ind]

fit_ind = [bisect(t_rpm,6.47),bisect(t_rpm,6.7)]
rpm_fit =np.poly1d(np.polyfit(t_rpm[fit_ind[0]:fit_ind[1]],rpm[fit_ind[0]:fit_ind[1]],1))
accel = rpm_fit[1]/60*np.pi*2

rev_skip = 0
rev_skip = rev_skip + 1
xn_list = [exp[tac_ind[i]:tac_ind[i + 1]] for i in range(len(tac_ind[::rev_skip]) - 1)]
t_list = [np.arange(n)*fs_exp**-1*(tac_rpm[i]/60) for i,n in enumerate(np.diff(tac_ind))]

xn_list_filt = [fun.ffilter(n,fs_exp, btype='bp', fc = [tac_rpm[i]/60*harm_filt[0],tac_rpm[i]/60*harm_filt[-1]], filt_shaft_harm=True,Nb=2)[1] for i,n in enumerate(xn_list)]
xn_flat_list = np.array([item for sublist in xn_list_filt for item in sublist])

f, fs1, spl, u_low, u_high, Xn_avg, Xm_avg, Xn_avg_filt, Xn_bb = fun.harm_extract(exp, tac_ind=tac_ind, fs=fs_exp, rev_skip=0, harm_filt=harm_filt, filt_shaft_harm =  True, Nb=Nb)
t_nondim = np.arange(len(Xn_avg_filt))*fs1**-1/(rpm_nom/60)**-1

#%%

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35)

#   Loops through each mic
for i,m in enumerate(mics):
    #   Plots the resulting spectra in dB
    if len(mics)>1:
        for rev in range(len(xn_list)):
            ax[i].plot(t_list[rev], xn_list_filt[rev][:, m-1],c = 'gray',lw = .5)
        ax[i].plot(t_nondim, Xn_avg_filt[:, m - 1],c ='r',lw = 2)

    else:
        for rev in range(len(xn_list)):
            ax.plot(t_list[rev], xn_list_filt[rev][:, m-1],c = 'black',lw = .25,alpha = .5)
        ax.plot(t_nondim, Xn_avg_filt[:, m - 1],c ='r',lw = 2)

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for ii, m in enumerate(mics):
        ax[ii].set_title('Mic: '+str(m))
        if ii!=len(mics)-1:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[ii].set_xlim([0,1])
        ax[ii].set_ylim([-.02,0.02])
        ax[ii].grid('on')
    ax[len(mics) - 1].set_xlabel('Revolution')
    ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title('Mic: ' + str(mics[0]))
    ax.set_xlim(0,1)
    ax.set_ylim([-.02, 0.02])
    # ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Revolution')
    ax.set_ylabel('Pressure [Pa]')
    ax.grid('on')

#%%

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35)

#   Loops through each mic
for i,m in enumerate(mics):
    #   Plots the resulting spectra in dB
    if len(mics)>1:
        ax[i].plot( xn_flat_list[:, m-1],c = 'gray',lw = .5)

    else:
        ax.plot( xn_flat_list[:, m-1],c = 'black',lw = .25,alpha = .5)

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for ii, m in enumerate(mics):
        ax[ii].set_title('Mic: '+str(m))
        if ii!=len(mics)-1:
            ax[ii].tick_params(axis='x', labelsize=0)
        # ax[ii].set_xlim([0,1])
        # ax[ii].set_ylim([-.02,0.02])
        ax[ii].grid('on')
    ax[len(mics) - 1].set_xlabel('Revolution')
    ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title('Mic: ' + str(mics[0]))
    ax.set_xlim(0,1)
    # ax.set_ylim([-.02, 0.02])
    # ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Revolution')
    ax.set_ylabel('Pressure [Pa]')
    ax.grid('on')

#%%

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(1,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35)

ax.plot(t_rpm[:-1],rpm)
ax.plot(t_rpm[fit_ind[0]:fit_ind[1]],rpm_fit(t_rpm[fit_ind[0]:fit_ind[1]]))
ax.set_xlabel('Time [s]')
ax.set_ylabel('RPM')
ax.axis([6,14,0,rpm_nom+250])
ax.grid('on')

bisect(t[LE_ind],6.5)