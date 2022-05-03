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
fontSize = 16
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})


#%%
#   Parent directory where all of the data files are contained.
exp_dir ='/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/h2b69'

save_fig = True
#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]

# min and max shaft order harmonics to exclude from the analysis (set max to -1 to include all upper order harmonics)
harm_filt = [2, 6]

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 75]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 15


#%% imports experimental data
with h5py.File(os.path.join(exp_dir, 'acs_data.h5'), 'r') as dat_file:
    exp = dat_file['Acoustic Data'][:].transpose() / (dat_file['Sensitivities'][:] * 1e-3)
    fs_exp = dat_file['Sampling Rate'][()]
    ttl = dat_file['Motor1 RPM'][()]
    fs_ttl = round((np.mean(np.diff(dat_file['Time (s)']))) ** -1)

save_dir = os.path.join(exp_dir,'figures')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


#%% Spherical spreading correction

# micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
# exp = exp * micR / micR[4]

#%%
# Xm = fft(exp[int(fs_exp*start_t):int(fs_exp*end_t),mics[0]-1]) * fs_exp ** -1
# Ym = fft(exp[int(fs_exp*start_t):int(fs_exp*end_t),mics[-1]-1]) * fs_exp ** -1
# Sxy = 1 / ((end_t-start_t)*fs_exp * fs_exp ** -1) * np.conj(Xm) * Ym
# Rxy = ifft(Sxy) * fs_exp
# N = len(Rxy)
# t = np.arange(N) * fs_exp**-1 - N * fs_exp**-1 / 2
# shiftRxy = np.concatenate((Rxy[int(N / 2):], Rxy[:int(N / 2)]))
# t[np.where(abs(shiftRxy) == np.max(abs(shiftRxy)))]
#
# #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(1,1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# # plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
# ax.plot(t,shiftRxy)
# ax.set_xlim(-5,5)
# ax.set_xlabel('Time Delay [s]')
# ax.set_ylabel('$Cross \ Correlation, \ R_{xy}$')

#%% Average tseries w/ ms average
# df = 2
# BPF = 119
#
# xn_inph = (exp[:,mics[0]-1]+exp[:,mics[-1]-1])/2
# f,Xm,Sxx,Gxx,Gxx_avg = fun.msPSD(exp[int(fs_exp*start_t):int(fs_exp*end_t)], fs=fs_exp, df=df, win=True,ovr=0.5, save_fig=False, plot=False)
# f,Xm_inph,Sxx_inph,Gxx_inph,Gxx_avg_inph = fun.msPSD(xn_inph[int(fs_exp*start_t):int(fs_exp*end_t)], fs=fs_exp, df=df, win=True,ovr=0.5, save_fig=False, plot=False)
#
#  #%% Plots predicted spectrum
#
# #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
#
# #   Loops through each mic
# for i,m in enumerate(mics):
#     ax[i].plot(f, 10*np.log10(Gxx_avg_inph[:,0]*df/20e-6**2),linestyle = ':')
#     ax[i].plot(f, 10*np.log10((Gxx_avg[:,m-1]-Gxx_avg_inph[:,0])*df/20e-6**2),linestyle = '-.')
#     ax[i].plot(f, 10*np.log10(Gxx_avg[:,m-1]*df/20e-6**2),linestyle = '-')
#
#     ax[i].set_title(f'Mic {m}')
#     if i!=len(mics)-1:
#         ax[i].tick_params(axis='x', labelsize=0)
#     # ax[ii].set_xscale('log')
#     ax[i].set_yticks(np.arange(0, axis_lim[-1], 20))
#     ax[i].axis(axis_lim)
#     ax[i].grid('on')
#     ax[i].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
#
# ax[len(mics) - 1].set_xlabel('Frequency (Hz)')
# ax[len(mics) - 1].legend(['In-phase', 'Out-of-phase', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.35))
#

#%% Sync average total

Nb = 2
t = np.arange(len(ttl)) / fs_ttl
t_acs = np.arange(len(exp)) / fs_exp

LE_ind, lim_ind, rpm_nom, u_rpm = fun.rpm_eval(ttl,fs_ttl,start_t,end_t)
ind = list(map(lambda x: bisect(t_acs,x),t[LE_ind[lim_ind[0]:lim_ind[1]]]))

#%%
BPF = Nb*rpm_nom/60

f, fs1, spl, u_low, u_high, Xn_avg, Xm_avg, Xn_avg_filt, Xn_bb = fun.harm_extract(exp, tac_ind=ind, fs=fs_exp, rev_skip=0, harm_filt=harm_filt, filt_shaft_harm =  True, Nb=Nb)

df = 2
f_nb,Xm_nb,Sxx_nb,Gxx_nb,Gxx_avg_nb = fun.msPSD(exp[int(fs_exp * start_t):int(fs_exp * end_t),:], fs = fs_exp, df = df, win = True, ovr = 0.5, f_lim =[10,5e3], levels = [0,100],save_fig = False, save_path = '',plot = False)
f_bb,Xm_bb,Sxx_bb,Gxx_bb,Gxx_avg_bb = fun.msPSD(Xn_bb, fs = fs1, df = df, save_fig = False, plot = False)

BPF_harm = np.arange(len(Xn_avg_filt)/2)/Nb


#%% Plots predicted pressure time series

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.3,bottom = 0.15)
#   Loops through each mic
for i,m in enumerate(mics):
    ax[i].plot(f_nb,10*np.log10(Gxx_avg_nb[:,m-1]*df/20e-6**2),linestyle='-')
    ax[i].plot(f_bb,10*np.log10(Gxx_avg_bb[:,m-1]*df/20e-6**2),linestyle='-.')
    ax[i].errorbar(f,spl[:,m-1],yerr=np.array([u_low[:,m-1],u_high[:,m-1]]),fmt='o',capsize=5,capthick=1.5,elinewidth = 1.5)

    ax[i].set_title(f'M{m}')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xscale('log')
    ax[i].set_yticks(np.arange(0, axis_lim[-1], 20))
    ax[i].axis(axis_lim)
    ax[i].grid('on')
ax[-1].set_xlabel('Frequency (Hz)',labelpad=-10)
ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax[-1].legend(['Narrowband', 'Broadband','Tonal'], loc='center', ncol=3,bbox_to_anchor=(.5, -.6))


if save_fig:
    plt.savefig(os.path.join(save_dir, f'rel_spec.eps'), format='eps')
    plt.savefig(os.path.join(save_dir, f'rel_spec.png'), format='png')

#%%

for m in range(np.shape(Gxx_avg_nb)[-1]):

    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig,ax = plt.subplots(1,1,figsize = (8,6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace = 0.3,bottom = 0.15)
    ax.plot(f_nb,10*np.log10(Gxx_avg_nb[:,m]*df/20e-6**2),linestyle='-')
    ax.plot(f_bb,10*np.log10(Gxx_avg_bb[:,m]*df/20e-6**2),linestyle='-.')
    ax.errorbar(f,spl[:,m],yerr=np.array([u_low[:,m],u_high[:,m]]),fmt='o',capsize=5,capthick=1.5,elinewidth = 1.5)
    ax.set_title(f'M{m}')
    ax.set_xscale('log')
    ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.axis(axis_lim)
    ax.grid('on')
    ax.set_xlabel('Frequency (Hz)',labelpad=-10)
    ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax.legend(['Narrowband', 'Broadband','Tonal'], loc='center', ncol=3,bbox_to_anchor=(.5, -.15))


    if save_fig:
        plt.savefig(os.path.join(save_dir, f'rel_spec_m{m+1}.eps'), format='eps')
        plt.savefig(os.path.join(save_dir, f'rel_spec.png{m+1}.png'), format='png')