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
exp_dir ='/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/rpm_sweep/h2b/h2b8'

save_h5 = False
#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,9]

# min and max shaft order harmonics to exclude from the analysis (set max to -1 to include all upper order harmonics)
harm_filt = [2, 6]

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 70]

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

f, fs1, spl, u_low, u_high, Xn_avg, Xm_avg, Xn_avg_filt, Xn_bb = fun.harm_extract(exp, tac_ind=ind, fs=fs_exp, rev_skip=0, harm_filt=harm_filt, filt_shaft_harm =  True, Nb=Nb)

xn_inph = ifft((Xm_avg[:,mics[0]-1]+Xm_avg[:,mics[-1]-1])/2)*fs1

t_nondim = np.arange(len(Xn_avg_filt))*fs1**-1/(rpm_nom/60)**-1
BPF_harm = np.arange(len(Xn_avg_filt)/2)/Nb
df = (len(Xn_avg_filt)*fs1**-1)**-1

Xm_inph = (Xm_avg[:,mics[0]-1]+Xm_avg[:,mics[-1]-1])/2
Gxx_inph = fun.SD(Xm_inph,fs1)

Xm_outph = Xm_avg-np.expand_dims(Xm_inph,axis =1)
Gxx_outph = fun.SD(Xm_outph,fs1)

#%% Plots predicted pressure time series

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.25,bottom = 0.16)

#   Loops through each mic
for i,m in enumerate(mics):
    ax[i].plot(t_nondim, xn_inph,linestyle=':')
    ax[i].plot(t_nondim, Xn_avg_filt[:,m-1]-xn_inph,linestyle='-.')
    ax[i].plot(t_nondim, Xn_avg_filt[:,m-1],linestyle='-')

    ax[i].set_title(f'Mic {m}')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xlim([0,1])
    ax[i].set_ylabel('Pressure [Pa]')
    ax[i].grid('on')

ax[len(mics) - 1].set_xlabel('Rotation')
ax[len(mics) - 1].legend(['In-phase', 'Out-of-phase', 'Total'], loc='center', ncol=3,bbox_to_anchor=(.5, -.4))
plt.savefig(os.path.join(exp_dir,'rel_tseries.png'),format = 'png')
plt.savefig(os.path.join(exp_dir,'rel_tseries.eps'),format = 'eps')

 #%% Plots predicted spectrum

# #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
#
# #   Loops through each mic
# for i,m in enumerate(mics):
#     ax[i].stem(BPF_harm, 10*np.log10(Gxx_inph*df/20e-6**2),linefmt =f'C{0}{":"}', markerfmt =f'C{0}o',basefmt=f'C{0}')
#     ax[i].stem(BPF_harm, 10*np.log10(Gxx_outph[:, m - 1]*df/20e-6**2),linefmt =f'C{1}{"-."}', markerfmt =f'C{1}o',basefmt=f'C{1}')
#     ax[i].stem(BPF_harm, spl[:, m - 1], linefmt =f'C{2}{"-"}', markerfmt =f'C{2}o', basefmt=f'C{2}')
#
#     ax[i].set_title(f'Mic {m}')
#     if i!=len(mics)-1:
#         ax[i].tick_params(axis='x', labelsize=0)
#     # ax[ii].set_xscale('log')
#     ax[i].axis([0, 4, axis_lim[2], axis_lim[-1]])
#     ax[i].set_xticks(np.arange(1, 5))
#     # ax[i].set_xlim([0,harm_filt[-1]/Nb+1])
#     ax[i].grid('on')
#     ax[i].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
#
# ax[len(mics) - 1].set_xlabel('BPF Harmonic')
# ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
# ax[len(mics) - 1].legend(['In-phase', 'Out-of-phase', 'Total'],loc='center',ncol = 4, bbox_to_anchor=(.5, -.35))
# plt.savefig(os.path.join(exp_dir,'rel_spec.png'),format = 'png')

#%%
width = .125
hatch = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.25,bottom = 0.16)

#   Loops through each mic
for i,m in enumerate(mics):
    #   Plots the resulting spectra in dB
    ax[i].bar(BPF_harm[::2][1:4]+width*0-width, 10*np.log10(Gxx_inph[::2][1:4]*df/20e-6**2),width = width,hatch =hatch[0]*2,align='center')
    ax[i].bar(BPF_harm[::2][1:4]+width*1-width, 10*np.log10(Gxx_outph[::2, m - 1][1:4]*df/20e-6**2),width = width,hatch =hatch[1]*2,align='center')
    ax[i].bar(BPF_harm[::2][1:4]+width*2-width, spl[::2, m - 1][1:4],width = width,hatch =hatch[2]*2,align='center')

    ax[i].set_title(f'Mic {m}')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[ii].set_xscale('log')
    ax[i].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[i].set_xticks(np.arange(1, 5))
    # ax[i].set_xlim([0,harm_filt[-1]/Nb+1])
    ax[i].grid('on')
    ax[i].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

ax[len(mics) - 1].set_xlabel('BPF Harmonic')
ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax[len(mics) - 1].legend(['In-phase', 'Out-of-phase', 'Total'],loc='center',ncol = 4, bbox_to_anchor=(.5, -.4))
plt.savefig(os.path.join(exp_dir,'rel_spec.png'),format = 'png')
plt.savefig(os.path.join(exp_dir,'rel_spec.eps'),format = 'eps')

#%%
# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.
save_dir = os.path.join(exp_dir, os.path.basename(exp_dir) + '_sdata.h5')
if save_h5:
    sdata = {'Nb':Nb,'LE_ind':LE_ind, 'lim_ind':lim_ind, 'rpm_nom':rpm_nom, 'u_rpm':u_rpm,'t_nondim':t_nondim,'df':df,'f':f, 'fs1':fs1, 'spl':spl, 'u_low':u_low, 'u_high':u_high, 'Xn_avg':Xn_avg, 'Xm_avg':Xm_avg, 'Xn_avg_filt':Xn_avg_filt, 'Xn_bb':Xn_bb,'xn_inph':xn_inph,'Xm_inph':Xm_inph,'Gxx_inph':Gxx_inph,'Xm_outph':Xm_outph,'Gxx_outph':Gxx_outph}

    if os.path.exists(save_dir):
        os.remove(save_dir)

    with h5py.File(save_dir, 'a') as h5_f:
        for k, dat in sdata.items():
            h5_f.create_dataset(k, shape=np.shape(dat), data=dat)

#%%
# xn_inph = (exp[:,mics[0]-1]+exp[:,mics[-1]-1])/2
# f_inph, fs1_inph, spl_inph, u_low_inph, u_high_inph, Xn_avg_inph, Xm_avg_inph, Xn_avg_filt_inph, Xn_bb_inph = fun.harm_extract(xn_inph, tac_ind=np.array(ind), fs=fs_exp, rev_skip=0, harm_filt=harm_filt,filt_shaft_harm =  True,Nb=Nb)
# Xm_outph = Xm_avg-Xm_avg_inph
# spl_outph = 10*np.log10(fun.SD(Xm_outph,fs1)*df/20e-6**2)
#
# #%% Plots predicted pressure time series
#
# #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
#
# #   Loops through each mic
# for i,m in enumerate(mics):
#     ax[i].plot(t_nondim, np.squeeze(Xn_avg_filt_inph))
#     ax[i].plot(t_nondim, Xn_avg_filt[:,m-1]-np.squeeze(Xn_avg_filt_inph))
#     ax[i].plot(t_nondim, Xn_avg_filt[:,m-1])
#
#     ax[i].set_title(f'Mic {m}')
#     if i!=len(mics)-1:
#         ax[i].tick_params(axis='x', labelsize=0)
#     ax[i].set_xlim([0,1])
#     ax[i].set_ylabel('Pressure [Pa]')
#     ax[i].grid('on')
#
# ax[len(mics) - 1].set_xlabel('Time [sec]')
# ax[len(mics) - 1].legend(['In-phase', 'Out-of-phase', 'Total'], loc='center', ncol=3,bbox_to_anchor=(.5, -.35))
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
#     ax[i].stem(BPF_harm, spl_inph,linefmt =f'C{0}{":"}', markerfmt =f'C{0}o',basefmt=f'C{0}')
#     ax[i].stem(BPF_harm, spl_outph[:,m-1],linefmt =f'C{1}{"-."}', markerfmt =f'C{1}o',basefmt=f'C{1}')
#     ax[i].stem(BPF_harm, spl[:,m-1],linefmt =f'C{2}{"-"}', markerfmt =f'C{2}o',basefmt=f'C{2}')
#
#     ax[i].set_title(f'Mic {m}')
#     if i!=len(mics)-1:
#         ax[i].tick_params(axis='x', labelsize=0)
#     # ax[ii].set_xscale('log')
#     ax[i].set_yticks(np.arange(0, axis_lim[-1], 20))
#     ax[i].set_xticks(np.arange(1,harm_filt[-1]/Nb+1))
#     ax[i].set_xlim([0,harm_filt[-1]/Nb+1])
#     ax[i].grid('on')
#     ax[i].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
#
# ax[len(mics) - 1].set_xlabel('BPF Harmonic')
# ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
# ax[len(mics) - 1].legend(['In-phase', 'Out-of-phase', 'Total'],loc='center',ncol = 4, bbox_to_anchor=(.5, -.35))
#
