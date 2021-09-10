
import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft

#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 22
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
def spl(dat):
    # number of points
    N = np.shape(dat)[2]
    # sampling rate [Hz]
    fs_pred = (N - 1) / dat[0, 0, -1, 0]
    # temporal resolution [s]
    dt = fs_pred ** -1
    # frequency resolution
    df = (dt * N) ** -1
    # frequency vector [Hz]
    f = np.arange(N) * df
    # linear spectrum [Pa]
    Xm = fft(np.squeeze(dat), axis=1) * dt
    # single sided power spectral density [Pa^2/Hz]
    Sxx = (dt * N) ** -1 * abs(Xm) ** 2
    Gxx = Sxx[:int(N / 2)]
    Gxx[1:-1] = 2 * Gxx[1:-1]
    # converts to single-sided PSD to SPL [dB]
    spl = 10 * np.log10(Gxx * df / 20e-6 ** 2)
    return f, spl

#%%
#   Parent directory where all of the data files are contained.
exp_dir ='///Users/danielweitsman/Box/Jan21Test/TAMU/runs/cshb/cshb20'
bet_dir = '///Users/danielweitsman/Desktop/Masters_Research/TAMU/validations/BEMT/straight/straight_DegenGeom'
cfd_dir = '///Users/danielweitsman/Desktop/Masters_Research/TAMU/validations/CFD/straight/straight_DegenGeom'

leglab = ['Measured','BEMT','CFD']
# leglab=''

#   Linestyle for each case
linestyle =['-','-.','--','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]

#   Frequency resolution of spectra [Hz]
df_exp = 2
#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [100, 2e3, 0, 80]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 15

#%%

bet_data={}
cfd_data={}

with h5py.File(os.path.join(exp_dir, 'acs_data.h5'), 'r') as dat_file:
    exp_data = dat_file['Acoustic Data'][:].transpose() / (dat_file['Sensitivities'][:] * 1e-3)
    fs = dat_file['Sampling Rate'][()]

with h5py.File(os.path.join(bet_dir, 'pressure.h5'), 'r') as dat_file:
    for i,n in dat_file.items():
        bet_data = {**bet_data, **{i:n[()]}}

with h5py.File(os.path.join(cfd_dir, 'pressure.h5'), 'r') as dat_file:
    for i,n in dat_file.items():
        cfd_data = {**cfd_data, **{i:n[()]}}

#%%

micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
exp_data = exp_data*micR/micR[4]

#%%
bet_data['geometry_values'] = np.flip(bet_data['geometry_values'], axis = 1)
bet_data['function_values'] = np.flip(bet_data['function_values'], axis = 1)

cfd_data['geometry_values'] = np.flip(cfd_data['geometry_values'], axis = 1)
cfd_data['function_values'] = np.flip(cfd_data['function_values'], axis = 1)


coord = np.squeeze(bet_data['geometry_values'][:, :, 0, :])
phi = np.arctan2(coord[:, -1], coord[:, 1]) * 180 / np.pi
azi = np.arctan2(coord[:, 1], coord[:, 0]) * 180 / np.pi

OASPL_bet = 10 * np.log10(np.mean((bet_data['function_values'][:, :, :, 1:] - np.expand_dims(np.mean(bet_data['function_values'][:, :, :, 1:], axis = 2), axis = 2)) ** 2, axis=2) / 20e-6 ** 2)
OASPL_cfd = 10 * np.log10(np.mean((cfd_data['function_values'][:, :, :, 1:] - np.expand_dims(np.mean(cfd_data['function_values'][:, :, :, 1:], axis = 2), axis = 2)) ** 2, axis=2) / 20e-6 ** 2)
OASPL_exp = 10 * np.log10(np.mean(exp_data[int(fs*start_t):int(fs*end_t)]**2,axis = 0)/ 20e-6 ** 2)

#%%
f_bet, spl_bet = spl(bet_data['function_values'])
f_cfd, spl_cfd = spl(cfd_data['function_values'])

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(np.shape(spl_bet)[-1] - 1):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(bet_data['function_values'][0, m - 1, :, 0], bet_data['function_values'][0, m - 1, :, src + 1], linestyle = linestyle[src])
        else:
            ax.plot(bet_data['function_values'][0, m - 1, :, 0], bet_data['function_values'][0, m - 1, :, src + 1], linestyle =  linestyle[src])

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for src, m in enumerate(mics):
        ax[src].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
        if src!=len(mics)-1:
            ax[src].tick_params(axis='x', labelsize=0)
        # ax[ii].set_xscale('log')
        # ax[ii].set_yticks(np.arange(0,axis_lim[-1],20))
        # ax[ii].axis(axis_lim)
        ax[src].grid('on')

    ax[len(mics) - 1].set_xlabel('Time [sec]')
    ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')

    ax[len(mics) - 1].legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
    # ax.set_xscale('log')
    # ax.axis(axis_lim)
    # ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Pressure [Pa]')
    ax.grid('on')
    ax.legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.175))
#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(np.shape(spl_bet)[-1] - 1):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(bet_data['function_values'][0, m - 1, :, 0], bet_data['function_values'][0, m - 1, :, src + 1], linestyle = linestyle[src])
        else:
            ax.plot(bet_data['function_values'][0, m - 1, :, 0], bet_data['function_values'][0, m - 1, :, src + 1], linestyle =  linestyle[src])

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for src, m in enumerate(mics):
        ax[src].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
        if src!=len(mics)-1:
            ax[src].tick_params(axis='x', labelsize=0)
        # ax[ii].set_xscale('log')
        # ax[ii].set_yticks(np.arange(0,axis_lim[-1],20))
        # ax[ii].axis(axis_lim)
        ax[src].grid('on')

    ax[len(mics) - 1].set_xlabel('Time [sec]')
    ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')

    ax[len(mics) - 1].legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
    # ax.set_xscale('log')
    # ax.axis(axis_lim)
    # ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Pressure [Pa]')
    ax.grid('on')
    ax.legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.175))
#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(np.shape(spl_bet)[-1] - 1):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(cfd_data['function_values'][0, m - 1, :, 0], cfd_data['function_values'][0, m - 1, :, src + 1], linestyle = linestyle[src])
        else:
            ax.plot(cfd_data['function_values'][0, m - 1, :, 0], cfd_data['function_values'][0, m - 1, :, src + 1], linestyle =  linestyle[src])

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for src, m in enumerate(mics):
        ax[src].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
        if src!=len(mics)-1:
            ax[src].tick_params(axis='x', labelsize=0)
        # ax[ii].set_xscale('log')
        # ax[ii].set_yticks(np.arange(0,axis_lim[-1],20))
        # ax[ii].axis(axis_lim)
        ax[src].grid('on')

    ax[len(mics) - 1].set_xlabel('Time [sec]')
    ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')

    ax[len(mics) - 1].legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
    # ax.set_xscale('log')
    # ax.axis(axis_lim)
    # ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Pressure [Pa]')
    ax.grid('on')
    ax.legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.175))

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(np.shape(spl_bet)[-1] - 1):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].stem(f_bet, spl_bet[m-1,:, src + 1], linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')
        else:
            ax.stem(f_bet, spl_bet[m-1, :, src + 1], linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for src, m in enumerate(mics):
        ax[src].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
        if src!=len(mics)-1:
            ax[src].tick_params(axis='x', labelsize=0)
        # ax[ii].set_xscale('log')
        ax[src].set_yticks(np.arange(0, axis_lim[-1], 20))
        ax[src].axis(axis_lim)
        ax[src].grid('on')

    ax[len(mics) - 1].set_xlabel('Frequency (Hz)')
    ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[len(mics) - 1].legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
    # ax.set_xscale('log')
    ax.axis(axis_lim)
    ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax.grid('on')
    ax.legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.175))
#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(np.shape(spl_bet)[-1] - 1):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].stem(f_cfd, spl_cfd[m-1,:, src + 1], linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')
        else:
            ax.stem(f_cfd, spl_cfd[m-1, :, src + 1], linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for src, m in enumerate(mics):
        ax[src].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
        if src!=len(mics)-1:
            ax[src].tick_params(axis='x', labelsize=0)
        # ax[ii].set_xscale('log')
        ax[src].set_yticks(np.arange(0, axis_lim[-1], 20))
        ax[src].axis(axis_lim)
        ax[src].grid('on')

    ax[len(mics) - 1].set_xlabel('Frequency (Hz)')
    ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[len(mics) - 1].legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
    # ax.set_xscale('log')
    ax.axis(axis_lim)
    ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax.grid('on')
    ax.legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.175))

#%%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
for src in range(np.shape(spl_bet)[-1] - 1):
    ax.plot(phi * np.pi / 180, np.squeeze(OASPL_bet[0, :, src]), linestyle = linestyle[src])
ax.set_thetamax(phi[0]+2)
ax.set_thetamin(phi[-1]-2)
ax.set_ylim([20,60])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.15),  labelpad = -100, rotation = phi[-1]-5)
ax.legend(['Thickness', 'Loading', 'Total'], ncol=1,loc='center',bbox_to_anchor=(.25, 0.9))

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.42,bottom = 0.19)

#   Loops through each mic
for i,m in enumerate(mics):
#   Computes the mean-square PSD spectra for each mic
    f_exp,Gxx_exp,Gxx_avg_exp = fun.msPSD(exp_data[int(fs*start_t):int(fs*end_t),m-1], fs, df = df_exp, ovr=0.75,plot = False,save_fig = False)
#   Plots the resulting spectra in dB
    if len(mics)>1:
        ax[i].plot(f_exp,10*np.log10(Gxx_avg_exp*df_exp/20e-6**2),linestyle=linestyle[0])
        ax[i].stem(f_bet,spl_bet[m-1,:,-1],linefmt = 'C1-',markerfmt = 'C1o',basefmt='C1')
        ax[i].stem(f_cfd,spl_cfd[m-1,:,-1],linefmt = 'C2:',markerfmt = 'C2^',basefmt='C2')

    else:
        ax.plot(f_exp,10*np.log10(Gxx_avg_exp*df_exp/20e-6**2),linestyle=linestyle[i])
        ax.stem(f_bet, spl_bet[m-1, :, -1],linefmt = 'C1-',markerfmt = 'C1o',basefmt='C1')
        ax.stem(f_cfd, spl_cfd[m-1, :, -1],linefmt = 'C2:',markerfmt = 'C2^',basefmt='C2')

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for src, m in enumerate(mics):
        ax[src].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
        if src!=len(mics)-1:
            ax[src].tick_params(axis='x', labelsize=0)
        # ax[ii].set_xscale('log')
        ax[src].set_yticks(np.arange(0, axis_lim[-1], 20))
        ax[src].axis(axis_lim)
        ax[src].grid('on')

    ax[len(mics) - 1].set_xlabel('Frequency (Hz)')
    ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[len(mics) - 1].legend(['Measured','BEMT','CFD'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.9))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
    # ax.set_xscale('log')
    ax.axis(axis_lim)
    ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax.grid('on')
    ax.legend(['Measured','BEMT','CFD'],loc='center',ncol = 3, bbox_to_anchor=(.5, 0))

#%%
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
ax.plot(phi * np.pi / 180, np.squeeze(OASPL_exp[:9]), linestyle = linestyle[src])
ax.plot(phi * np.pi / 180, np.squeeze(OASPL_bet[0, :, -1]), linestyle = linestyle[src+1])
ax.plot(phi * np.pi / 180, np.squeeze(OASPL_cfd[0, :, -1]), linestyle = linestyle[src+2])
ax.set_thetamax(phi[0]+2)
ax.set_thetamin(phi[-1]-2)
ax.set_ylim([0,100])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.15),  labelpad = -100, rotation = phi[-1]-5)
ax.legend(['Measured','BET','CFD'], ncol=1,loc='center',bbox_to_anchor=(.25, 0.9))
