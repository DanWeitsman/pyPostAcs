
import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft
import sys
sys.path.insert(0, '/Users/danielweitsman/Desktop/Masters_Research/py scripts/WOPWOP_PostProcess/pyWopwop')
import wopwop
import matplotlib.colors as mcolors

#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%

def PSD(data):
    '''
    This function computes the power spectral density (dB) of the tonal components predicted by wopwop and saved in the pressure.fn file
    :param data: 'function value' entree in the 'pressure dictionary'
    :return:
    :param f: frequency vector [Hz]
    :param spl: sound pressure level [dB]
    '''
    # number of points
    N = np.shape(data)[2]
    # sampling rate [Hz]
    fs_pred = N / data[0, 0, -1, 0]
    # temporal resolution [s]
    dt = fs_pred ** -1
    # frequency resolution
    df = (dt * N) ** -1
    # frequency vector [Hz]
    f = np.arange(N) * df
    # linear spectrum [Pa]
    Xm = fft(np.squeeze(data[:,:,:,1:]), axis=1) * dt
    # single sided power spectral density [Pa^2/Hz]
    Sxx = (dt * N) ** -1 * abs(Xm) ** 2
    Gxx = Sxx[:int(N / 2)]
    Gxx[1:-1] = 2 * Gxx[1:-1]
    # converts to single-sided PSD to SPL [dB]
    spl = 10 * np.log10(Gxx * df / 20e-6 ** 2)

    return f, spl


#%%
#   Parent directory where all of the data files are contained.
exp_dir ='/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/rpm_sweep/h2b/h2b8'

pred_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b8_1rev/'

#   raw wopwop output file names
file = ['pressure.h5']

# set equal to true to reformat the raw data from wopwop and write it out as an HDF5 file, this only needs to be set
# to true if this is your first time working with the predicted data.
h5_write = False

#   legend labels
leglab = ['Measured','Predicted']
# leglab=''

#   Linestyle for each case
linestyle =['-','-.','--','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]

#   Frequency resolution of spectra [Hz]
df_exp = 2
#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 70]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 15

#%%

with h5py.File(os.path.join(exp_dir, 'acs_data.h5'), 'r') as dat_file:
    exp = dat_file['Acoustic Data'][:].transpose() / (dat_file['Sensitivities'][:] * 1e-3)
    fs = dat_file['Sampling Rate'][()]

#%%
#   generates and imports the HDF5 file containing the reformatted pressure time series and BPM spectral data.
if h5_write:
    functions = []
    for f in file:
        f1 = lambda a: wopwop.extract_wopwop_quant(case_directory=a, prefix = os.path.splitext(f)[0])
        functions.append(f1)
    wopwop.apply_to_namelist(functions, cases_directory=pred_dir, cases='cases.nam')

#   imports reformatted data from wopwop in a dictionary
pred = wopwop.import_from_namelist(file, cases_directory=pred_dir, cases='cases.nam')

#%%
#   imports several performance quantities from the MainDict.h5 file.
with h5py.File(os.path.join(pred_dir, 'MainDict.h5'), "r") as f:
    R = f[list(f.keys())[0]]['geomParams']['R'][()]
    T = f[list(f.keys())[1]]['T'][()]
    omega = f[list(f.keys())[1]]['omega'][()]

#%%

micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
exp = exp * micR / micR[4]

#%%
# Tip-Mach number
M = np.array(omega)/60*2*np.pi**R/340

#   number of observers
nObservers = np.shape(list(pred[list(pred.keys())[0]]['pressure'].values())[1])[1]
#   number of time series data sets (3)
ndata = np.shape(list(pred[list(pred.keys())[0]]['pressure'].values())[1])[3]

#   flips the data sets so that they corresponds to descending elevation angle
pred[list(pred.keys())[0]]['pressure']['geometry_values'] = np.flip(pred[list(pred.keys())[0]]['pressure']['geometry_values'],axis = 1)
pred[list(pred.keys())[0]]['pressure']['function_values'] = np.flip(pred[list(pred.keys())[0]]['pressure']['function_values'],axis = 1)
#   computes the OASPL (dB) for thickness, loading, and total noise for each observer, which is added as an additional dictionary entree
pred[list(pred.keys())[0]]['OASPL'] = 10 * np.log10(np.mean((pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:]-np.expand_dims(np.mean(pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:],axis = 2),axis = 2)) ** 2, axis=2) / 20e-6 ** 2)

# #    replaces 'nan' values with zeros
# pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][np.where(np.isnan(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values']))] = 0
# # flips dataset
# pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'] = np.flip(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'],axis = 1)
#
# # sums the SPL (dB) of the broadband noise sources, which is added as an additional dictionary entree
# pred[list(pred.keys())[0]]['BPM_total_dB'] =20*np.log10(np.sum(10**(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,:,:,4:-1]/20),axis = 3))
# # computes the OASPL (dB) of the total broadband noise across all predicted frequency bands, which is added as an additional dictionary entree
# pred[list(pred.keys())[0]]['BPM_OASPL_dB'] =  20*np.log10(np.sum(10**(pred[list(pred.keys())[0]]['BPM_total_dB']/20),axis = 2))

#%%
# computes the observer position
coord = np.squeeze(pred[list(pred.keys())[0]]['pressure']['geometry_values'][:, :, 0, :])

phi = np.arctan2(coord[:, 2],coord[:, 1]) * 180 / np.pi
azi = np.arctan2(coord[:, 1],coord[:, 0]) * 180 / np.pi

OASPL_pred = 10 * np.log10(np.mean((pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:] - np.expand_dims(np.mean(pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:], axis = 2), axis = 2)) ** 2, axis=2) / 20e-6 ** 2)
OASPL_exp = 10 * np.log10(np.mean(exp[int(fs * start_t):int(fs * end_t)] ** 2, axis = 0) / 20e-6 ** 2)

f,spl = PSD(pred[list(pred.keys())[0]]['pressure']['function_values'])

#%% Plots predicted pressure time series

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(np.shape(spl)[-1]):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0], pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, src + 1], linestyle = linestyle[src])
        else:
            ax.plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0], pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, src + 1], linestyle =  linestyle[src])

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

#%% Plots relative contributions of broadband noise sources

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig, ax = plt.subplots(len(mics), 1, figsize=(8, 6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace=0.35, bottom=.1,right = 0.725,left = .075)

for ii, m in enumerate(mics):

    for src in range(np.shape(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'])[-1]-1):
        if len(mics) > 1:
            ax[ii].plot(np.squeeze(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,0]), np.squeeze(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,src+1]),marker = 'o')
        else:
            ax.plot(np.squeeze(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,0]), np.squeeze(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,src+1]),marker = 'o')

    if len(mics) > 1:
        ax[ii].grid('on')
        ax[ii].axis(axis_lim)
        ax[ii].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m - 1])}^\circ)$')
        if ii != len(mics) - 1:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[len(mics) - 1].set_xlabel('Frequency [Hz]')
        ax[int((len(mics) - 1) / 2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
        ax[int((len(mics) - 1) / 2)].legend(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['names'][1:], loc='center', ncol=1, bbox_to_anchor=(1.21, 0.5))

    else:
        ax.grid('on')
        ax.axis(axis_lim)
        ax.set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m - 1])}^\circ)$')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
        ax.legend(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['names'][1:], loc='center', ncol=1, bbox_to_anchor=(1.21, 0.5))

#%%

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(np.shape(spl)[-1]):
        if src == 0:
            #   Plots the resulting spectra in dB
            if len(mics) > 1:
                ax[i].stem(np.squeeze(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,0]), np.squeeze(pred[list(pred.keys())[0]]['BPM_total_dB'][:,m-1,:]), linefmt=f'C{3}{linestyle[src]}', markerfmt=f'C{3}o', basefmt=f'C{3}')
            else:
                ax.stem(np.squeeze(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,0]), np.squeeze(pred[list(pred.keys())[0]]['BPM_total_dB'][:,m-1,:]), linefmt=f'C{3}{linestyle[src]}', markerfmt=f'C{3}o',basefmt=f'C{3}')

        #   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].stem(f, spl[m-1,:, src], linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')
        else:
            ax.stem(f, spl[m-1, :, src], linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')

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
    ax[len(mics) - 1].legend(['Broadband','Thickness','Loading', 'Total'],loc='center',ncol = 4, bbox_to_anchor=(.5, -.625))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
    # ax.set_xscale('log')
    ax.axis(axis_lim)
    ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax.grid('on')
    ax.legend(['Broadband','Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.175))

#%% Plots predicted directivities of tonal noise sources

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
for src in range(np.shape(spl)[-1]):
    ax.plot(phi * np.pi / 180, np.squeeze(OASPL_pred[0, :, src]), linestyle = linestyle[src])
ax.set_thetamax(phi[0]+2)
ax.set_thetamin(phi[-1]-2)
ax.set_ylim([20,60])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.15),  labelpad = -100, rotation = phi[-1]-5)
ax.legend(['Thickness', 'Loading', 'Total'], ncol=1,loc='center',bbox_to_anchor=(.25, 0.9))

#%% Overlays the predicted and measured spectra

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
#   Computes the mean-square PSD spectra for each mic
    f_exp,Xm_exp,Sxx_exp,Gxx_exp,Gxx_avg_exp = fun.msPSD(exp[int(fs * start_t):int(fs * end_t), m - 1], fs, df = df_exp, ovr=0.75, plot = False, save_fig = False)
#   Plots the resulting spectra in dB
    if len(mics)>1:
        ax[i].plot(f_exp,10*np.log10(Gxx_avg_exp*df_exp/20e-6**2),linestyle=linestyle[1])
        ax[i].stem(f,spl[m-1,:,-1],linefmt =f'C{1}{linestyle[0]}', markerfmt =f'C{1}o',basefmt=f'C{1}')
        # ax[i].plot(np.squeeze(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:, m - 1, :, 0]),np.squeeze(pred[list(pred.keys())[0]]['BPM_total_dB'][:, m - 1, :]))


    else:
        ax.plot(f,10*np.log10(Gxx_avg_exp*df_exp/20e-6**2),linestyle=linestyle[1])
        ax.stem(f, spl[m-1, :, -1],linefmt =f'C{1}{linestyle[0]}', markerfmt =f'C{1}o',basefmt=f'C{1}')
        ax.plot(np.squeeze(pred[list(pred.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:, m - 1, :, 0]),np.squeeze(pred[list(pred.keys())[0]]['BPM_total_dB'][:, m - 1, :]))

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
    ax[len(mics) - 1].legend(['Measured','Predicted'],loc='center',ncol = 2, bbox_to_anchor=(.5, -.625))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
    # ax.set_xscale('log')
    ax.axis(axis_lim)
    ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax.grid('on')
    ax.legend(['Measured','Predicted'],loc='center',ncol = 2, bbox_to_anchor=(.5, -.175))

#%% Overlays the predicted and measured OASPL directivities

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
ax.plot(phi * np.pi / 180, np.squeeze(OASPL_exp[:9]), linestyle = linestyle[src])
ax.plot(phi * np.pi / 180, np.squeeze(OASPL_pred[0, :, -1]), linestyle = linestyle[src])
ax.set_thetamax(phi[0]+2)
ax.set_thetamin(phi[-1]-2)
ax.set_ylim([0,100])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.15),  labelpad = -100, rotation = phi[-1]-5)
ax.legend(['Measured','Predicted'], ncol=1,loc='center',bbox_to_anchor=(.25, 0.9))
