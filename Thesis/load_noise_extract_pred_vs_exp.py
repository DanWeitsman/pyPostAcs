import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft
import sys
from bisect import bisect
sys.path.insert(0, '/Users/danielweitsman/Desktop/Masters_Research/py scripts/WOPWOP_PostProcess/pyWopwop')
import wopwop
#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})


#%%
#   directory containing the experimental data file.
exp_dir ='/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/h2b69/'
#   directory containing the prediction data file.
pred_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b69/'

#   raw wopwop output file names
file = ['pressure.h5']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,9]

# min and max shaft order harmonics to exclude from the analysis (set max to -1 to include all upper order harmonics)
harm_filt = [2, 4]

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 60]

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

#%%
#   imports reformatted data from wopwop in a dictionary
pred = wopwop.import_from_namelist(file, cases_directory=pred_dir, cases='cases.nam')

#%%
#   imports several performance quantities from the MainDict.h5 file.
with h5py.File(os.path.join(pred_dir, 'MainDict.h5'), "r") as f:
    R = f[list(f.keys())[0]]['geomParams']['R'][()]
    th = f[list(f.keys())[0]]['loadParams']['th'][()]
    T = f[list(f.keys())[1]]['T'][()]
    omega = f[list(f.keys())[1]]['omega'][()]

#%% Spherical spreading correction

micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
exp = exp * micR / micR[4]

#%% Sync average experimental measurements

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
spl_inph = 10*np.log10(fun.SD(Xm_inph,fs1)*df/20e-6**2)

Xm_outph = Xm_avg-np.expand_dims(Xm_inph,axis =1)
spl_outph = 10*np.log10(fun.SD(Xm_outph,fs1)*df/20e-6**2)

#%%
#   number of observers
nObservers = np.shape(list(pred[list(pred.keys())[0]]['pressure'].values())[1])[1]
#   number of time series data sets (3)
ndata = np.shape(list(pred[list(pred.keys())[0]]['pressure'].values())[1])[3]

#   flips the data sets so that they corresponds to descending elevation angle
pred[list(pred.keys())[0]]['pressure']['geometry_values'] = np.flip(pred[list(pred.keys())[0]]['pressure']['geometry_values'],axis = 1)
pred[list(pred.keys())[0]]['pressure']['function_values'] = np.flip(pred[list(pred.keys())[0]]['pressure']['function_values'],axis = 1)
#   computes the OASPL (dB) for thickness, loading, and total noise for each observer, which is added as an additional dictionary entree
pred[list(pred.keys())[0]]['OASPL'] = 10 * np.log10(np.mean((pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:]-np.expand_dims(np.mean(pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:],axis = 2),axis = 2)) ** 2, axis=2) / 20e-6 ** 2)

#%%
# computes the observer position
coord = np.squeeze(pred[list(pred.keys())[0]]['pressure']['geometry_values'][:, :, 0, :])

phi = np.arctan2(coord[:, 2],coord[:, 1]) * 180 / np.pi
azi = np.arctan2(coord[:, 1],coord[:, 0]) * 180 / np.pi

OASPL_pred = 10 * np.log10(np.mean((pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:] - np.expand_dims(np.mean(pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:], axis = 2), axis = 2)) ** 2, axis=2) / 20e-6 ** 2)

#%%
xn_inph_pred = (pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1] - 1, :, 3] + pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0] - 1, :, 3]) / 2
xn_outph_pred = (pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1] - 1, :, 3] - pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0] - 1, :, 3]) / 2

L = np.shape(pred[list(pred.keys())[0]]['pressure']['function_values'])[2]
fs_pred = L / pred[list(pred.keys())[0]]['pressure']['function_values'][0, 0, -1, 0]
df = ((fs_pred**-1)*L)**-1

N = ((fs_pred ** -1 * df)**-1)
Nfft = np.floor(L/N)

# f_pred,Gxx_pred,spl_pred = PSD(pred[list(pred.keys())[0]]['pressure']['function_values'][:,:,-1])
f,Xm_pred,Sxx_pred,Gxx_pred,Gxx_avg_pred = fun.msPSD(np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values'][0, :, :, 3]).transpose(), fs_pred, df = df, win = False, ovr = 0, save_fig = False, plot = False)
f,Xm_inph_pred,Sxx_inph_pred,Gxx_inph_pred,Gxx_avg_inph_pred = fun.msPSD(np.squeeze(xn_inph_pred), fs_pred, df = df, win = False, ovr = 0, save_fig = False, plot = False)
f,Xm_outph_pred,Sxx_outph_pred,Gxx_outph_pred,Gxx_avg_outph_pred = fun.msPSD(np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values'][0, :, :, 3] - xn_inph_pred).transpose(), fs_pred, df = df, win = False, ovr = 0, save_fig = False, plot = False)

#%%

#   Loops through each mic
for i,m in enumerate(mics):
    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace=0.35, bottom=0.15)

    ax[0].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega / 60) ** -1,
               xn_inph_pred, linestyle='-.')

    ax[1].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega / 60) ** -1,
               pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 3] - xn_inph_pred,
               linestyle='-.')

    ax[2].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega / 60) ** -1,
               pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, -1], linestyle='-.')

    ax[0].plot(t_nondim, -xn_inph,linestyle='-')
    ax[1].plot(t_nondim, -(Xn_avg_filt[:,m-1]-xn_inph),linestyle='-')
    ax[2].plot(t_nondim, -Xn_avg_filt[:,m-1],linestyle='-')

    for ii in range(3):
        if ii!=2:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[ii].set_xlim([0,1])
        ax[ii].grid('on')

    ax[0].set_title('In-Phase')
    ax[1].set_title('Out Of Phase')
    ax[2].set_title('Total')

    ax[1].set_ylabel('Pressure [Pa]')
    plt.suptitle(f'$Mic\ {m} \ ( \phi = {round(phi[m - 1])}^\circ)$')
    ax[-1].set_xlabel('Rotation')
    ax[-1].legend(['Predicted', 'Measured'], loc='center', ncol=3,bbox_to_anchor=(.5, -.65))

#%%
    #   Loops through each mic
for i, m in enumerate(mics):
    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace=0.35, bottom=0.15)

    ax[0].stem(f/(omega/60*Nb), 10 * np.log10(Gxx_avg_inph_pred * df / 20e-6 ** 2), linefmt=f'C{0}{"-."}', markerfmt=f'C{0}o',basefmt=f'C{0}')
    ax[1].stem(f/(omega/60*Nb), 10 * np.log10(Gxx_avg_outph_pred[:,m-1] * df / 20e-6 ** 2), linefmt=f'C{0}{"-."}', markerfmt=f'C{0}o',basefmt=f'C{0}')
    ax[2].stem(f/(omega/60*Nb), 10 * np.log10(Gxx_avg_pred[:,m-1] * df / 20e-6 ** 2), linefmt=f'C{0}{"-."}', markerfmt=f'C{0}o',basefmt=f'C{0}')

    ax[0].stem(BPF_harm, spl_inph,linefmt =f'C{1}{"-"}', markerfmt =f'C{1}o',basefmt=f'C{1}')
    ax[1].stem(BPF_harm, spl_outph[:,m-1],linefmt =f'C{1}{"-"}', markerfmt =f'C{1}o',basefmt=f'C{1}')
    ax[2].stem(BPF_harm, spl[:,m-1],linefmt =f'C{1}{"-"}', markerfmt =f'C{1}o',basefmt=f'C{1}')

    for ii in range(3):
        if ii != 2:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[ii].set_xticks(np.arange(1, harm_filt[-1] / Nb + 1))
        ax[ii].set_xlim([0, harm_filt[-1] / Nb + 1])
        ax[ii].set_ylim([axis_lim[-2], axis_lim[-1]])
        ax[ii].grid('on')

    ax[0].set_title('In-Phase')
    ax[1].set_title('Out Of Phase')
    ax[2].set_title('Total')

    ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    plt.suptitle(f'$Mic\ {m} \ ( \phi = {round(phi[m - 1])}^\circ)$')
    ax[-1].set_xlabel('BPF Harmonic')
    ax[-1].legend(['Predicted', 'Measured'], loc='center', ncol=3, bbox_to_anchor=(.5, -.65))

#%%
linestyle = ['-','-.']
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig, ax = plt.subplots(len(mics), 1, figsize=(8, 6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace=0.35, bottom=0.15)

#   Loops through each mic
for i,m in enumerate(mics):

    ax[0].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega / 60) ** -1,
               pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, -1], linestyle=linestyle[i])

    ax[1].plot(t_nondim, -Xn_avg_filt[:,m-1],linestyle=linestyle[i])

    if i!=1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xlim([0,1])
    ax[i].grid('on')
    ax[i].set_ylabel('Pressure [Pa]')

ax[0].set_title('Prediction')
ax[1].set_title('Measurement')
ax[-1].set_xlabel('Rotation')
ax[-1].legend([f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0] - 1])}^\circ)$',f'$Mic\ {mics[-1]} \ ( \phi = {round(phi[mics[-1] - 1])}^\circ)$'], loc='center', ncol=3,bbox_to_anchor=(.5, -.35))