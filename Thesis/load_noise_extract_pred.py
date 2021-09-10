import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft
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

    return f,Gxx, spl

#%%

pred_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b69/'

#   raw wopwop output file names
file = ['pressure.h5']

# set equal to true to reformat the raw data from wopwop and write it out as an HDF5 file, this only needs to be set
# to true if this is your first time working with the predicted data.
h5_write = True

#   legend labels
leglab = ['Measured','Predicted']
# leglab=''

#   Linestyle for each case
linestyle =['-','-.','--','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,9]

#   Frequency resolution of spectra [Hz]
df_exp = 2
#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 55]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 15

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

#%%

micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
# exp = exp * micR / micR[4]
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

#%%
# computes the observer position
coord = np.squeeze(pred[list(pred.keys())[0]]['pressure']['geometry_values'][:, :, 0, :])

phi = np.arctan2(coord[:, 2],coord[:, 1]) * 180 / np.pi
azi = np.arctan2(coord[:, 1],coord[:, 0]) * 180 / np.pi

OASPL_pred = 10 * np.log10(np.mean((pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:] - np.expand_dims(np.mean(pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:], axis = 2), axis = 2)) ** 2, axis=2) / 20e-6 ** 2)

#%%
p_thickness = (pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1] - 1, :, 3]+pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0] - 1, :, 3])/2
p_loading = (pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1] - 1, :, 3]-pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0] - 1, :, 3])/2

#%%
L = np.shape(pred[list(pred.keys())[0]]['pressure']['function_values'])[2]
fs_pred = L / pred[list(pred.keys())[0]]['pressure']['function_values'][0, 0, -1, 0]
df = ((fs_pred**-1)*L)**-1

N = ((fs_pred ** -1 * df)**-1)
Nfft = np.floor(L/N)

f_pred,Gxx_pred,spl_pred = PSD(pred[list(pred.keys())[0]]['pressure']['function_values'])
f,Xm,Sxx,Gxx,Gxx_avg = fun.msPSD(np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0]-1, :, 3]-p_thickness), fs_pred, df = df, win = False, ovr = 0,save_fig = False, plot = False)
f,Ym,Syy,Gyy,Gyy_avg = fun.msPSD(np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values'][0,mics[-1]-1, :, 3]-p_thickness), fs_pred, df = df, win = False, ovr = 0,save_fig = False,plot = False)

#%% Plots predicted pressure time series

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.2)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(ndata):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1, pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, src + 1], linestyle = linestyle[src])
        else:
            ax.plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1, pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, src + 1], linestyle =  linestyle[src])

ax[0].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1,pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0]-1, :, 3]-p_thickness,linestyle = ':')
ax[0].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1,p_thickness,linestyle = ':')
# ax[0].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0],(p_thickness-pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0]-1, :, 1]),linestyle = '-')


ax[1].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1,pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1]-1, :, 3]-p_thickness,linestyle = ':')
ax[1].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1,p_thickness,linestyle = ':')
# ax[1].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0],(p_thickness-pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1]-1, :, 1]),linestyle = '-')

for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[ii].set_xscale('log')
    # ax[ii].set_yticks(np.arange(0,axis_lim[-1],20))
    ax[i].set_xlim([0,1])
    ax[i].grid('on')
    ax[i].set_ylabel('Pressure [Pa]')

ax[- 1].set_xlabel('Roatation')
# ax[len(mics) - 1].legend(['Thickness','Loading', 'Total','Extracted Loading'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))
ax[- 1].legend(['Thickness', 'Loading', 'Total', 'In-phase', 'Out-of-phase'], loc='center', ncol=3,
                         bbox_to_anchor=(.5, -.45))

 #%% Plots predicted spectrum

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(np.shape(spl_pred)[-1]):
        #   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].stem(f_pred, spl_pred[m-1,:, src], linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')
        else:
            ax.stem(f_pred, spl_pred[m-1, :, src], linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')

for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[ii].set_xscale('log')
    ax[i].set_yticks(np.arange(0, axis_lim[-1], 20))
    ax[i].axis(axis_lim)
    ax[i].grid('on')
    ax[i].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

ax[len(mics) - 1].set_xlabel('Frequency (Hz)')
ax[len(mics) - 1].legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.35))


#%%
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
ax.plot(phi * np.pi / 180, np.squeeze(OASPL_pred[0, :, :]), linestyle = linestyle[src])
# ax.plot((phi-th[0]*180/np.pi) * np.pi / 180, np.squeeze(OASPL_pred[0, :, 1]), linestyle = linestyle[src])

ax.set_thetamax(phi[0]+2)
ax.set_thetamin(phi[-1]-2)
ax.set_ylim([0,60])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.25),  labelpad = -15, rotation = phi[-1]-2)
ax.legend(['Thickness','Loading','Total'], ncol=1,loc='center',bbox_to_anchor=(-.1, 1))

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(2,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

ax[0].stem(f_pred,spl_pred[mics[0]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
ax[0].stem(f,10*np.log10(Gxx_avg * df/20e-6**2),linefmt =f'C{1}{linestyle[1]}', markerfmt =f'C{1}o',basefmt=f'C{1}')
ax[0].axis(axis_lim)
ax[0].tick_params(axis='x', labelsize=0)
ax[0].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax[0].set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
ax[0].grid('on')

ax[1].stem(f_pred,spl_pred[mics[-1]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
ax[1].stem(f,10*np.log10(Gyy_avg * df/20e-6**2),linefmt =f'C{1}{linestyle[1]}', markerfmt =f'C{1}o',basefmt=f'C{1}')
ax[1].axis(axis_lim)
ax[1].set_title(f'$Mic\ {mics[-1]} \ ( \phi = {round(phi[mics[-1]-1])}^\circ)$')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax[1].grid('on')
ax[1].legend(['Predicted', 'Computed'], loc='center', ncol=2, bbox_to_anchor=(.5, -.35))

