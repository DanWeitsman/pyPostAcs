
import os
print(f'print: {os.getcwd()}')
import h5py
import numpy as np
import sys
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.fft import fft
sys.path.insert(0, '/Users/danielweitsman/codes/github/OpenWopWop/pyWopwop')
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
    N = np.max(np.shape(data))
    # sampling rate [Hz]
    fs_pred = N / data[0, 0, -1, 0]
    # temporal resolution [s]
    dt = fs_pred ** -1
    # frequency resolution
    df = (dt * N) ** -1
    # frequency vector [Hz]
    f = np.arange(int(N/2)+1) * df
    # linear spectrum [Pa]
    Xm = fft(np.squeeze(data[:,:,:,1:]), axis=1) * dt
    # single sided power spectral density [Pa^2/Hz]
    Sxx = (dt * N) ** -1 * abs(Xm) ** 2
    Gxx = Sxx[:,:int(N / 2)+1]
    Gxx[:,1:-1] = 2 * Gxx[:,1:-1]
    # converts to single-sided PSD to SPL [dB]
    spl = 10 * np.log10(Gxx * df / 20e-6 ** 2)

    return f, spl


#%%
pred_dir = '/Users/danielweitsman/Documents/research/prediction/ols/3022_wake_15R/'
save_dir = '/Users/danielweitsman/Documents/research/BVI_helmholtz'
#   raw wopwop output file names
file = ['pressure.h5']

# set equal to true to reformat the raw data from wopwop and write it out as an HDF5 file, this only needs to be set
# to true if this is your first time working with the predicted data.
h5_write = False


#   Linestyle for each case
linestyle =['-','-.','--','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,3,4]

#   Frequency resolution of spectra [Hz]
df = 2
#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 70]



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
    Nb = f[list(f.keys())[1]]['Nb'][()]


#%%

# micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
# exp = exp * micR / micR[4]

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

#%%

fs = np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values']).shape[1]/np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values'])[0,-1,0]

# f,Xm,Sxx,Gxx,Gxx_avg = fun.msPSD(np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values'])[:,:,-1].transpose(), fs = fs, df = df, win = False, ovr = 0, f_lim =[10,5e3], levels = [0,100],save_fig = False, save_path = '',plot = False)

#%%
# computes the observer position
coord = np.squeeze(pred[list(pred.keys())[0]]['pressure']['geometry_values'][:, :, 0, :])

phi = np.arctan2(coord[:, 2],coord[:, 0]) * 180 / np.pi
azi = np.arctan2(coord[:, 1],coord[:, 0]) * 180 / np.pi

OASPL_pred = 10 * np.log10(np.mean((pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:] - np.expand_dims(np.mean(pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1:], axis = 2), axis = 2)) ** 2, axis=2) / 20e-6 ** 2)

f,spl = PSD(pred[list(pred.keys())[0]]['pressure']['function_values'])
# Xm,xn_filt = fun.ffilter(pred[list(pred.keys())[0]]['pressure']['function_values'][0,0,:,-1],fs, btype='bp', fc = [omega/60*Nb*10,omega/60*Nb*49],filt_shaft_harm=True,Nb=2)
BVI_OASPL = 10*np.log10(np.sum(10**(spl[:,::2,-1][:,10:50]/10),axis = 1))




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
            ax[i].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1, pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, src + 1], linestyle = linestyle[src])
        else:
            ax.plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1, pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, src + 1], linestyle =  linestyle[src])
#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for src, m in enumerate(mics):
        ax[src].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
        if src!=len(mics)-1:
            ax[src].tick_params(axis='x', labelsize=0)
        # ax[ii].set_xscale('log')
        # ax[ii].set_yticks(np.arange(0,axis_lim[-1],20))
        ax[src].set_xlim([0,1])
        # ax[ii].axis(axis_lim)
        ax[src].grid('on')

    ax[len(mics) - 1].set_xlabel('Rotation')
    ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')

    ax[len(mics) - 1].legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))
plt.savefig(os.path.join(save_dir, f'rel_tseries.png'), format='png')


#%%
width = .25
hatch = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):

    ax[i].bar(f[::2]/ (omega /60 * Nb)+width*0.5-1.5*width, spl[m-1,::2,0],width = width,hatch =hatch[0]*2,align='center')
    ax[i].bar(f[::2]/ (omega /60 * Nb)+width*0.5-.5*width, spl[m-1,::2,1],width = width,hatch =hatch[1]*2,align='center')
    ax[i].bar(f[::2]/ (omega /60 * Nb)+width*0.5+.5*width, spl[m-1,::2,2],width = width,hatch =hatch[2]*2,align='center')

    if i!=2:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[i].set_yticks(np.arange(0, 60, 20))
    # ax[i].set_xticks(np.arange(1, 5))

    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    ax[i].axis([0, 100, 50, 110])
    ax[i].grid('on')

ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax[-1].set_xlabel('BPF Harmonic')
# ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
ax[-1].legend(['Thickness','Loading', 'Total'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))
plt.savefig(os.path.join(save_dir, f'rel_spec.png'), format='png')
plt.show()



#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    ax[i].stem(f[::2]/ (omega /60 * Nb), spl[m-1,::2,1])
    if i!=2:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[i].set_yticks(np.arange(0, 60, 20))
    # ax[i].set_xticks(np.arange(1, 5))

    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    ax[i].axis([0, 100, 50, 110])
    ax[i].grid('on')

ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax[-1].set_xlabel('BPF Harmonic')
plt.savefig(os.path.join(save_dir, f'ln_spec.png'), format='png')
plt.show()

