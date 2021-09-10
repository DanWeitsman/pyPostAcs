import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
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

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]
#   set equal to True in order to save the figure as an eps
save = False
#   set equal to true to reformat the raw data from WOPWOP and write it out as an HDF5 file.
h5_write = False

#   path to the root directory of the case
cases_directory = '/Users/danielweitsman/Desktop/Masters_Research/lynx/e2b_sweep/'

#   names of files to import
file = ['pressure.h5','octaveFilterSP/spl_octFilt_spectrum.h5']


#%%
#   generates and imports the HDF5 file containing the reformatted pressure time series and BPM spectral data.
if h5_write:
    functions = []
    for f in file:
        f1 = lambda a: wopwop.extract_wopwop_quant(case_directory=a, prefix = os.path.splitext(f)[0])
        functions.append(f1)
    wopwop.apply_to_namelist(functions, cases_directory=cases_directory, cases='cases.nam')

output = wopwop.import_from_namelist(file, cases_directory=cases_directory, cases='cases.nam')

#%%
#   imports several performance quantities from the MainDict.h5 file.
with h5py.File(os.path.join(cases_directory, 'MainDict.h5'), "r") as f:
    R = f[list(f.keys())[0]]['geomParams']['R'][()]
    T = f[list(f.keys())[1]]['T'][()]
    omega = f[list(f.keys())[1]]['omega'][()]

#%%
# Tip-Mach number
M = np.array(omega)/60*2*np.pi**R/340

#   number of observers
nObservers = np.shape(list(output[list(output.keys())[0]]['pressure'].values())[1])[1]
#   number of time series data sets (3)
ndata = np.shape(list(output[list(output.keys())[0]]['pressure'].values())[1])[3]

for k,data in output.items():
    #   flips the data sets so that they corresponds to descending elevation angle
    data['pressure']['geometry_values'] = np.flip(data['pressure']['geometry_values'],axis = 1)
    data['pressure']['function_values'] = np.flip(data['pressure']['function_values'],axis = 1)
    #   computes the OASPL (dB) for thickness, loading, and total noise for each observer, which is added as an additional dictionary entree
    output[k]['OASPL'] = 10 * np.log10(np.mean((data['pressure']['function_values'][:, :, :, 1:]-np.expand_dims(np.mean(data['pressure']['function_values'][:, :, :, 1:],axis = 2),axis = 2)) ** 2, axis=2) / 20e-6 ** 2)

    #    replaces 'nan' values with zeros
    data['octaveFilterSP/spl_octFilt_spectrum']['function_values'][np.where(np.isnan(data['octaveFilterSP/spl_octFilt_spectrum']['function_values']))] = 0
    # flips dataset
    data['octaveFilterSP/spl_octFilt_spectrum']['function_values'] = np.flip(data['octaveFilterSP/spl_octFilt_spectrum']['function_values'],axis = 1)
    # computes the OASPL (dB) of the broadband noise sources, which is added as an additional dictionary entree
    output[k]['BPM_total_dB'] =  20*np.log10(np.sum(np.sum(10**(data['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,:,:,4:-1]/20),axis = 2),axis = 2))

# computes the observer position
coord = np.squeeze(output[list(output.keys())[0]]['pressure']['geometry_values'][:, :, 0, :])
phi = np.arctan2(coord[:, 2],coord[:, 1]) * 180 / np.pi
azi = np.arctan2(coord[:, 1],coord[:, 0]) * 180 / np.pi

#%% Plots the broadband noise spectra for each case

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [100, 5e3, 0, 80]

for i,v in enumerate(output.items()):

    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig, ax = plt.subplots(len(mics), 1, figsize=(8, 6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace=0.35, bottom=.1,right = 0.725,left = .075)

    for ii, m in enumerate(mics):
        for src in range(np.shape(output[list(output.keys())[0]]['octaveFilterSP/spl_octFilt_spectrum']['function_values'])[-1]-1):
            if len(mics) > 1:
                ax[ii].plot(np.squeeze(v[1]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,0]), np.squeeze(v[1]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,src+1]),marker = 'o')
            else:
                ax.plot(np.squeeze(v[1]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,0]), np.squeeze(v[1]['octaveFilterSP/spl_octFilt_spectrum']['function_values'][:,m-1,:,src+1]),marker = 'o')

        if len(mics) > 1:
            ax[ii].grid('on')
            ax[ii].axis(axis_lim)
            ax[ii].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m - 1])}^\circ)$')
            if ii != len(mics) - 1:
                ax[ii].tick_params(axis='x', labelsize=0)
            ax[len(mics) - 1].set_xlabel('Frequency [Hz]')
            ax[int((len(mics) - 1) / 2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
            ax[int((len(mics) - 1) / 2)].legend(v[1]['octaveFilterSP/spl_octFilt_spectrum']['names'][1:], loc='center', ncol=1, bbox_to_anchor=(1.21, 0.5))

        else:
            ax.grid('on')
            ax.axis(axis_lim)
            ax.set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m - 1])}^\circ)$')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
            ax.legend(v[1]['octaveFilterSP/spl_octFilt_spectrum']['names'][1:], loc='center', ncol=1, bbox_to_anchor=(1.21, 0.5))

#%% Plots the OASPL of each noise source as a function of the tip Mach number (M)

c  = list(mcolors.TABLEAU_COLORS.keys())[:ndata+1]

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

for i,v in enumerate(output.items()):
    for ii, m in enumerate(mics):
        ax[ii].plot(M[i], v[1]['BPM_total_dB'][:, m - 1], marker='o', c=c[-1])
        for src in range(ndata):
            if len(mics) > 1:
                ax[ii].plot(M[i], v[1]['OASPL'][:,m-1,src],marker = 'o',c = c[src])
            else:
                ax.plot(M[i], v[1]['OASPL'][:,m-1,src],marker = 'o',c = c[src])

        if len(mics) > 1:
            ax[ii].grid('on')
            ax[ii].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m - 1])}^\circ)$')
            if ii != len(mics)-1:
                ax[ii].tick_params(axis='x', labelsize=0)
            ax[len(mics) - 1].set_xlabel('$M_{Tip}$')
            ax[int((len(mics) - 1) / 2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
            ax[len(mics) - 1].legend(['Broadband','Thickness', 'Loading', 'Total'], loc='center', ncol=4, bbox_to_anchor=(.5, -.625))
        else:
            ax.set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m - 1])}^\circ)$')
            ax.plot(M[i], v[1]['OASPL'][:, m - 1, src], marker='o', c=c[src])
            ax.set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(phi[mics[0]-1])}^\circ)$')
            ax.set_xlabel('$M_{Tip}$')
            ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
            ax.grid('on')
            ax.legend(['Broadband','Thickness','Loading', 'Total'],loc='center',ncol = 4, bbox_to_anchor=(.5, -.175))

if save:
    if not os.path.exists(os.path.join(os.path.dirname(cases_directory), 'Figures')):
        os.mkdir(os.path.join(os.path.dirname(cases_directory), 'Figures'))
    plt.savefig(os.path.join(cases_directory, 'Figures', 'OASPL_sweep.eps'), format='eps')
    plt.savefig(os.path.join(cases_directory, 'Figures','OASPL_sweep.png'), format='png')

