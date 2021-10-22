import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
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
dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b69/'

#   raw wopwop output file names
file = ['pressure.h5']

# set equal to true to reformat the raw data from wopwop and write it out as an HDF5 file, this only needs to be set
# to true if this is your first time working with the predicted data.
save_h5= False

save_fig = False
#   legend labels
leglab = ['Measured','Predicted']
# leglab=''

#   Linestyle for each case
linestyle =['--','-.','-','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [5,16,-4]

#   Frequency resolution of spectra [Hz]
df_exp = 2
#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 60]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 15

#%%
lowson_dat = {}
#   imports several performance quantities from the MainDict.h5 file.
with h5py.File(os.path.join(dir,os.path.basename(os.path.dirname(dir))+'_form1a_LN_sdata.h5'),'r') as f:
    for k, dat in f.items():
        lowson_dat = {**lowson_dat,**{k:dat[()]}}

#%%

#   imports reformatted data from wopwop in a dictionary
wopwop_dat = wopwop.import_from_namelist(file, cases_directory=dir, cases='cases.nam')

#%%
#   imports several performance quantities from the MainDict.h5 file.
with h5py.File(os.path.join(dir, 'MainDict.h5'), "r") as f:
    R = f[list(f.keys())[0]]['geomParams']['R'][()]
    th = f[list(f.keys())[0]]['loadParams']['th'][()]
    T = f[list(f.keys())[1]]['T'][()]
    omega = f[list(f.keys())[1]]['omega'][()]

#%%

# Tip-Mach number
M = np.array(omega)/60*2*np.pi**R/340

#   number of observers
nObservers = np.shape(list(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure'].values())[1])[1]
#   number of time series data sets (3)
ndata = np.shape(list(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure'].values())[1])[3]

#   flips the data sets so that they corresponds to descending elevation angle
wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['geometry_values'] = np.flip(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['geometry_values'], axis = 1)
wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'] = np.flip(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'], axis = 1)
#   computes the OASPL (dB) for thickness, loading, and total noise for each observer, which is added as an additional dictionary entree
wopwop_dat[list(wopwop_dat.keys())[0]]['OASPL'] = 10 * np.log10(np.mean((wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'][:, :, :, 1:] - np.expand_dims(np.mean(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'][:, :, :, 1:], axis = 2), axis = 2)) ** 2, axis=2) / 20e-6 ** 2)

#%%
# computes the observer position
coord = np.squeeze(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['geometry_values'][:, :, 0, :])

phi = np.arctan2(coord[:, 2],coord[:, 1]) * 180 / np.pi
azi = np.arctan2(coord[:, 1],coord[:, 0]) * 180 / np.pi

OASPL_pred = 10 * np.log10(np.mean((wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'][:, :, :, 1:] - np.expand_dims(np.mean(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'][:, :, :, 1:], axis = 2), axis = 2)) ** 2, axis=2) / 20e-6 ** 2)

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(lowson_dat['ts'][:-1]/(lowson_dat['dt']*360), lowson_dat['p_total'][m-1])
            ax[i].plot(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1, wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'][0, m - 1, :,  2])

        else:
            ax.plot(lowson_dat['ts'][:-1]/(lowson_dat['dt']*360), lowson_dat['p_total'][m-1])
            ax.plot(wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1, wopwop_dat[list(wopwop_dat.keys())[0]]['pressure']['function_values'][0, m - 1, :,  2])

for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xlim([0,1])
    # ax[i].set_ylim([-0.015,0.015])
    ax[i].grid('on')

ax[int(len(mics)/2)].set_ylabel('Pressure [Pa]')
ax[- 1].set_xlabel('Roatation')
ax[-1].legend(["Form1A",'WOPWOP'], ncol=2,loc='center',bbox_to_anchor=(0.5, -0.55))


#%%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
ax.plot(lowson_dat['phi'], lowson_dat['OASPL_tot'])
ax.plot(phi*np.pi/180, np.squeeze(OASPL_pred)[:,1])
ax.set_thetamax(phi[0]+2)
ax.set_thetamin(phi[-1]-2)
ax.set_ylim([0,70])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.25),  labelpad = -20, rotation = phi[-1]-3)
ax.legend(["Form1A",'WOPWOP'], ncol=1,loc='center',bbox_to_anchor=(-0.05, 0.9))
