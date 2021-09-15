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

pred_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b69/'

#   raw wopwop output file names
file = ['pressure.h5']

# set equal to true to reformat the raw data from wopwop and write it out as an HDF5 file, this only needs to be set
# to true if this is your first time working with the predicted data.
save_h5= True

save_fig = True
#   legend labels
leglab = ['Measured','Predicted']
# leglab=''

#   Linestyle for each case
linestyle =['--','-.','-','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,9]

#   Frequency resolution of spectra [Hz]
df_exp = 2
#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 60]

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
p_inph = (pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1] - 1, :, 3] + pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0] - 1, :, 3]) / 2
p_outph = (pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1] - 1, :, 3] - pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0] - 1, :, 3]) / 2

#%%
Nb = 2
L = np.shape(pred[list(pred.keys())[0]]['pressure']['function_values'])[2]
fs_pred = L / pred[list(pred.keys())[0]]['pressure']['function_values'][0, 0, -1, 0]
df = ((fs_pred**-1)*L)**-1

N = ((fs_pred ** -1 * df)**-1)
Nfft = np.floor(L/N)

pred_spec = list(map(lambda i: fun.msPSD(np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, i]).transpose(), fs_pred, df = df, win = False, ovr = 0,save_fig = False, plot = False),np.arange(1,4)))
f_pred = pred_spec[0][0]
Gxx_avg_pred = np.array([pred_spec[i][-1] for i in range(len(pred_spec))]).transpose()

f, Xm_inph, Sxx, Gxx, Gxx_avg_inph = fun.msPSD(np.squeeze(p_inph).transpose(), fs_pred, df = df, win = False, ovr = 0, save_fig = False, plot = False)
f, Xm, Sxx, Gxx, Gxx_avg_inph_load = fun.msPSD(np.squeeze(p_inph-pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 1]).transpose(), fs_pred, df = df, win = False, ovr = 0, save_fig = False, plot = False)
f, Xm_outph, Sxx, Gxx, Gxx_avg_outph = fun.msPSD(np.squeeze(pred[list(pred.keys())[0]]['pressure']['function_values'][:, :, :, 3] - p_inph).transpose(), fs_pred, df = df, win = False, ovr = 0, save_fig = False, plot = False)

#%%
c  = list(mcolors.TABLEAU_COLORS.keys())[:10]
#%% Plots predicted pressure time series

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(ndata):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1, pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, src + 1], linestyle = linestyle[src])
        else:
            ax.plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0]/(omega/60)**-1, pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, src + 1], linestyle =  linestyle[src])

ax[0].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega/60) ** -1, p_inph, linestyle =':')
ax[0].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega/60) ** -1,pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0]-1, :, 3] - p_inph, linestyle =':')
# ax[0].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega/60) ** -1,(p_inph-pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[0]-1, :, 1]-p_outph),linestyle = '-')

ax[1].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega/60) ** -1, p_inph, linestyle =':')
ax[1].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega/60) ** -1,pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1]-1, :, 3] - p_inph, linestyle =':')
# ax[1].plot(pred[list(pred.keys())[0]]['pressure']['function_values'][0, m - 1, :, 0] / (omega/60) ** -1,(p_inph-pred[list(pred.keys())[0]]['pressure']['function_values'][0, mics[-1]-1, :, 1]+p_outph),linestyle = '-')

for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xlim([0,1])
    ax[i].set_ylim([-0.015,0.015])
    ax[i].grid('on')
    ax[i].set_ylabel('Pressure [Pa]')

ax[- 1].set_xlabel('Roatation')
# ax[len(mics) - 1].legend(['Thickness','Loading', 'Total','Extracted Loading'],loc='center',ncol = 3, bbox_to_anchor=(.5, -.625))
ax[- 1].legend(['Thickness', 'Loading', 'Total', 'In-phase', 'Out-of-phase'], loc='center', ncol=5,
                         bbox_to_anchor=(.5, -.35))

if save_fig:
    if not os.path.exists(os.path.join(os.path.dirname(pred_dir), 'Figures')):
        os.mkdir(os.path.join(os.path.dirname(pred_dir), 'Figures'))
    plt.savefig(os.path.join(pred_dir, 'Figures',os.path.basename(os.path.dirname(pred_dir))+ '_p_tseries'+'.eps'), format='eps')
    plt.savefig(os.path.join(pred_dir, 'Figures',os.path.basename(os.path.dirname(pred_dir))+ '_p_tseries'+'.png'), format='png')

 #%% Plots predicted spectrum

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    for src in range(3):
        #   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].stem(f_pred/(omega/60*Nb), 10*np.log10(Gxx_avg_pred[m-1,:,src] * df/20e-6**2), linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')
        else:
            ax.stem(f_pred/(omega/60*Nb), 10*np.log10(Gxx_avg_pred[m-1,:,src] * df/20e-6**2), linefmt =f'C{src}{linestyle[src]}', markerfmt =f'C{src}o',basefmt=f'C{src}')

    ax[i].stem(f / (omega / 60 * Nb), 10 * np.log10(Gxx_avg_inph[:,0] * df / 20e-6 ** 2),linefmt=f'C{3}{":"}', markerfmt=f'C{3}^', basefmt=f'C{3}')
    ax[i].stem(f / (omega / 60 * Nb), 10 * np.log10(Gxx_avg_outph[:,m - 1] * df / 20e-6 ** 2),linefmt=f'C{4}{":"}', markerfmt=f'C{4}^', basefmt=f'C{4}')

for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[ii].set_xscale('log')
    ax[i].set_yticks(np.arange(0, axis_lim[-1], 20))
    ax[i].axis([0,4,axis_lim[2],axis_lim[-1]])
    ax[i].set_xticks(np.arange(1, 5))

    ax[i].grid('on')
    ax[i].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

ax[len(mics) - 1].set_xlabel('BPF Harmonic')
ax[len(mics) - 1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = 5, bbox_to_anchor=(.5, -.35))

plt.savefig(os.path.join(pred_dir, 'Figures', os.path.basename(os.path.dirname(pred_dir)) + '_rel_spec' + '.eps'),
            format='eps')
plt.savefig(os.path.join(pred_dir, 'Figures', os.path.basename(os.path.dirname(pred_dir)) + '_rel_spec' + '.png'),
            format='png')

#%%
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
ax.plot(phi * np.pi / 180, np.squeeze(OASPL_pred[0, :, :]), linestyle = linestyle[src])
# ax.plot((phi-th[0]*180/np.pi) * np.pi / 180, np.squeeze(OASPL_pred[0, :, 1]), linestyle = linestyle[src])

ax.set_thetamax(phi[0])
ax.set_thetamin(phi[-1])
ax.set_ylim([0,60])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.25),  labelpad = -20, rotation = phi[-1])
ax.legend(['Thickness','Loading','Total'], ncol=1,loc='center',bbox_to_anchor=(0, 1))

plt.savefig(os.path.join(pred_dir, 'Figures', os.path.basename(os.path.dirname(pred_dir)) + '_OASPL_direct' + '.eps'),
            format='eps')
plt.savefig(os.path.join(pred_dir, 'Figures', os.path.basename(os.path.dirname(pred_dir)) + '_OASPL_direct' + '.png'),
            format='png')

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(2,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

for i, m in enumerate(mics):
    ax[i].stem(f / (omega/60*Nb), 10 * np.log10(Gxx_avg_inph_load[:, m - 1] * df / 20e-6 ** 2), linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o', basefmt=f'C{0}')
    ax[i].stem(f / (omega/60*Nb), 10 * np.log10(Gxx_avg_outph[:, m - 1] * df / 20e-6 ** 2), linefmt =f'C{1}{linestyle[1]}', markerfmt =f'C{1}^', basefmt=f'C{1}')
    ax[i].stem(f_pred/(omega/60*Nb),10*np.log10(Gxx_avg_pred[m-1,:,1] * df/20e-6**2),linefmt =f'C{2}{linestyle[2]}', markerfmt =f'C{2}s',basefmt=f'C{2}')

    if i != len(mics) - 1:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[i].axis([0,4,axis_lim[2],axis_lim[-1]])
    ax[i].axis([0, 4, 0, 40])
    ax[i].set_xticks(np.arange(1, 5))
    ax[i].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    ax[i].grid('on')

ax[-1].set_xlabel('BPF Harmonic')
ax[-1].legend(['In-phase', 'Out-of-phase','Total'], loc='center', ncol=3, bbox_to_anchor=(.5, -.35))

plt.savefig(os.path.join(pred_dir, 'Figures', os.path.basename(os.path.dirname(pred_dir)) + '_ln_inph_vs_outph_spec' + '.eps'),
            format='eps')
plt.savefig(os.path.join(pred_dir, 'Figures', os.path.basename(os.path.dirname(pred_dir)) + '_ln_inph_vs_outph_spec' + '.png'),
            format='png')
#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(2,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

for i, m in enumerate(mics):

    ax[i].stem(f_pred/(omega/60*Nb),10*np.log10(Gxx_avg_pred[m-1,:,1] * df/20e-6**2),linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[i].stem(f / (omega/60*Nb), 10 * np.log10(Gxx_avg_outph[:, m - 1] * df / 20e-6 ** 2), linefmt =f'C{1}{linestyle[1]}', markerfmt =f'C{1}o', basefmt=f'C{1}')

    if i != len(mics) - 1:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[i].axis([0,4,axis_lim[2],axis_lim[-1]])
    ax[i].axis([0, 4, 30, 40])
    ax[i].set_xticks(np.arange(1, 5))
    ax[i].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1])}^\circ)$')
    ax[i].grid('on')

ax[-1].set_xlabel('BPF Harmonic')
ax[-1].legend(['Predicted', 'Extracted'], loc='center', ncol=2, bbox_to_anchor=(.5, -.35))

plt.savefig(os.path.join(pred_dir, 'Figures', os.path.basename(os.path.dirname(pred_dir)) + '_ln_pred_vs_extract_spec' + '.eps'),
            format='eps')
plt.savefig(os.path.join(pred_dir, 'Figures', os.path.basename(os.path.dirname(pred_dir)) + '_ln_pred_vs_extract_spec' + '.png'),
            format='png')

#%%
# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.
if save_h5:
    sdata = {'df':df,'f_pred':f_pred, 'f':f,  'Gxx_predict':Gxx_avg_pred, 'Gxx_extract':Gxx_avg_outph, 'phi':phi}

    if os.path.exists(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_sdata.h5')):
        os.remove(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_sdata.h5'))

    with h5py.File(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_sdata.h5'), 'a') as h5_f:
        for k, dat in sdata.items():
            h5_f.create_dataset(k, shape=np.shape(dat), data=dat)
