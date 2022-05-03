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
exp_dir ='/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/'
runs = ['h2b69']
# runs = ['lat_cyc_sweep/lowM/h2b/h2b15','lat_cyc_sweep/lowM/h2b/h2b25','lat_cyc_sweep/lowM/h2b/h2b36','long_cyc_sweep/lowM/h2b/h2b48','h2b69']
#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]

# min and max shaft order harmonics to exclude from the analysis (set max to -1 to include all upper order harmonics)
harm_filt = [2,6]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t =15

save_fig = True
linestyle =['--','-.','-','-',':','--']

#%% imports experimental data
stored_data = {}

for i,run in enumerate(runs):
    # configures save directory
    save_dir = os.path.join(exp_dir,run,'figures')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(exp_dir,'figures')):
        os.mkdir(os.path.join(exp_dir,'figures'))

    with h5py.File(os.path.join(exp_dir,run, 'acs_data.h5'), 'r') as dat_file:
        exp = dat_file['Acoustic Data'][:].transpose() / (dat_file['Sensitivities'][:] * 1e-3)
        fs_exp = dat_file['Sampling Rate'][()]
        ttl = dat_file['Motor1 RPM'][()]
        fs_ttl = round((np.mean(np.diff(dat_file['Time (s)']))) ** -1)

#%% Spherical spreading correction

# micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
# exp = exp * micR / micR[4]
#%%

    Nb = 2
    t = np.arange(len(ttl)) / fs_ttl
    t_acs = np.arange(len(exp)) / fs_exp
    rpm = (np.diff(np.squeeze(np.where(np.diff(ttl) == 1))) / fs_ttl / 60) ** -1

    LE_ind, lim_ind, rpm_nom, u_rpm = fun.rpm_eval(ttl,fs_ttl,start_t,end_t)

    tac_ind = list(map(lambda x: bisect(t_acs,x),t[LE_ind[lim_ind[0]:lim_ind[1]]]))
    tac_rpm = rpm[lim_ind[0]:lim_ind[1]]
    t_rpm = t[LE_ind]


    # fit_ind = [bisect(t_rpm,6.47),bisect(t_rpm,6.7)]
    # rpm_fit =np.poly1d(np.polyfit(t_rpm[fit_ind[0]:fit_ind[1]],rpm[fit_ind[0]:fit_ind[1]],1))
    # accel = rpm_fit[1]/60*np.pi*2

    rev_skip = 0
    rev_skip = rev_skip + 1
    dtac_ind = np.diff(tac_ind[::rev_skip])
    N = np.max(dtac_ind)
    N_avg = np.mean(dtac_ind)
    fs1 = N / N_avg * fs_exp
    dt = fs1 ** -1
    df = (N * dt) ** -1

    xn_rev = [exp[tac_ind[i]:tac_ind[i + 1]] for i in range(len(tac_ind[::rev_skip]) - 1)]
    xn_rev_up,Xm_rev_up = np.array(list(map(lambda x: fun.upsample(x, fs_exp, N), xn_rev))).transpose(1,0,2,3)
    xn_rev_up_filt = np.array([fun.ffilter(n,fs_exp, btype='bp', fc = [tac_rpm[i]/60*harm_filt[0],tac_rpm[i]/60*harm_filt[-1]], filt_shaft_harm=True,Nb=2)[1] for i,n in enumerate(xn_rev_up)])
    xn_rev_up_filt_avg = np.mean(xn_rev_up_filt,axis = 0)
    CI = np.std(xn_rev_up_filt, axis=0)
    t = np.arange(N)*fs1**-1

    store = {'tac_ind':tac_ind,'tac_rpm':tac_rpm,'t':t,'rpm_nom':rpm_nom,'fs1':fs1,'dt':dt,'df':df,'xn_rev_up_filt':xn_rev_up_filt,'xn_rev_up_filt_avg':xn_rev_up_filt_avg,'CI':CI}
    stored_data = {**stored_data,**{os.path.basename(run):store}}

#%%

    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace = 0.35)
    #   Loops through each mic
    for i,m in enumerate(mics):
        ax[i].plot(t/(rpm_nom/60)**-1, xn_rev_up_filt_avg[:, m - 1],c ='r',lw = 2)
        ax[i].fill_between(t/(rpm_nom/60)**-1, (xn_rev_up_filt_avg - 2*CI)[:, m - 1], (xn_rev_up_filt_avg + 2*CI)[:, m - 1],color = 'lightgrey')

        ax[i].set_title(f'M{m}')
        if i!=len(mics)-1:
            ax[i].tick_params(axis='x', labelsize=0)
        ax[i].set_xlim([0,1])
        ax[i].grid('on')
    ax[-1].set_xlabel('Rotation')
    ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')

    if save_fig:
        plt.savefig(os.path.join(save_dir, f'avg_tseries.eps'), format='eps')
        plt.savefig(os.path.join(save_dir, f'avg_tseries.png'), format='png')

    #%%
    for m in range(np.shape(xn_rev_up_filt_avg)[-1]):
        fig,ax = plt.subplots(1,1,figsize = (8,6))
        #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
        plt.subplots_adjust(hspace = 0.35)
        ax.plot(t/(rpm_nom/60)**-1, xn_rev_up_filt_avg[:, m],c ='r',lw = 2)
        ax.fill_between(t/(rpm_nom/60)**-1, (xn_rev_up_filt_avg - 2*CI)[:, m], (xn_rev_up_filt_avg + 2*CI)[:, m])
        ax.set_title(f'M{m+1}')
        ax.set_xlim([0,1])
        ax.grid('on')
        ax.set_xlabel('Rotation')
        ax.set_ylabel('Pressure [Pa]')

        if save_fig:
            plt.savefig(os.path.join(save_dir, f'avg_tseries_{m+1}.eps'),format='eps')
            plt.savefig(os.path.join(save_dir, f'avg_tseries_{m+1}.png'),format='png')
plt.close('all')

#%%


fig, ax = plt.subplots(3, 1, figsize=(8, 6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace=0.35)
for run_itr, run in enumerate(list(stored_data.keys())):
    for i, m in enumerate(mics):
        ax[i].plot(stored_data[run]['t'] / (stored_data[run]['rpm_nom'] / 60) ** -1,stored_data[run]['xn_rev_up_filt_avg'][:, m], linestyle = linestyle[run_itr], lw=2)

        ax[i].set_title(f'M{m}')
        if i != len(mics) - 1:
            ax[i].tick_params(axis='x', labelsize=0)
        ax[i].set_xlim([0, 1])
        ax[i].grid('on')
    ax[-1].set_xlabel('Rotation')
    ax[int((len(mics) - 1) / 2)].set_ylabel('Pressure [Pa]')
    ax[-1].legend(list(stored_data.keys()), ncol=2, loc='center', bbox_to_anchor=(0.5, -0.175))

if save_fig:
    plt.savefig(os.path.join(os.path.join(exp_dir,'figures'), f'avg_tseries_compare.eps'), format='eps')
    plt.savefig(os.path.join(os.path.join(exp_dir,'figures'), f'avg_tseries_compare.png'), format='png')

#%%

# #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# plt.subplots_adjust(hspace = 0.35)
#
# #   Loops through each mic
# for i,m in enumerate(mics):
#     #   Plots the resulting spectra in dB
#     if len(mics)>1:
#         ax[i].plot( xn_rev_up_filt[:, m-1],c = 'gray',lw = .5)
#
#     else:
#         ax.plot( xn_rev_up_filt[:, m-1],c = 'black',lw = .25,alpha = .5)
#
# #   Configures axes, plot, and legend if several mics are plotted
# if len(mics)>1:
#     for ii, m in enumerate(mics):
#         ax[ii].set_title('Mic: '+str(m))
#         if ii!=len(mics)-1:
#             ax[ii].tick_params(axis='x', labelsize=0)
#         # ax[ii].set_xlim([0,1])
#         # ax[ii].set_ylim([-.02,0.02])
#         ax[ii].grid('on')
#     ax[len(mics) - 1].set_xlabel('Revolution')
#     ax[int((len(mics) - 1)/2)].set_ylabel('Pressure [Pa]')
#
# #   Configures axes, plot, and legend if only a single mic is plotted
# else:
#     ax.set_title('Mic: ' + str(mics[0]))
#     ax.set_xlim(0,1)
#     # ax.set_ylim([-.02, 0.02])
#     # ax.set_yticks(np.arange(0, axis_lim[-1], 20))
#     ax.set_xlabel('Revolution')
#     ax.set_ylabel('Pressure [Pa]')
#     ax.grid('on')

#%%

# #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(1,1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# plt.subplots_adjust(hspace = 0.35)
#
# ax.plot(t_rpm[:-1],rpm)
# ax.plot(t_rpm[fit_ind[0]:fit_ind[1]],rpm_fit(t_rpm[fit_ind[0]:fit_ind[1]]))
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('RPM')
# ax.axis([6,14,0,rpm_nom+250])
# ax.grid('on')
#
# bisect(t[LE_ind],6.5)