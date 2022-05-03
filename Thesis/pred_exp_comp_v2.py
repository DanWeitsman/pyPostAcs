
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
from bisect import bisect
#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 16
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

pred_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b8_rgrid'


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
axis_lim = [50, 1e3, 0, 60]

# min and max shaft order harmonics to exclude from the analysis (set max to -1 to include all upper order harmonics)
harm_filt = [2, 6]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 12


#%% imports experimental data
with h5py.File(os.path.join(exp_dir, 'acs_data.h5'), 'r') as dat_file:
    exp = dat_file['Acoustic Data'][:].transpose() / (dat_file['Sensitivities'][:] * 1e-3)
    fs_exp = dat_file['Sampling Rate'][()]
    ttl = dat_file['Motor1 RPM'][()]
    fs_ttl = round((np.mean(np.diff(dat_file['Time (s)']))) ** -1)

#%% Sync average total

Nb = 2
t = np.arange(len(ttl)) / fs_ttl
t_acs = np.arange(len(exp)) / fs_exp

LE_ind, lim_ind, rpm_nom, u_rpm = fun.rpm_eval(ttl,fs_ttl,start_t,end_t)
ind = list(map(lambda x: bisect(t_acs,x),t[LE_ind[lim_ind[0]:lim_ind[1]]]))

#%%
f_exp, Xm_exp, Sxx_exp, Gxx_exp, Gxx_avg_exp = fun.msPSD(exp[int(fs_exp * start_t):int(fs_exp * end_t), :], fs = fs_exp, df=df_exp,ovr=0.75, plot=False, save_fig=False)
f, fs1, spl, u_low, u_high, Xn_avg, Xm_avg, Xn_avg_filt, Xn_bb = fun.harm_extract(exp, tac_ind=ind, fs=fs_exp, rev_skip=0, harm_filt=harm_filt, filt_shaft_harm =  True, Nb=Nb)

xn_inph = ifft((Xm_avg[:,mics[0]-1]+Xm_avg[:,mics[-1]-1])/2)*fs1

t_nondim = np.arange(len(Xn_avg_filt))*fs1**-1/(rpm_nom/60)**-1
BPF_harm = np.arange(len(Xn_avg_filt)/2)/Nb
df_exp = (len(Xn_avg_filt)*fs1**-1)**-1

Xm_inph = (Xm_avg[:,mics[0]-1]+Xm_avg[:,mics[-1]-1])/2
Gxx_inph_exp = fun.SD(Xm_inph,fs1)

Xm_outph = Xm_avg-np.expand_dims(Xm_inph,axis =1)
Gxx_outph_exp = fun.SD(Xm_outph,fs1)

#%%

#   imports reformatted data from wopwop in a dictionary
out = {}
temp_out = {}
with h5py.File(os.path.join(pred_dir, os.path.basename(pred_dir) + '_mywopwop_out.h5'), 'r') as f:
    for dict_k in list(f.keys()):
        for k, v in f[dict_k].items():
            temp_out = {**temp_out, **{k: v[()]}}
        out = {**out, **{dict_k: temp_out}}
    LN_data = out['LN_data']
    TN_data = out['TN_data']

#%%
#   imports several performance quantities from the MainDict.h5 file.
UserIn = {}
geomParams = {}
loadParams = {}
#   imports the dictionaries saved in the MainDict.h5 file from VSP2WOPWOP.
with h5py.File(os.path.join(pred_dir, 'MainDict.h5'), "r") as f:
    for k, v in f[list(f.keys())[0]]['geomParams'].items():
        geomParams = {**geomParams, **{k: v[()]}}

    for k, v in f[list(f.keys())[0]]['loadParams'].items():
        loadParams = {**loadParams, **{k: v[()]}}

    for k, v in f[list(f.keys())[1]].items():
        UserIn = {**UserIn, **{k: v[()]}}

#%%
p_tot = LN_data['p_tot']+TN_data['p']
p_in_phase = (p_tot[mics[0] - 1] + p_tot[mics[-1] - 1]) / 2
p_out_phase = p_tot - p_in_phase

L = np.shape(p_tot)[-1]
fs = L / (loadParams['omega']/60)**-1
df_pred = ((fs ** -1) * L) ** -1
N = ((fs ** -1 * df_pred) ** -1)

f_pred, Xm_tot, Sxx_tot, Gxx_tot_pred, Gxx_avg_tot = fun.msPSD(p_tot.transpose(), fs = fs, df = df_pred, win = False, ovr = 0, save_fig = False, plot = False)
f_pred, Xm_inph, Sxx_inph, Gxx_inph_pred, Gxx_avg_inph = fun.msPSD(p_in_phase.transpose(), fs=fs, df=df_pred, win=False, ovr=0,save_fig=False, plot=False)
f_pred, Xm_outph, Sxx_outph, Gxx_outph_pred, Gxx_avg_outph = fun.msPSD(p_out_phase.transpose(), fs=fs, df=df_pred, win=False, ovr=0,save_fig=False, plot=False)

#%%
# micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
# exp = exp * micR / micR[4]

#%% Plots predicted pressure time series
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(3,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
for i, m in enumerate(mics):
    #   Loops through each mic
    # ax[0].plot(LN_data['ts'][:-1] / (LN_data['dt'] * 360), LN_data['p_tot'][m - 1])
    # ax[1].plot(TN_data['ts'][:-1] / (TN_data['dt'] * 360), TN_data['p'][m - 1])
    ax[i].plot(TN_data['ts'][:-1] / (TN_data['dt'] * 360), TN_data['p'][m - 1] + LN_data['p_tot'][m - 1])
    ax[i].plot(t_nondim, Xn_avg_filt[:,m-1],linestyle='-')
#   Configures axes, plot, and legend if several mics are plotted

    if m!=mics[-1]:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xlim([0,1])
    ax[i].ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))
    ax[i].grid('on')
    ax[i].set_title(f'$Mic {m} \ ( \phi = {round(TN_data["phi"][m - 1] * 180 / np.pi)}^\circ)$')

ax[-1].set_xlabel('Rotation')
ax[1].set_ylabel('Pressure [Pa]')
ax[-1].legend(['Predicted','Measured'],loc='center',ncol = 2, bbox_to_anchor=(.5, -.625))
plt.savefig(os.path.join(pred_dir,'figures','exp_vs_pred_tseries.png'),format = 'png')
plt.savefig(os.path.join(pred_dir,'figures','exp_vs_pred_tseries.eps'),format = 'eps')

#%%

width = .125
hatch = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
    #   Plots the resulting spectra in dB
    ax[i].bar(f_pred[::2]/ (loadParams['omega'] /60 * Nb)+width*0-.5*width, 10*np.log10(Gxx_avg_tot[::2,m-1] * df_exp/20e-6**2),width = width,hatch =hatch[1]*2,align='center')
    ax[i].bar(BPF_harm[::2][1:4]+width*1-.5*width, spl[::2, m - 1][1:4],width = width,hatch =hatch[0]*2,align='center')

    ax[i].set_title(f'$Mic {m} \ ( \phi = {round(TN_data["phi"][m - 1] * 180 / np.pi)}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    # ax[ii].set_xscale('log')
    ax[i].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[i].set_xticks(np.arange(1, 5))
    ax[i].set_yticks(np.linspace(axis_lim[2], axis_lim[-1],4))
    # ax[i].set_xlim([0,harm_filt[-1]/Nb+1])
    ax[i].grid('on')

ax[len(mics) - 1].set_xlabel('BPF Harmonic')
ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax[len(mics) - 1].legend(['Predicted', 'Measured'],loc='center',ncol = 2, bbox_to_anchor=(.5, -.625))
plt.savefig(os.path.join(pred_dir,'figures','exp_vs_pred_spec.png'),format = 'png')
plt.savefig(os.path.join(pred_dir,'figures','exp_vs_pred_spec.eps'),format = 'eps')


#%%

for m_itr, m in enumerate([mics[0],mics[-1]]):
    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig,ax = plt.subplots(3,1,figsize = (8,6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace = 0.35,bottom = 0.15,top = 0.87)

    ax[0].plot(TN_data['ts'][:-1] / (TN_data['dt'] * 360), p_in_phase)
    ax[0].plot(t_nondim, xn_inph,linestyle='-')

    ax[1].plot(TN_data['ts'][:-1] / (TN_data['dt'] * 360), p_out_phase[m - 1])
    ax[1].plot(t_nondim, Xn_avg_filt[:,m-1]-xn_inph,linestyle='-')

    ax[2].plot(TN_data['ts'][:-1] / (TN_data['dt'] * 360), TN_data['p'][m - 1] + LN_data['p_tot'][m - 1])
    ax[2].plot(t_nondim, Xn_avg_filt[:,m-1],linestyle='-')

    for i in range(3):
        if i!=2:
            ax[i].tick_params(axis='x', labelsize=0)
        ax[i].set_xlim([0, 1])
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(-3, -2))
        ax[i].grid('on')

    ax[0].set_title('In-Phase')
    ax[1].set_title('Out-of-Phase')
    ax[2].set_title('Total')
    ax[1].set_ylabel('Pressure [Pa]')
    ax[-1].set_xlabel('Rotation')
    # ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
    plt.suptitle(f'$Mic\ {m} \ ( \phi = {round(LN_data["phi"][m - 1] * 180 / np.pi)}^\circ)$')
    ax[-1].legend(['Predicted', 'Measured'], loc='center', ncol=2, bbox_to_anchor=(.5, -.625))
    plt.savefig(os.path.join(pred_dir, 'figures', f'exp_vs_pred_ph_sep_tseries_m{m}.png'), format='png')
    plt.savefig(os.path.join(pred_dir, 'figures', f'exp_vs_pred_ph_sep_tseries_m{m}.eps'), format='eps')

#%%
for m_itr, m in enumerate([mics[0],mics[-1]]):
    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig,ax = plt.subplots(3,1,figsize = (8,6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace = 0.35,bottom = 0.15,top = 0.87)

    ax[0].bar(f_pred[::2]/ (loadParams['omega'] /60 * Nb)+width*0-.5*width, 10*np.log10(np.squeeze(Gxx_inph_pred)[::2] * df_pred/20e-6**2),width = width,hatch =hatch[1]*2,align='center')
    ax[0].bar(BPF_harm[::2][1:4]+width*1-.5*width, 10*np.log10(Gxx_inph_exp[::2][1:4] * df_exp/20e-6**2),width = width,hatch =hatch[1]*2,align='center')

    ax[1].bar(f_pred[::2]/ (loadParams['omega'] /60 * Nb)+width*0-.5*width, 10*np.log10(np.squeeze(Gxx_outph_pred)[::2,m-1] * df_pred/20e-6**2),width = width,hatch =hatch[1]*2,align='center')
    ax[1].bar(BPF_harm[::2][1:4]+width*1-.5*width, 10*np.log10(Gxx_outph_exp[::2,m-1][1:4] * df_exp/20e-6**2),width = width,hatch =hatch[1]*2,align='center')

    ax[2].bar(f_pred[::2]/ (loadParams['omega'] /60 * Nb)+width*0-.5*width, 10*np.log10(np.squeeze(Gxx_tot_pred)[::2,m-1] * df_pred/20e-6**2),width = width,hatch =hatch[1]*2,align='center')
    ax[2].bar(BPF_harm[::2][1:4]+width*1-.5*width, spl[::2,m-1][1:4],width = width,hatch =hatch[1]*2,align='center')

    for i in range(3):
        if i!=2:
            ax[i].tick_params(axis='x', labelsize=0)
        ax[i].set_yticks(np.arange(0, 60, 20))
        ax[i].axis([0, 4, 0, 60])
        ax[i].set_xticks(np.arange(1, 5))
        ax[i].grid('on')

    ax[0].set_title('In-Phase')
    ax[1].set_title('Out-of-Phase')
    ax[2].set_title('Total')
    ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[-1].set_xlabel('BPF Harmonic')
    # ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
    plt.suptitle(f'$Mic\ {m} \ ( \phi = {round(LN_data["phi"][m - 1] * 180 / np.pi)}^\circ)$')
    ax[-1].legend(['Predicted', 'Measured'], loc='center', ncol=2, bbox_to_anchor=(.5, -.625))
    plt.savefig(os.path.join(pred_dir, 'figures', f'exp_vs_pred_ph_sep_spec_m{m}.png'), format='png')
    plt.savefig(os.path.join(pred_dir, 'figures', f'exp_vs_pred_ph_sep_spec_m{m}.eps'), format='eps')

