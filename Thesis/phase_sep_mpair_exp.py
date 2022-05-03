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
exp_dir ='/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/rpm_sweep/h2b/h2b8'
save_dir = '/Users/danielweitsman/Desktop/Masters_Research/lynx/figures/exp/mic_pair_ph_sep/hover/h2b8'

save_h5 = False

# min and max shaft order harmonics to exclude from the analysis (set max to -1 to include all upper order harmonics)
harm_filt = [2, 6]

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 70]

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

#%% Spherical spreading correction       
phi = np.array([23,18,12,6,0])
# micR = np.array([65.19,62.97,61.34,60.34,60.00,60.34,61.34,62.97,65.19,67.93,71.14,74.75])
# exp = exp * micR / micR[4]

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig1,ax1 = plt.subplots(3,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.17)
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig2,ax2 = plt.subplots(3,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.17)
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig3,ax3 = plt.subplots(3,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.17)
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig4,ax4 = plt.subplots(3,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.17)

#%% Sync average total

Nb = 2
data = {}
t = np.arange(len(ttl)) / fs_ttl
t_acs = np.arange(len(exp)) / fs_exp

LE_ind, lim_ind, rpm_nom, u_rpm = fun.rpm_eval(ttl,fs_ttl,start_t,end_t)
ind = list(map(lambda x: bisect(t_acs,x),t[LE_ind[lim_ind[0]:lim_ind[1]]]))

    #%%
leglab = []
width = .125
hatch = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
linestyle = ['--','-.',':','-']
f, fs1, spl, u_low, u_high, Xn_avg, Xm_avg, Xn_avg_filt, Xn_bb = fun.harm_extract(exp, tac_ind=ind, fs=fs_exp, rev_skip=0, harm_filt=harm_filt, filt_shaft_harm =  True, Nb=Nb)
t_nondim = np.arange(len(Xn_avg_filt))*fs1**-1/(rpm_nom/60)**-1
BPF_harm = np.arange(len(Xn_avg_filt)/2)/Nb
df = (len(Xn_avg_filt)*fs1**-1)**-1

Nobs = np.shape(exp)[-1]-3
for m_itr, m in enumerate(range(int(Nobs / 2))):
    print(f'{m_itr},{Nobs - m_itr - 1}')
    # leglab.append(f'Mics {m_itr + 1} & {Nobs - m_itr}')
    leglab.append(f'$\phi = \pm {phi[m_itr]}^\circ$')

    xn_inph = ifft((Xm_avg[:,m_itr]+Xm_avg[:,Nobs-m_itr-1])/2)*fs1
    Xm_inph = (Xm_avg[:,m_itr]+Xm_avg[:,Nobs-m_itr-1])/2
    Gxx_inph = fun.SD(Xm_inph,fs1)
    Xm_outph = Xm_avg-np.expand_dims(Xm_inph,axis =1)
    Gxx_outph = fun.SD(Xm_outph,fs1)

#%% Plots predicted pressure time series

    ax1[0].bar(BPF_harm[::2] + width * m_itr - 1.5*width , 10 * np.log10(Gxx_inph[::2] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)
    ax1[1].bar(BPF_harm[::2] + width * m_itr - 1.5* width , 10 * np.log10(Gxx_outph[::2, m_itr]* df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)
    ax1[2].bar(BPF_harm[::2] + width * m_itr - 1.5* width , spl[::2,m_itr], width = width, align='center', hatch =hatch[m_itr] * 2)

    ax2[0].bar(BPF_harm[::2] + width * m_itr - 1.5*width , 10 * np.log10(Gxx_inph[::2] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)
    ax2[1].bar(BPF_harm[::2]+ width * m_itr - 1.5* width , 10 * np.log10(Gxx_outph[::2, Nobs-m_itr-1]* df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)
    ax2[2].bar(BPF_harm[::2]+ width * m_itr - 1.5* width , spl[::2,Nobs-m_itr-1], width = width, align='center', hatch =hatch[m_itr] * 2)

    ax3[0].plot(t_nondim, xn_inph,linestyle = linestyle[m_itr])
    ax3[1].plot(t_nondim, Xn_avg_filt[:,m_itr]-xn_inph,linestyle = linestyle[m_itr])
    ax3[2].plot(t_nondim, Xn_avg_filt[:,m_itr],linestyle = linestyle[m_itr])

    ax4[0].plot(t_nondim, xn_inph,linestyle = linestyle[m_itr])
    ax4[1].plot(t_nondim, Xn_avg_filt[:,Nobs-m_itr-1]-xn_inph,linestyle = linestyle[m_itr])
    ax4[2].plot(t_nondim, Xn_avg_filt[:,Nobs-m_itr-1],linestyle = linestyle[m_itr])

for i in range(3):
    if i != 2:
        ax1[i].tick_params(axis='x', labelsize=0)
    ax1[i].set_yticks(np.arange(0, 60, 20))
    ax1[i].axis([0, 4, 0, 65])
    ax1[i].set_xticks(np.arange(1, 5))
    ax1[i].grid('on')

ax1[0].set_title('In-phase')
ax1[1].set_title('Out-of-Phase')
ax1[2].set_title('Total')
ax1[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax1[-1].set_xlabel('BPF Harmonic')
fig1.suptitle(f'Upper Observers')
ax1[-1].legend(leglab, loc='center', ncol=4, bbox_to_anchor=(.5, -.68), columnspacing=1, handlelength=1.25)

# if save_fig:
fig1.savefig(os.path.join(save_dir, f'mic_pair_avg_spec_upper.eps'), format='eps')
fig1.savefig(os.path.join(save_dir, f'mic_pair_avg_spec_upper.png'), format='png')
    # %%

for i in range(3):
    if i != 2:
        ax2[i].tick_params(axis='x', labelsize=0)
    ax2[i].set_yticks(np.arange(0, 60, 20))
    ax2[i].axis([0, 4, 0, 65])
    ax2[i].set_xticks(np.arange(1, 5))
    ax2[i].grid('on')

ax2[0].set_title('In-phase')
ax2[1].set_title('Out-of-Phase')
ax2[2].set_title('Total')
ax2[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax2[-1].set_xlabel('BPF Harmonic')
# ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
fig2.suptitle(f'Lower Observers')
ax2[-1].legend(leglab, loc='center', ncol=4, bbox_to_anchor=(.5, -.68), columnspacing=1, handlelength=1.25)
# if save_fig:
fig2.savefig(os.path.join(save_dir, f'mic_pair_avg_spec_lower.eps'), format='eps')
fig2.savefig(os.path.join(save_dir, f'mic_pair_avg_spec_lower.png'), format='png')

for i in range(3):
    if i != 2:
        ax3[i].tick_params(axis='x', labelsize=0)
    ax3[i].set_xlim([0, 1])
    ax3[i].ticklabel_format(axis='y', style='sci', scilimits=(-3, -2))

    ax3[i].grid('on')

ax3[0].set_title('In-phase')
ax3[1].set_title('Out-of-Phase')
ax3[2].set_title('Total')
ax3[1].set_ylabel('Pressure [Pa]')
ax3[-1].set_xlabel('Revolution')
# ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
fig3.suptitle(f'Upper Observers')
ax3[-1].legend(leglab, loc='center', ncol=4, bbox_to_anchor=(.5, -.68), columnspacing=1, handlelength=1.25)
# if save_fig:
fig3.savefig(os.path.join(save_dir, f'mic_pair_avg_tseries_upper.eps'), format='eps')
fig3.savefig(os.path.join(save_dir, f'mic_pair_avg_tseries_upper.png'), format='png')

for i in range(3):
    if i != 2:
        ax4[i].tick_params(axis='x', labelsize=0)
    ax4[i].set_xlim([0, 1])
    ax4[i].ticklabel_format(axis='y', style='sci', scilimits=(-3, -2))
    ax4[i].grid('on')

ax4[0].set_title('In-phase')
ax4[1].set_title('Out-of-Phase')
ax4[2].set_title('Total')
ax4[1].set_ylabel('Pressure [Pa]')
ax4[-1].set_xlabel('Revolution')
# ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
fig4.suptitle(f'Lower Observers')
ax4[-1].legend(leglab, loc='center', ncol=4, bbox_to_anchor=(.5, -.68), columnspacing=1, handlelength=1.25)
# if save_fig:
fig4.savefig(os.path.join(save_dir, f'mic_pair_avg_tseries_lower.eps'), format='eps')
fig4.savefig(os.path.join(save_dir, f'mic_pair_avg_tseries_lower.png'), format='png')
