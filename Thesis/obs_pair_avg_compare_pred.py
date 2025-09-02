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
fontSize = 16
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
c  = list(mcolors.TABLEAU_COLORS.keys())[:10]
cmap = plt.cm.Spectral.reversed()

#%%
case_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/'
# If you want to compare specific points in this directory their names can be specified in caseName list. Otherwise,
# all the cases in this directory directory would be compared and used to generate the thrust/torque profiles.
caseName = ['h2b69_rgrid']

save_dir = '//Users/danielweitsman/Desktop/Masters_Research/lynx/figures/exp/mic_pair_ph_sep/hover/h2b69'
save_fig = False
#   Linestyle for each case
linestyle =['-.','--',':','-']

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 60]

width = .125
hatch = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']

#%%
# mics = np.delete(mics,1)
leglab = []
data = {}
UserIn = {}
geomParams = {}
loadParams = {}

for case in caseName:
    out = {}
    temp_out = {}
    with h5py.File(os.path.join(case_dir,case,case+'_mywopwop_out.h5'), 'r') as f:
        for dict_k in list(f.keys()):
            for k, v in f[dict_k].items():
                temp_out = {**temp_out, **{k: v[()]}}
            out = {**out, **{dict_k: temp_out}}
    data = {**data,**{case:{'LN_data':out['LN_data'],'TN_data':out['TN_data']}}}

#%%

    #   imports the dictionaries saved in the MainDict.h5 file from VSP2WOPWOP.
    with h5py.File(os.path.join(case_dir,case, 'MainDict.h5'), "r") as f:
        geomParams_temp = {}
        for k, v in f[list(f.keys())[0]]['geomParams'].items():
            geomParams_temp = {**geomParams_temp,**{k: v[()]}}
        geomParams = {**geomParams, **{case:geomParams_temp}}
        loadParams_temp = {}
        for k, v in f[list(f.keys())[0]]['loadParams'].items():
            loadParams_temp = {**loadParams_temp,**{k: v[()]}}
        loadParams = {**loadParams, **{case: loadParams_temp}}
        UserIn_temp = {}
        for k, v in f[list(f.keys())[1]].items():
            UserIn_temp = {**UserIn_temp,**{k: v[()]}}
        UserIn = {**UserIn, **{case: UserIn_temp}}

# %%

for case_itr,case in enumerate(caseName):

    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig1, ax1 = plt.subplots(3, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.35, bottom=0.17)
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.35, bottom=0.17)
    fig3, ax3 = plt.subplots(3, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.35, bottom=0.17)
    fig4, ax4 = plt.subplots(3, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.35, bottom=0.17)


    Nobs = len(data[case]['LN_data']['p_tot'])

    for m_itr,m in enumerate(range(int(Nobs/2))):
        print(f'{m_itr},{Nobs-m_itr-1}')
        leglab.append(f'M{m_itr+1} & M{Nobs-m_itr}')
        p_tot = data[case]['LN_data']['p_tot'] + data[case]['TN_data']['p']
        p_in_phase = (p_tot[m_itr] + p_tot[Nobs-m_itr-1]) / 2
        p_out_phase = p_tot - p_in_phase

        L = np.shape(p_tot)[-1]
        fs = L / (loadParams[case]['omega']/(2*np.pi))**-1
        df = ((fs ** -1) * L) ** -1
        N = ((fs ** -1 * df) ** -1)
        Nfft = np.floor(L / N)

        f, Xm_tot, Sxx_tot, Gxx_tot, Gxx_avg_tot = fun.msPSD(p_tot.transpose(), fs = fs, df = df, win = False, ovr = 0, save_fig = False, plot = False)
        # f, Xm_tn, Sxx_tn, Gxx_tn, Gxx_avg_tn = fun.msPSD(data[case]['TN_data']['p'].transpose(), fs = fs, df = df, win = False, ovr = 0, save_fig = False, plot = False)
        # f, Xm_ln, Sxx_ln, Gxx_ln, Gxx_avg_ln = fun.msPSD(data[case]['LN_data']['p_tot'].transpose(), fs = fs,df=df, win=False, ovr=0, save_fig=False, plot=False)
        f, Xm_inph, Sxx_inph, Gxx_inph, Gxx_avg_inph = fun.msPSD(p_in_phase.transpose(), fs = fs, df = df, win = False, ovr = 0, save_fig = False, plot = False)
        f, Xm_outph, Sxx_outph, Gxx_outph, Gxx_avg_outph = fun.msPSD(p_out_phase.transpose(), fs = fs,df=df, win=False, ovr=0, save_fig=False, plot=False)

        # ph_sep = {'p_tot':p_tot,'p_in_phase':p_in_phase,'p_out_phase':p_out_phase,'f':f,'df':df,'Gxx_avg_tot':Gxx_avg_tot,'Gxx_avg_inph':Gxx_avg_inph,'Gxx_avg_outph':Gxx_avg_outph}
        # data[case]['ph_sep'] = ph_sep


#%%

            #   Loops through each mic
            #   Plots the resulting spectra in dB
        ax1[0].bar(f[::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * m_itr - 1.5*width , 10 * np.log10(Gxx_avg_inph[::2, 0][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)
        ax1[1].bar(f[::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * m_itr - 1.5* width , 10 * np.log10(Gxx_avg_outph[::2, m_itr][1:4]* df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)
        ax1[2].bar(f[::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * m_itr - 1.5* width , 10 * np.log10(Gxx_avg_tot[::2, m_itr][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)

        ax2[0].bar(f[::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * m_itr - 1.5*width , 10 * np.log10(Gxx_avg_inph[::2, 0][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)
        ax2[1].bar(f[::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * m_itr - 1.5* width , 10 * np.log10(Gxx_avg_outph[::2, Nobs-m_itr-1][1:4]* df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)
        ax2[2].bar(f[::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * m_itr - 1.5* width , 10 * np.log10(Gxx_avg_tot[::2, Nobs-m_itr-1][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[m_itr] * 2)

        ax3[0].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , p_in_phase,linestyle = linestyle[m_itr])
        ax3[1].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , p_out_phase[m_itr], linestyle = linestyle[m_itr])
        ax3[2].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , p_tot[m_itr],linestyle = linestyle[m_itr])

        ax4[0].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , p_in_phase,linestyle = linestyle[m_itr])
        ax4[1].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , p_out_phase[Nobs-m_itr-1], linestyle = linestyle[m_itr])
        ax4[2].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , p_tot[Nobs-m_itr-1],linestyle = linestyle[m_itr])

    for i in range(3):
        if i!=2:
            ax1[i].tick_params(axis='x', labelsize=0)
        ax1[i].set_yticks(np.arange(0, 60, 20))
        ax1[i].axis([0, 4, 0, 60])
        ax1[i].set_xticks(np.arange(1, 5))
        ax1[i].grid('on')

    ax1[0].set_title('In-phase')
    ax1[1].set_title('Out-of-Phase')
    ax1[2].set_title('Total')
    ax1[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax1[-1].set_xlabel('BPF Harmonic')
    fig1.suptitle(f'Upper Observers')
    ax1[-1].legend(leglab, loc='center', ncol=4,bbox_to_anchor=(.5, -.68),columnspacing = 1,handlelength = 1.25)

    if save_fig:
        fig1.savefig(os.path.join(save_dir, f'mic_pair_avg_spec_upper.eps'), format='eps')
        fig1.savefig(os.path.join(save_dir, f'mic_pair_avg_spec_upper.png'), format='png')
        #%%

    for i in range(3):
        if i!=2:
            ax2[i].tick_params(axis='x', labelsize=0)
        ax2[i].set_yticks(np.arange(0, 60, 20))
        ax2[i].axis([0, 4, 0, 60])
        ax2[i].set_xticks(np.arange(1, 5))
        ax2[i].grid('on')

    ax2[0].set_title('In-phase')
    ax2[1].set_title('Out-of-Phase')
    ax2[2].set_title('Total')
    ax2[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax2[-1].set_xlabel('BPF Harmonic')
    # ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
    fig2.suptitle(f'Lower Observers')
    ax2[-1].legend(leglab, loc='center', ncol=4,bbox_to_anchor=(.5, -.68),columnspacing = 1,handlelength = 1.25)
    if save_fig:
        fig2.savefig(os.path.join(save_dir, f'mic_pair_avg_spec_lower.eps'), format='eps')
        fig2.savefig(os.path.join(save_dir, f'mic_pair_avg_spec_lower.png'), format='png')

    for i in range(3):
        if i!=2:
            ax3[i].tick_params(axis='x', labelsize=0)
        ax3[i].set_xlim([0,1])
        ax3[i].ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))

        ax3[i].grid('on')

    ax3[0].set_title('In-phase')
    ax3[1].set_title('Out-of-Phase')
    ax3[2].set_title('Total')
    ax3[1].set_ylabel('Pressure [Pa]')
    ax3[-1].set_xlabel('Rotation')
    # ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
    fig3.suptitle(f'Upper Observers')
    ax3[-1].legend(leglab, loc='center', ncol=4,bbox_to_anchor=(.5, -.68),columnspacing = 1,handlelength = 1.25)
    if save_fig:
        fig3.savefig(os.path.join(save_dir, f'mic_pair_avg_tseries_upper.eps'), format='eps')
        fig3.savefig(os.path.join(save_dir, f'mic_pair_avg_tseries_upper.png'), format='png')

    for i in range(3):
        if i!=2:
            ax4[i].tick_params(axis='x', labelsize=0)
        ax4[i].set_xlim([0,1])
        ax4[i].ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))
        ax4[i].grid('on')

    ax4[0].set_title('In-phase')
    ax4[1].set_title('Out-of-Phase')
    ax4[2].set_title('Total')
    ax4[1].set_ylabel('Pressure [Pa]')
    ax4[-1].set_xlabel('Rotation')
    # ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
    fig4.suptitle(f'Lower Observers')
    ax4[-1].legend(leglab, loc='center', ncol=4,bbox_to_anchor=(.5, -.68),columnspacing = 1,handlelength = 1.25)
    if save_fig:
        fig4.savefig(os.path.join(save_dir, f'mic_pair_avg_tseries_lower.eps'), format='eps')
        fig4.savefig(os.path.join(save_dir, f'mic_pair_avg_tseries_lower.png'), format='png')
