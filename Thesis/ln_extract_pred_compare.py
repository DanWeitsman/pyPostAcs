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
caseName = ['h2b69_n3deg_th1c_rgrid_2','h2b69_3deg_th1c_rgrid_2']
leglab = [r'$\theta_{1c} = -3^\circ$',r'$\theta_{1c} = 3^\circ$']

save_dir = '/Users/danielweitsman/Desktop/Masters_Research/lynx/figures/h2b69_th1c'
save_fig = True
#   Linestyle for each case
linestyle =['-','-.',':','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,9]

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 60]


#%%
# mics = np.delete(mics,1)

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
    print(case)

    p_tot = data[case]['LN_data']['p_tot'] + data[case]['TN_data']['p']
    p_in_phase = (p_tot[mics[0]-1] + p_tot[mics[-1]-1]) / 2

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

    ph_sep = {'p_tot':p_tot,'p_in_phase':p_in_phase,'p_out_phase':p_out_phase,'f':f,'df':df,'Gxx_avg_tot':Gxx_avg_tot,'Gxx_avg_inph':Gxx_avg_inph,'Gxx_avg_outph':Gxx_avg_outph}
    data[case]['ph_sep'] = ph_sep

#%%%

width = .125
hatch = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(2,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.17)

for case_itr,case in enumerate(caseName):
    #   Loops through each mic
    for m_itr, m in enumerate(mics):

        #   Plots the resulting spectra in dB
        ax[m_itr].bar(data[case]['ph_sep']['f'][::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * case_itr - width , 10 * np.log10(data[case]['ph_sep']['Gxx_avg_inph'][::2, 0][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[case_itr] * 2)

for m_itr, m in enumerate(mics):
    ax[m_itr].set_title(f'$M{m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')
    if m_itr!=len(mics)-1:
        ax[m_itr].tick_params(axis='x', labelsize=0)
    ax[m_itr].set_yticks(np.arange(0, 60, 20))
    ax[m_itr].axis([0, 4, 0, 60])
    ax[m_itr].set_xticks(np.arange(1, 5))
    ax[m_itr].grid('on')
    ax[m_itr].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

ax[-1].set_xlabel('BPF Harmonic')
# ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
plt.suptitle('In-Phase')

if isinstance(leglab,list):
    ax[-1].legend(leglab, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))
else:
    ax[-1].legend(caseName, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))

if save_fig:
    plt.savefig(os.path.join(save_dir, f'ph_sep_in_phase_spec_compare.eps'), format='eps')
    plt.savefig(os.path.join(save_dir, f'ph_sep_in_phase_spec_compare.png'), format='png')

#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(2,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.17)

for case_itr,case in enumerate(caseName):
    #   Loops through each mic
    for m_itr, m in enumerate(mics):

        #   Plots the resulting spectra in dB
        ax[m_itr].bar(data[case]['ph_sep']['f'][::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * case_itr - width , 10 * np.log10(data[case]['ph_sep']['Gxx_avg_outph'][::2,m-1][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[case_itr] * 2)

for m_itr, m in enumerate(mics):
    ax[m_itr].set_title(f'$M{m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')
    if m_itr!=len(mics)-1:
        ax[m_itr].tick_params(axis='x', labelsize=0)
    ax[m_itr].set_yticks(np.arange(0, 60, 20))
    ax[m_itr].axis([0, 4, 0, 60])
    ax[m_itr].set_xticks(np.arange(1, 5))
    ax[m_itr].grid('on')
    ax[m_itr].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

ax[-1].set_xlabel('BPF Harmonic')
# ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
plt.suptitle('Out-of-Phase')

if isinstance(leglab,list):
    ax[-1].legend(leglab, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))
else:
    ax[-1].legend(caseName, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))

if save_fig:
    plt.savefig(os.path.join(save_dir, f'ph_sep_out_phase_spec_compare.eps'), format='eps')
    plt.savefig(os.path.join(save_dir, f'ph_sep_out_phase_spec_compare.png'), format='png')
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(2,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.17)

for case_itr,case in enumerate(caseName):
    #   Loops through each mic
    for m_itr, m in enumerate(mics):

        #   Plots the resulting spectra in dB
        ax[m_itr].bar(data[case]['ph_sep']['f'][::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * case_itr - width , 10 * np.log10(data[case]['ph_sep']['Gxx_avg_tot'][::2,m-1][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[case_itr] * 2)

for m_itr, m in enumerate(mics):
    ax[m_itr].set_title(f'$M{m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')
    if m_itr!=len(mics)-1:
        ax[m_itr].tick_params(axis='x', labelsize=0)
    ax[m_itr].set_yticks(np.arange(0, 60, 20))
    ax[m_itr].axis([0, 4, 0, 60])
    ax[m_itr].set_xticks(np.arange(1, 5))
    ax[m_itr].grid('on')
    ax[m_itr].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

ax[-1].set_xlabel('BPF Harmonic')
# ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
plt.suptitle('Total')

if isinstance(leglab,list):
    ax[-1].legend(leglab, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))
else:
    ax[-1].legend(caseName, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))

if save_fig:
    plt.savefig(os.path.join(save_dir, f'ph_sep_tot_spec_compare.eps'), format='eps')
    plt.savefig(os.path.join(save_dir, f'ph_sep_tot_spec_compare.png'), format='png')

#%%

for m_itr, m in enumerate(mics):
    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig,ax = plt.subplots(3,1,figsize = (8,6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace = 0.35,bottom = 0.17)

    for case_itr,case in enumerate(caseName):
        #   Loops through each mic
            #   Plots the resulting spectra in dB
        ax[0].bar(data[case]['ph_sep']['f'][::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * case_itr - 1.5*width , 10 * np.log10(data[case]['ph_sep']['Gxx_avg_inph'][::2, 0][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[case_itr] * 2)
        ax[1].bar(data[case]['ph_sep']['f'][::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * case_itr - 1.5* width , 10 * np.log10(data[case]['ph_sep']['Gxx_avg_outph'][::2,m-1][1:4]* df / 20e-6 ** 2), width = width, align='center', hatch =hatch[case_itr] * 2)
        ax[2].bar(data[case]['ph_sep']['f'][::2][1:4] / (loadParams[case]['omega'] / (2 * np.pi) * UserIn[case]['Nb']) + width * case_itr - 1.5* width , 10 * np.log10(data[case]['ph_sep']['Gxx_avg_tot'][::2,m-1][1:4] * df / 20e-6 ** 2), width = width, align='center', hatch =hatch[case_itr] * 2)

    for i in range(3):
        if i!=2:
            ax[i].tick_params(axis='x', labelsize=0)
        ax[i].set_yticks(np.arange(0, 60, 20))
        ax[i].axis([0, 4, 0, 60])
        ax[i].set_xticks(np.arange(1, 5))
        ax[i].grid('on')

    ax[0].set_title('In-phase')
    ax[1].set_title('Out-of-Phase')
    ax[2].set_title('Total')
    ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[-1].set_xlabel('BPF Harmonic')
    # ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
    plt.suptitle(f'$M{m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')

    if isinstance(leglab,list):
        ax[-1].legend(leglab, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))
    else:
        ax[-1].legend(caseName, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))

    if save_fig:
        plt.savefig(os.path.join(save_dir, f'ph_sep_spec_compare_m{m}.eps'), format='eps')
        plt.savefig(os.path.join(save_dir, f'ph_sep_spec_compare_m{m}.png'), format='png')

#%%

for m_itr, m in enumerate(mics):
    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig,ax = plt.subplots(3,1,figsize = (8,6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace = 0.35,bottom = 0.17)

    for case_itr,case in enumerate(caseName):
        #   Loops through each mic
            #   Plots the resulting spectra in dB
        ax[0].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , data[case]['ph_sep']['p_in_phase'],linestyle = linestyle[case_itr])
        ax[1].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , data[case]['ph_sep']['p_out_phase'][m-1],linestyle = linestyle[case_itr])
        ax[2].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt']*360) , data[case]['ph_sep']['p_tot'][m-1],linestyle = linestyle[case_itr])

    for i in range(3):
        if i!=2:
            ax[i].tick_params(axis='x', labelsize=0)
        # ax[i].set_yticks(np.arange(0, 60, 20))
        # ax[i].axis([0, 5, 0, 60])
        # ax[i].set_xticks(np.arange(1, 6))
        ax[i].set_xlim([0,1])
        ax[i].grid('on')
        ax[i].ticklabel_format(axis = 'y',style = 'sci',scilimits=(-2,-2))

    ax[0].set_title('In-phase')
    ax[1].set_title('Out-of-Phase')
    ax[2].set_title('Total')
    ax[1].set_ylabel('Pressure [Pa]')
    ax[-1].set_xlabel('Rotation')
    # ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
    plt.suptitle(f'$M{m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')

    if isinstance(leglab,list):
        ax[-1].legend(leglab, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))
    else:
        ax[-1].legend(caseName, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.68))
    if save_fig:
        plt.savefig(os.path.join(save_dir, f'ph_sep_tseries_compare_m{m}.eps'),format='eps')
        plt.savefig(os.path.join(save_dir, f'ph_sep_tseries_compare_m{m}.png'),format='png')

#%%

    # for i in range(3):
    #     if i!=2:
    #         ax[i].tick_params(axis='x', labelsize=0)
    #     # ax[i].set_yticks(np.arange(0, 60, 20))
    #     # ax[i].axis([0, 5, 0, 60])
    #     # ax[i].set_xticks(np.arange(1, 6))
    #     ax[i].set_xlim([0,1])
    #     ax[i].grid('on')
    #
    # ax[0].set_title('In-phase')
    # ax[1].set_title('Out-of-Phase')
    # ax[2].set_title('Total')
    # ax[1].set_ylabel('Pressure [Pa]')
    # ax[-1].set_xlabel('Revolution')
    # # ax[-1].legend(['Thickness','Loading', 'Total','In-phase', 'Out-of-phase'],loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.35))
    # plt.suptitle(f'$Mic\ {m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')
    #
    # if isinstance(leglab,list):
    #     ax[-1].legend(leglab, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.55))
    # else:
    #     ax[-1].legend(caseName, loc='center', ncol=len(caseName),bbox_to_anchor=(.5, -.55))

# for m_itr, m in enumerate(mics):
#     fig,ax = plt.subplots(1,len(caseName),figsize = (8,6),subplot_kw=dict(projection='polar'))
#     #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
#     plt.subplots_adjust(hspace=0.35,wspace = .35)
#     for case_itr,case in enumerate(caseName):
#         quant = data[case]['LN_data']['unsteady_ff'][m]
#         # levels = np.linspace(-.005, .005, 50)
#         levels = np.linspace(np.min(quant), np.max(quant), 50)
#         dist = ax[case_itr].contourf(data[case]['LN_data']['psi'], geomParams[case]['rdim'], quant, levels=levels)
#         ax[case_itr].set_ylim(geomParams[case]['rdim'][0], geomParams[case]['rdim'][-1])
#         if isinstance(leglab,list):
#             ax[case_itr].set_title(leglab[case_itr])
#         else:
#             ax[case_itr].set_title(caseName[case_itr])
#
#     plt.suptitle(f'$Mic\ {m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')

#%%
# for m_itr, m in enumerate(mics):
#     fig,ax = plt.subplots(1,len(caseName),figsize = (8,6),subplot_kw=dict(projection='polar'))
#     #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
#     plt.subplots_adjust(hspace=0.35,wspace = .35)
#     for case_itr,case in enumerate(caseName):
#         quant = (data[case]['LN_data']['unsteady_ff']+data[case]['LN_data']['steady_ff']+data[case]['LN_data']['nf'])[m]
#         # levels = np.linspace(-.005, .005, 50)
#         levels = np.linspace(np.min(quant), np.max(quant), 50)
#         dist = ax[case_itr].contourf(data[case]['LN_data']['psi'], geomParams[case]['rdim'], quant, levels=levels,cmap = cmap)
#         ax[case_itr].set_ylim(geomParams[case]['rdim'][0], geomParams[case]['rdim'][-1])
#         if isinstance(leglab,list):
#             ax[case_itr].set_title(leglab[case_itr])
#         else:
#             ax[case_itr].set_title(caseName[case_itr])
#
#     plt.suptitle(f'$Mic\ {m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')
#
# cbar = fig.colorbar(dist)
# cbar.ax[case_itr].set_ylabel(r'$\left[\frac{Q_T}{r(1-M_r)}\right]_{ret}$')