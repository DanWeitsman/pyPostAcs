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
leglab = [r'$\theta_{1c} = -4^\circ$',r'$\theta_{1c} = 4^\circ$']

save_dir = '/Users/danielweitsman/Desktop/Masters_Research/lynx/figures/h2b69_th1c'
save_fig = True
#   Linestyle for each case
linestyle =['-','-.',':','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]

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

    L = np.shape(data[case]['LN_data']['p_tot'])[-1]
    fs = L / (loadParams[case]['omega']/(2*np.pi))**-1
    df = ((fs ** -1) * L) ** -1
    N = ((fs ** -1 * df) ** -1)
    Nfft = np.floor(L / N)

    f, Xm_unsteady, Sxx_tot_unsteady, Gxx_unsteady, Gxx_avg_unsteady = fun.msPSD(data[case]['LN_data']['p_unsteady_ff'].transpose(), fs = fs, df = df, win = False, ovr = 0, save_fig = False, plot = False)
    f, Xm_steady, Sxx_tot_steady, Gxx_steady, Gxx_avg_steady = fun.msPSD(data[case]['LN_data']['p_steady_ff'].transpose(), fs = fs, df = df, win = False, ovr = 0, save_fig = False, plot = False)
    f, Xm_tot, Sxx_tot_tot, Gxx_tot, Gxx_avg_tot = fun.msPSD(data[case]['LN_data']['p_tot'].transpose(), fs = fs, df = df, win = False, ovr = 0, save_fig = False, plot = False)
    data[case]['LN_data']['Gxx_avg_unsteady'] = Gxx_avg_unsteady
    data[case]['LN_data']['Gxx_avg_unsteady'] = Gxx_avg_steady
    data[case]['LN_data']['Gxx_avg_unsteady'] = Gxx_avg_tot

#%%%

width = .125
hatch = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig, ax = plt.subplots(len(mics), 1, figsize=(8, 6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace=0.35, bottom=0.17,top = 0.93)

for case_itr,case in enumerate(caseName):    
    #   Loops through each mic
    for i, m in enumerate(mics):
        #   Plots the resulting spectra in dB
            ax[i].plot(data[case]['LN_data']['ts'][:-1] / (data[case]['LN_data']['dt'] * 360), data[case]['LN_data']['p_unsteady_ff'][m - 1], linestyle=linestyle[case_itr])
    
for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m - 1] * 180 / np.pi)}^\circ)$')
    if i != len(mics) - 1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xlim([0, 1])
    ax[i].ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))
    # ax[i].set_ylim([-0.015,0.015])
    ax[i].grid('on')

ax[int(len(mics) / 2)].set_ylabel('Pressure [Pa]')
ax[- 1].set_xlabel('Rotation')
if isinstance(leglab, list):
    ax[-1].legend(leglab, ncol=len(caseName), loc='center',bbox_to_anchor=(0.5, -0.68))
else:
    ax[-1].legend(caseName, ncol=len(caseName), loc='center', bbox_to_anchor=(0.5, -0.68))

# ax[-1].legend(['$\partial l_r/\partial t$','$l_r(1-M_r)^{-1}(\partial M_r/\partial t)$','Near-Field','$\partial l_r/\partial t+l_r(1-M_r)^{-1}(\partial M_r/\partial t)$'], ncol=3,loc='center',bbox_to_anchor=(0.5, -0.55))
plt.savefig(os.path.join(save_dir, 'p_unsteady_tseries_compare.png'), format='png')
plt.savefig(os.path.join(save_dir, 'p_unsteady_tseries_compare.eps'), format='eps')

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.17,top = .93)

for case_itr,case in enumerate(caseName):    
    #   Loops through each mic
    for i,m in enumerate(mics):
        #   Plots the resulting spectra in dB
        ax[i].bar(f[::2][1:4]/(loadParams[case]['omega']/(2*np.pi)*UserIn[case]['Nb'])+width*case_itr-width*.5, 10*np.log10(data[case]['LN_data']['Gxx_avg_unsteady'][::2,m-1][1:4] * df/20e-6**2),width = width,hatch =hatch[case_itr]*2,align='center')

for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(data[case]["LN_data"]["phi"][m-1]*180/np.pi)}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_yticks(np.arange(0, 60, 20))
    ax[i].axis([0,4,0,60])
    ax[i].set_xticks(np.arange(1, 5))
    ax[i].grid('on')
ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax[-1].set_xlabel('BPF Harmonic')
if isinstance(leglab, list):
    ax[-1].legend(leglab, ncol=len(caseName), loc='center',bbox_to_anchor=(0.5, -0.68))
else:
    ax[-1].legend(caseName, ncol=len(caseName), loc='center', bbox_to_anchor=(0.5, -0.68))
plt.savefig(os.path.join(save_dir, 'p_unsteady_spec_compare.eps'),format='eps')
plt.savefig(os.path.join(save_dir, 'p_unsteady_spec_compare.png'),format='png')

#%%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5), subplot_kw=dict(polar=True))
for case_itr,case in enumerate(caseName):
    ax.plot(data[case]["LN_data"]['phi'], data[case]["LN_data"]['oaspl_unsteady_ff'])
ax.set_thetamax(data[case]["LN_data"]['phi'][0] * 180 / np.pi + 2)
ax.set_thetamin(data[case]["LN_data"]['phi'][-1] * 180 / np.pi - 2)
ax.set_ylim([0, 60])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)', position=(1, .25), labelpad=-20,
              rotation=data[case]["LN_data"]['phi'][-1] * 180 / np.pi - 3)
if isinstance(leglab, list):
    ax.legend(leglab, ncol=1, loc='center',bbox_to_anchor=(-.115, 0.9))
else:
    ax.legend(caseName, ncol=1, loc='center', bbox_to_anchor=(-.115, 0.9))

plt.savefig(os.path.join(save_dir, 'p_unsteady_direct_compare.eps'),format='eps')
plt.savefig(os.path.join(save_dir, 'p_unsteady_direct_compare.png'),format='png')
