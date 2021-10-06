import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})


#%%
dir ='//Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/long_cyc_sweep/lowM/h2b/'
save_dir = '/Users/danielweitsman/Desktop/Masters_Research/Thesis/Analysis/9_15_21'
# If you want to compare specific points in this directory their names can be specified in caseName list. Otherwise,
# all the cases in this directory directory would be compared and used to generate the thrust/torque profiles.
caseName = ['h2b48','h2b46','h2b56']

# leglab = ['Collective trim', 'Cyclic trim']
leglab = [r"$\theta_{1s} =0^\circ$",r'$\theta_{1x} = 4^\circ$',r'$\theta_{1s} = -4^\circ$']
#   Linestyle for each case
linestyle =['-','-.','--','-',':']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,9]

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 70]

#%%
sdata = {}
#   imports several performance quantities from the MainDict.h5 file.
for case in caseName:
    # os.path.join(os.path.dirname(dir), caseName[0], caseName[0] + '_sdata.h5')
    with h5py.File(os.path.join(os.path.dirname(dir), case, case + '_sdata.h5'),'r') as f:
        temp = {}
        for k, dat in f.items():
            temp = {**temp,**{k:dat[()]}}
    sdata = {**sdata,**{case:temp}}

#%%
if not isinstance(leglab,list):
    leglab = caseName

#%%
c  = list(mcolors.TABLEAU_COLORS.keys())[:len(caseName)]

case_str = ''
for case in caseName:
    case_str = case_str+'_'+case
case_str = case_str[1:]
#%%
#   Loops through each mic
for i,m in enumerate(mics):
    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace=0.35, bottom=0.15)

    for ii,case in enumerate(caseName):

        ax[0].plot(sdata[case]['t_nondim'], sdata[case]['xn_inph'],c = c[ii] ,linestyle=linestyle[ii])
        ax[1].plot(sdata[case]['t_nondim'], (sdata[case]['Xn_avg_filt'][:,m-1]-sdata[case]['xn_inph']),c = c[ii] ,linestyle=linestyle[ii])
        ax[2].plot(sdata[case]['t_nondim'], sdata[case]['Xn_avg_filt'][:,m-1],c = c[ii] ,linestyle=linestyle[ii])

    for ii in range(3):
        if ii!=2:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[ii].set_xlim([0,1])
        # ax[ii].set_ylim([-0.07, .07])
        ax[ii].grid('on')

    ax[0].set_title('In-Phase')
    ax[1].set_title('Out-of-Phase')
    ax[2].set_title('Total')

    ax[1].set_ylabel('Pressure [Pa]')
    plt.suptitle(f'Mic {m}')
    ax[-1].set_xlabel('Rotation')
    ax[-1].legend(leglab, loc='center', ncol=3,bbox_to_anchor=(.5, -.65))


    plt.savefig(os.path.join(save_dir,case_str+ f'_m{m}'+ '_tseries' + '.eps'),format='eps')
    plt.savefig(os.path.join(save_dir,case_str+ f'_m{m}'+ '_tseries' + '.png'),format='png')

#%%
#   Loops through each mic
for i, m in enumerate(mics):
    #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
    plt.subplots_adjust(hspace=0.35, bottom=0.15)

    for ii,case in enumerate(caseName):

        ax[0].stem(sdata[case]["f"]/(sdata[case]["rpm_nom"]/60*sdata[case]["Nb"]), 10*np.log10(sdata[case]['Gxx_inph']*sdata[case]['df']/20e-6**2),linefmt =f'C{ii}{linestyle[ii]}', markerfmt =f'C{ii}^',basefmt=f'C{ii}')
        ax[1].stem(sdata[case]["f"]/(sdata[case]["rpm_nom"]/60*sdata[case]["Nb"]), 10*np.log10(sdata[case]['Gxx_outph']*sdata[case]['df']/20e-6**2)[:,m-1],linefmt =f'C{ii}{linestyle[ii]}', markerfmt =f'C{ii}^',basefmt=f'C{ii}')
        ax[2].stem(sdata[case]["f"]/(sdata[case]["rpm_nom"]/60*sdata[case]["Nb"]), sdata[case]['spl'][:,m-1],linefmt =f'C{ii}{linestyle[ii]}', markerfmt =f'C{ii}^',basefmt=f'C{ii}')

    for ii in range(3):
        if ii != 2:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[ii].axis([0, 4, axis_lim[2], 40])
        ax[ii].set_xticks(np.arange(1, 5))
        ax[ii].set_ylim([axis_lim[-2], axis_lim[-1]])
        ax[ii].grid('on')

    ax[0].set_title('In-Phase')
    ax[1].set_title('Out-of-Phase')
    ax[2].set_title('Total')

    ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    plt.suptitle(f'Mic {m}')
    ax[-1].set_xlabel('BPF Harmonic')
    ax[-1].legend(leglab, loc='center', ncol=3, bbox_to_anchor=(.5, -.65))

    plt.savefig(os.path.join(save_dir,case_str+ f'_m{m}'+ '_spec' + '.eps'),format='eps')
    plt.savefig(os.path.join(save_dir,case_str+ f'_m{m}'+ '_spec' + '.png'),format='png')

#%%
phi = np.arange(360)*np.pi/180
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace=0.35, bottom=0.15)
ax.plot(phi*180/np.pi,np.sin(phi))
ax.plot(phi*180/np.pi,-np.sin(phi)**2)
