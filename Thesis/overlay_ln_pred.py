import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/'
save_dir = '/Users/danielweitsman/Desktop/Masters_Research/Thesis/Analysis/9_15_21'
# If you want to compare specific points in this directory their names can be specified in caseName list. Otherwise,
# all the cases in this directory directory would be compared and used to generate the thrust/torque profiles.
caseName = ['e2b93','e2b92']
leglab = ['Collective trim','Cyclic trim']
save_fig = True
#   Linestyle for each case
linestyle =['-','-.','--','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,9]

#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [50, 1e3, 0, 60]


#%%
sdata = {}
#   imports several performance quantities from the MainDict.h5 file.
for case in caseName:
    with h5py.File(os.path.join(os.path.dirname(dir), case, case + '_sdata.h5'),'r') as f:
        temp = {}
        for k, dat in f.items():
            temp = {**temp,**{k:dat[()]}}
    sdata = {**sdata,**{case:temp}}
#%%

case_str = ''
for case in caseName:
    case_str = case_str+'_'+case
case_str = case_str[1:]

if not isinstance(leglab,list):
    leglab = caseName
#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(2,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

for i,case in enumerate(caseName):

    # ax[0].stem(sdata[case]['f_pred'],sdata[case]['spl_pred'][mics[0]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[0].stem(sdata[case]['f']/(sdata[case]['omega']/60*sdata[case]['Nb']),10*np.log10(np.squeeze(sdata[case]['Gxx_avg_outph'][:,mics[0]-1]) * sdata[case]['df']/20e-6**2),linefmt =f'C{i}{linestyle[i]}', markerfmt =f'C{i}o',basefmt=f'C{i}')

    ax[0].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[0].set_xticks(np.arange(1, 5))

    ax[0].tick_params(axis='x', labelsize=0)
    ax[0].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[0].set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(sdata[case]["phi"][mics[0]-1])}^\circ)$')
    ax[0].grid('on')

    # ax[1].stem(sdata[case]['f_pred'],sdata[case]['spl_pred'][mics[-1]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[1].stem(sdata[case]['f']/(sdata[case]['omega']/60*sdata[case]['Nb']),10*np.log10(np.squeeze(sdata[case]['Gxx_avg_outph'][:,mics[-1]-1]) * sdata[case]['df']/20e-6**2),linefmt =f'C{i}{linestyle[i]}', markerfmt =f'C{i}o',basefmt=f'C{i}')

    ax[1].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[1].set_xticks(np.arange(1, 5))

    ax[1].set_title(f'$Mic\ {mics[-1]} \ ( \phi = {round(sdata[case]["phi"][mics[-1]-1])}^\circ)$')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[1].grid('on')
    ax[1].legend(leglab, loc='center', ncol=2, bbox_to_anchor=(.5, -.35))
    plt.suptitle('Out-of-Phase Loading')

    plt.savefig(os.path.join(save_dir,case_str+  '_outph_spec' + '.eps'),format='eps')
    plt.savefig(os.path.join(save_dir,case_str + '_outph_spec' + '.png'),format='png')

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(2,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

for i,case in enumerate(caseName):

    # ax[0].stem(sdata[case]['f_pred'],sdata[case]['spl_pred'][mics[0]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[0].stem(sdata[case]['f']/(sdata[case]['omega']/60*sdata[case]['Nb']),10*np.log10(np.squeeze(sdata[case]['Gxx_avg_inph_load'][:,mics[0]-1]) * sdata[case]['df']/20e-6**2),linefmt =f'C{i}{linestyle[i]}', markerfmt =f'C{i}o',basefmt=f'C{i}')

    ax[0].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[0].tick_params(axis='x', labelsize=0)
    ax[0].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[0].set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(sdata[case]["phi"][mics[0]-1])}^\circ)$')
    ax[0].grid('on')
    ax[0].set_xticks(np.arange(1, 5))

    # ax[1].stem(sdata[case]['f_pred'],sdata[case]['spl_pred'][mics[-1]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[1].stem(sdata[case]['f']/(sdata[case]['omega']/60*sdata[case]['Nb']),10*np.log10(np.squeeze(sdata[case]['Gxx_avg_inph_load'][:,mics[-1]-1]) * sdata[case]['df']/20e-6**2),linefmt =f'C{i}{linestyle[i]}', markerfmt =f'C{i}o',basefmt=f'C{i}')

    ax[1].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[1].set_xticks(np.arange(1, 5))

    ax[1].set_title(f'$Mic\ {mics[-1]} \ ( \phi = {round(sdata[case]["phi"][mics[-1]-1])}^\circ)$')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[1].grid('on')
    ax[1].legend(leglab, loc='center', ncol=2, bbox_to_anchor=(.5, -.35))
    plt.suptitle('In-Phase Loading')

    plt.savefig(os.path.join(save_dir,case_str+  '_inph_spec' + '.eps'),format='eps')
    plt.savefig(os.path.join(save_dir,case_str + '_inph_spec' + '.png'),format='png')


    # %%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace=0.35, bottom=0.15)

for i, case in enumerate(caseName):
    # ax[0].stem(sdata[case]['f_pred'],sdata[case]['spl_pred'][mics[0]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[0].stem(sdata[case]['f'] / (sdata[case]['omega'] / 60 * sdata[case]['Nb']),
               10 * np.log10(
                   np.squeeze(sdata[case]['Gxx_avg_pred'][mics[0] - 1, :, 1]) * sdata[case]['df'] / 20e-6 ** 2),
               linefmt=f'C{i}{linestyle[i]}', markerfmt=f'C{i}o', basefmt=f'C{i}')

    ax[0].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[0].set_xticks(np.arange(1, 5))

    ax[0].tick_params(axis='x', labelsize=0)
    ax[0].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[0].set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(sdata[case]["phi"][mics[0] - 1])}^\circ)$')
    ax[0].grid('on')

    # ax[1].stem(sdata[case]['f_pred'],sdata[case]['spl_pred'][mics[-1]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[1].stem(sdata[case]['f'] / (sdata[case]['omega'] / 60 * sdata[case]['Nb']),
               10 * np.log10(
                   np.squeeze(sdata[case]['Gxx_avg_pred'][mics[-1] - 1, :, 1]) * sdata[case]['df'] / 20e-6 ** 2),
               linefmt=f'C{i}{linestyle[i]}', markerfmt=f'C{i}o', basefmt=f'C{i}')

    ax[1].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[1].set_xticks(np.arange(1, 5))

    ax[1].set_title(f'$Mic\ {mics[-1]} \ ( \phi = {round(sdata[case]["phi"][mics[-1] - 1])}^\circ)$')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[1].grid('on')
    ax[1].legend(leglab, loc='center', ncol=2, bbox_to_anchor=(.5, -.35))
    plt.suptitle('Total Loading')

    plt.savefig(os.path.join(save_dir,case_str+  '_tot_load_spec' + '.eps'),format='eps')
    plt.savefig(os.path.join(save_dir,case_str + '_tot_load_spec' + '.png'),format='png')

    # %%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace=0.35, bottom=0.15)

for i, case in enumerate(caseName):
    # ax[0].stem(sdata[case]['f_pred'],sdata[case]['spl_pred'][mics[0]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[0].stem(sdata[case]['f']/(sdata[case]['omega']/60*sdata[case]['Nb']),
               10 * np.log10(np.squeeze(sdata[case]['Gxx_avg_pred'][mics[0]-1,:,-1]) * sdata[case]['df'] / 20e-6 ** 2),
               linefmt=f'C{i}{linestyle[i]}', markerfmt=f'C{i}o', basefmt=f'C{i}')

    ax[0].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[0].set_xticks(np.arange(1, 5))

    ax[0].tick_params(axis='x', labelsize=0)
    ax[0].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[0].set_title(f'$Mic\ {mics[0]} \ ( \phi = {round(sdata[case]["phi"][mics[0] - 1])}^\circ)$')
    ax[0].grid('on')

    # ax[1].stem(sdata[case]['f_pred'],sdata[case]['spl_pred'][mics[-1]-1,:,1],linefmt =f'C{0}{linestyle[0]}', markerfmt =f'C{0}o',basefmt=f'C{0}')
    ax[1].stem(sdata[case]['f']/(sdata[case]['omega']/60*sdata[case]['Nb']),
               10 * np.log10(np.squeeze(sdata[case]['Gxx_avg_pred'][mics[-1]-1,:,-1]) * sdata[case]['df'] / 20e-6 ** 2),
               linefmt=f'C{i}{linestyle[i]}', markerfmt=f'C{i}o', basefmt=f'C{i}')

    ax[1].axis([0, 4, axis_lim[2], axis_lim[-1]])
    ax[1].set_xticks(np.arange(1, 5))

    ax[1].set_title(f'$Mic\ {mics[-1]} \ ( \phi = {round(sdata[case]["phi"][mics[-1] - 1])}^\circ)$')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax[1].grid('on')
    ax[1].legend(leglab, loc='center', ncol=2, bbox_to_anchor=(.5, -.35))
    plt.suptitle('Total')

    plt.savefig(os.path.join(save_dir,case_str+  '_tot_spec' + '.eps'),format='eps')
    plt.savefig(os.path.join(save_dir,case_str + '_tot_spec' + '.png'),format='png')
