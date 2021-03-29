
import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt

#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
#   Parent directory where all of the data files are contained.
dir ='/Users/danielweitsman/Box/Jan21Test/TAMU/runs/'

# Names of the sub folders that correspond to the cases you want to compare. If the case folder is in another
# subfolder the relative path from the parent directory ('dir') can be included in each of the folders of CaseName
caseName  = ['bg/bg14','mn/mn6','cshb/cshb18','cghb/cghb20']

#   set equal to True in order to save the figure as an eps
save = False
#   Legend labels, if set equal to '', the contents of CaseName would be used to generate the legend.
leglab = ['Background','Motor Noise','Straight','GoFly']
# leglab=''

#   Linestyle for each case
linestyle =['-','-.','--','-']

#   Mic #'s that you want to plot and compare. A subplot will be generated for each mic.
mics = [1,5,9]

#   Frequency resolution of spectra [Hz]
df = 5
#   Axis limits specified as: [xmin,xmax,ymin,ymax]
axis_lim = [30, 5e3, 0, 85]

#   Starting time from which to compute the spectra
start_t = 10
#   End time to which to compute the spectra
end_t = 15

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through the cases
for i,case in enumerate(caseName):
#   Opens and reads in the acoustic data in the h5 file
    with h5py.File(os.path.join(dir,case, 'acs_data.h5'), 'r') as dat_file:
        data = (dat_file['Acoustic Data'][:].transpose()/(dat_file['Sensitivities'][:]*1e-3))[int(start_t*dat_file['Sampling Rate'][()]):int(end_t*dat_file['Sampling Rate'][()])]
#   Loops through each mic
        for ii,m in enumerate(mics):
#   Computes the mean-square PSD spectra for each mic
            f,Gxx,Gxx_avg = fun.msPSD(data[:,int(m-1)], dat_file['Sampling Rate'][()], df=df, ovr=0.5,plot = False,save_fig = False)
#   Plots the resulting spectra in dB
            if len(mics)>1:
                ax[ii].plot(f,10*np.log10(Gxx_avg*df/20e-6**2),label = case,linestyle=linestyle[i])
            else:
                ax.plot(f,10*np.log10(Gxx_avg*df/20e-6**2),label = case,linestyle=linestyle[i])

#   Configures axes, plot, and legend if several mics are plotted
if len(mics)>1:
    for ii, m in enumerate(mics):
        ax[ii].set_title('Mic: '+str(m))
        if ii!=len(mics)-1:
            ax[ii].tick_params(axis='x', labelsize=0)
        ax[ii].set_xscale('log')
        ax[ii].set_yticks(np.arange(0,axis_lim[-1],20))
        ax[ii].axis(axis_lim)
        ax[ii].grid('on')

    ax[len(mics) - 1].set_xlabel('Frequency (Hz)')
    ax[int((len(mics) - 1)/2)].set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')

    if isinstance(leglab, list):
        ax[len(mics) - 1].legend(leglab,loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.625))
    else:
        ax[len(mics) - 1].legend(loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.625))

#   Configures axes, plot, and legend if only a single mic is plotted
else:
    ax.set_title('Mic: ' + str(mics[0]))
    ax.set_xscale('log')
    ax.axis(axis_lim)
    ax.set_yticks(np.arange(0, axis_lim[-1], 20))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
    ax.grid('on')

    if isinstance(leglab, list):
        ax.legend(leglab,loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.175))
    else:
        ax.legend(loc='center',ncol = len(caseName), bbox_to_anchor=(.5, -.175))

if save:
    if not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures')):
        os.mkdir(os.path.join(os.path.dirname(dir), 'Figures'))
    plt.savefig(os.path.join(dir, 'Figures', 'hover_spectra.eps'), format='eps')
