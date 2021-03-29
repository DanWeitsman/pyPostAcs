import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt

#%%
fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
#   parent directory containing the points of interest
dir ='/Users/danielweitsman/Box/Jan21Test/TAMU/runs/rpm_sweeps'
#   names of points you want to compare
caseName  = ['cshb','cghb']
#   legend labels, if equated to an empty string (''), the caseName would be used as the legend labels
leglab = ['Straight','Gofly']

linestyle = ['-.','-']

fig_T,ax_T = plt.subplots(1,1,figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)

fig_Q,ax_Q = plt.subplots(1,1,figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)

for i,case in enumerate(caseName):
    with h5py.File(os.path.join(dir,caseName[i]+'.h5'), 'r') as dat_file:
        ax_T.errorbar(dat_file['rpm_avg_2'], dat_file['T_tot'], yerr=dat_file['T_tot_err'], fmt= linestyle[i]+'o')
        ax_Q.errorbar(dat_file['rpm_avg_2'], dat_file['Q_tot'], yerr=dat_file['Q_tot_err'], fmt=linestyle[i]+'o')

ax_T.set_title(' vs. '.join(caseName))
ax_T.set_ylabel('Thrust, T (N)')
ax_T.set_xlabel('RPM')
ax_T.set_xlim([1000, 5250])
ax_T.set_ylim([0, 70])
ax_T.grid()

ax_Q.set_title(' vs. '.join(caseName))
ax_Q.set_ylabel('Torque, Q (Nm)')
ax_Q.set_xlabel('RPM')
ax_Q.set_xlim([1000, 5250])
ax_Q.set_ylim([-1.5, 1.5])
ax_Q.grid()

if isinstance(leglab, list):
    ax_T.legend(leglab)
    ax_Q.legend(leglab)
else:
    ax_T.legend()
    ax_Q.legend()

fig_T.savefig(os.path.join(dir, 'Figures', '_'.join(caseName)+ '_T.eps'), format='eps')
fig_Q.savefig(os.path.join(dir, 'Figures', '_'.join(caseName) +'_Q.eps'), format='eps')
