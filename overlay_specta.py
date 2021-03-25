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
# f1 = lambda a: fun.msPSD(a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3), fs =  a['Sampling Rate'][()], df = 5, win= True, ovr = 0.5, plot = False,save_fig=False,f_lim =[10,10e3],levels = [0, 80])

dir ='/Users/danielweitsman/Desktop/Masters_Research/NASA_Coax_2021/Runs/'
caseName  = ['BG6','Run151']
leglab = ''

mic =5
df = 5
start_t = 8
end_t = 15

fig,ax = plt.subplots(1,1,figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)
for i,case in enumerate(caseName):
    with h5py.File(os.path.join(dir,caseName[i], 'acs_data.h5'), 'r') as dat_file:
        data = (dat_file['Acoustic Data'][:].transpose()/(dat_file['Sensitivities'][:]*1e-3))[int(start_t*dat_file['Sampling Rate'][()]):int(end_t*dat_file['Sampling Rate'][()])]
        f,Gxx,Gxx_avg = fun.msPSD(data[:,int(mic-1)], dat_file['Sampling Rate'][()], df=df, ovr=0.5,plot = False,save_fig = False)
        ax.plot(f,10*np.log10(Gxx_avg*df/20e-6**2),label = case)

ax.set_xscale('log')
ax.axis([10,20e3,0,80])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('$SPL, \: dB\: (re:\: 20 \: \mu Pa)$')
ax.grid('on')

if isinstance(leglab, list):
    ax.legend(leglab)
else:
    ax.legend()

ax.set_title('Mic: ' + str(mic))