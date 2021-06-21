
import pyPostAcsFun as fun
import numpy as np
#%%
mic = 9
#   prefix to path containing cases
dir = '/Users/danielweitsman/Box/May21Test/rpm_sweep/ushb'
#   functions to apply to all cases contained in the folder specified as "dir"
f1 = lambda a,b: fun.msPSD(a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3), fs =  a['Sampling Rate'][()], df = 5, win= True, ovr = 0.5, save_path = b,save_fig=True,plot = True,f_lim =[10,10e3],levels = [0, 100])
f2 = lambda a,b: fun.tseries((a['Acoustic Data (mV)'][:].transpose()/(a['Sensitivities'][:]*1e-3)), fs =  a['Sampling Rate'][()],t_lim = [0,10], levels = [-0.25,0.25], save_path = b,save_fig=True)
f3 = lambda a,b: fun.spectrogram((a['Acoustic Data'][mic,:].transpose()/(a['Sensitivities'][mic]*1e-3)),fs=  a['Sampling Rate'][()],df = 10,ovr = 0.5,win = True,save_path = b,save_fig=False,plot=True,t_lim = [0,20],f_lim =[10,2e3],levels = np.arange(0,70,2))

#   applies functions to all data contained in "dir". Desired functions must be specified as a comma delimited list.
fun.apply_fun_to_h5(dir, [] , append_perf = True)

