
import pyPostAcsFun as fun

#%%
#   prefix to path containing cases
dir = '/Users/danielweitsman/Box/Jan21Test/TAMU/runs/cgeb'
#   functions to apply to all cases contained in the folder specified as "dir"
f1 = lambda a,b: fun.msPSD(a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3), fs =  a['Sampling Rate'][()], df = 5, win= True, ovr = 0.5, save_path = b,save_fig=True,plot = True,f_lim =[10,10e3],levels = [0, 100])
f2 = lambda a,b: fun.tseries((a['Acoustic Data (mV)'][:].transpose()/(a['Sensitivities'][:]*1e-3)), fs =  a['Sampling Rate'][()],t_lim = [0,10], levels = [-0.25,0.25], save_path = b,save_fig=True)
f3 = lambda a,b: fun.spectrogram((a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3)),fs=  a['Sampling Rate'][()],df = 25,ovr = 0.5,win = True,save_path = b,save_fig=True,plot=True,t_lim = [0,8.5],f_lim =[10,2.5e3],levels = [0, 100])

#   applies functions to all data contained in "dir". Desired functions must be specified as a comma delimited list.
fun.apply_fun_to_h5(dir, [f1,f2] ,append_perf = True)

