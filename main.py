
import pyPostAcsFun as fun

#%%


f1 = lambda a,b: fun.msPSD(a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3), fs =  a['Sampling Rate'][()], df = 5, win= True, ovr = 0.5, save_path = b,save_fig=True,f_lim =[10,10e3],levels = [0, 80])
f2 = lambda a,b: fun.tseries((a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3)), fs =  a['Sampling Rate'][()],t_lim = [3.1,3.25], levels = [0,0.4], save_path = b)
f3 = lambda a,b: fun.spectrogram((a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3)),fs=  a['Sampling Rate'][()],df = 25,ovr = 0.5,win = True,save_path = b,save_fig=True,t_lim = [1,4],f_lim =[10,5e3],levels = [10, 60])

fun.apply_fun_to_h5('/Users/danielweitsman/OneDrive - The Pennsylvania State University/January2021TestCampaign/Runs/2_3_21/test', [f2])


