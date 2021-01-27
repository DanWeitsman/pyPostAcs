
import pyPostAcsFun as fun

#%%


f1 = lambda a,b: fun.msPSD(a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3), fs =  a['Sampling Rate'][()], df = 5, win= True, ovr = 0.5, save_path = b,save_fig=True,f_lim =[10,10e3],levels = [0, 80])
f2 = lambda a,b: fun.spectrogram(a['Acoustic Data'][:].transpose()/(a['Sensitivities'][:]*1e-3),fs=  a['Sampling Rate'][()],df = 25,ovr = 0.5,win = True,save_path = b,save_fig=False,f_lim =[10,10e3],levels = [0, 80])

fun.apply_fun_to_h5('/Users/danielweitsman/OneDrive - The Pennsylvania State University/January2021TestCampaign/Runs/1_26_21', [f1])


