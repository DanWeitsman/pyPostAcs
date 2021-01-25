
import pyPostAcsFun as fun

#%%


f1 = lambda a,b: fun.msPSD(a['Acoustic Data (mV)'][:,int(a.attrs['Sampling Rate (Hz)']*0):].transpose()/44.3e-3, fs =  a.attrs['Sampling Rate (Hz)'], df = 25, win= True, ovr = 0.5, save_path = b,save_fig=False,axis_lim=[10,10e3,20,110])
f2 = lambda a,b: fun.spectrogram(a['Acoustic Data (mV)'][:].transpose()/44.3e-3,fs=  a.attrs['Sampling Rate (Hz)'],df = 25,ovr = 0.5,win = True,save_path = b,save_fig=False)

fun.apply_fun_to_h5('/Users/danielweitsman/OneDrive - The Pennsylvania State University/January2021TestCampaign/CalibrationMicrophones/22Jan2021/Mic4', [f2])

#todo copy apply_fun_to_h5 over to pyPostAcsFun and add parser. verify spectrum function. Add plotting options to both functions.