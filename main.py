
import pyPostAcsFun as fun

#%%


f1 = lambda a,b: fun.msPSD(a['Acoustic Data (mV)'][:].transpose()/a['Sensitivities (mV_Pa)'], fs =  a.attrs['Sampling Rate (Hz)'], df = 25, win= 'hann', ovr = 0.5, save_path = b)


fun.apply_fun_to_h5('/Users/danielweitsman/OneDrive - The Pennsylvania State University/January2021TestCampaign/PhasedArrayCalibration', [f1])

#todo copy apply_fun_to_h5 over to pyPostAcsFun and add parser. verify spectrum function. Add plotting options to both functions.