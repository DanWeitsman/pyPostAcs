import os
from shutil import rmtree
import h5py
import numpy as np
import pyPostAcsFun as fun
#%%
def apply_fun_to_folder(dir):
    for item in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, item)):
            apply_fun_to_folder(os.path.join(dir, item))

    if os.path.exists(os.path.join(dir, 'acs_data.h5')):
        print('h5 file exists in' + dir)
        if os.path.exists(os.path.join(dir, 'Figures')) is False:
            os.mkdir(os.path.join(dir, 'Figures'))
        with h5py.File(os.path.join(dir, 'acs_data.h5'),'r') as dat_file:
            print(dat_file.keys())
            data = dat_file['Acoustic Data (mV)'][:].transpose()/dat_file['Sensitivities (mV_Pa)']
            f,Gxx = fun.msPSD(data, dat_file.attrs['Sampling Rate (Hz)'], df = 25, win= 'hann', ovr = 0.5)


    else:
        print('h5 file does not exist in' + dir)

apply_fun_to_folder('/Users/danielweitsman/OneDrive - The Pennsylvania State University/January2021TestCampaign/PhasedArrayCalibration')