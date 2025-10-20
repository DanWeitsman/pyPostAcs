#!/usr/bin/env python3

"""
Filename: mic_calibration.py
Author: Daniel Weitsman
Date: 2025-09-11
Description: This script is used to calibrate the microphones in the anechoic chamber in the Hammond bldg. It expects that all the raw calibration data files are saved in the same directory. 
The calibration sensitivities are computed in the time-domain. First, the data is bandpass filtered and the sensitivities are computed only over the duration when the calibration signal is steady. 
This script supports both pistonphone and handheld calibrators. The calibration can be performed with all mic channels enabled in the acosutic data acquisition vi, this script detects which channel 
is being calibrated automatically.  
"""

import numpy as np
import os
from scipy.signal import butter,lfilter
import pyPostAcsFun as fun
import json
import argparse
import matplotlib.pyplot as plt

def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--correction", default=0,type=float, \
        help="Barometric pressure correction factor to add to pistonephone calibration [dB]")
    parser.add_argument("-pist", "--pistonphone", action="store_true", \
            help="Include when using a pistonphone calibrator")
    args = parser.parse_args(argv)

    # computes the rolling average of the dataset given a window size
    def rolling_avg(x, window):
        cumsum = np.cumsum(np.insert(x, 0, 0.0))
        mean = (cumsum[window:] - cumsum[:-window]) / window
        return mean

    # returns all files in the current directory
    files_in_dir = np.asarray(os.listdir(os.getcwd()))
    # determines and retains only the ones that are folders
    f_name =files_in_dir[[os.path.isdir(path) for path in files_in_dir]]
    # temporal resolution [sec] for detecting when the calibrator is on the mic without any significant movement
    t_int = 0.01

    # selects the calibration frequency and sound pressure level depending on whether a pistonphone or handheld calibrator is used. 
    if args.pistonphone:
        cal_freq = 250
        cal_db = 124.03
    else:
        cal_freq = 1e3
        cal_db = 94

    #%%

    # imports the first case to determine channel count and preallocates arrays for storing sensitivities 
    data = fun.import_h5(os.path.join(os.getcwd(),f_name[0],f"acs_data.h5"))
    N_channels = len(data['Acoustic Data'])
    sens = np.zeros(N_channels)
    keys = [0]*N_channels

    # loops through each case in the folder and computes the sensitivity
    for file in f_name: 

        data = fun.import_h5(os.path.join(os.getcwd(),file,f"acs_data.h5"))
        
        # detects channel that is being calibrated
        if N_channels> 1:
            channel_ind = np.sqrt(np.mean(data['Acoustic Data']**2,axis = -1)).argmax()
        else:
             channel_ind = 0

        # applies 2nd order butterworth bandpass filter to channel being calibrated about the calibration frequency
        b,a = butter(2, Wn =[cal_freq-25,cal_freq+25], btype='bandpass', analog=False, output='ba',fs = data['Sampling Rate'])
        filt_data = lfilter(b,a, data['Acoustic Data'][channel_ind])

        # determines the window length 
        pnts_per_record = int(t_int*data['Sampling Rate'])
        
        # computes the rolling rms and the std of the rms for each window
        rms = np.sqrt(rolling_avg(filt_data**2, pnts_per_record))
        rms_std = np.sqrt(rolling_avg(rms**2, pnts_per_record)-rolling_avg(rms, pnts_per_record)**2)

        # determines the index corresponding to when the calibrater signal is steady. This is defined by the point where the std of the rms of the signal is twice it's final "steady state" value. 
        start_ind = len(rms)-np.where(rms_std[::-1]>4*rms_std[-1])[0][0]
        # calculates sensitivity and defines the corresponding channel key
        sens[channel_ind] = np.round(np.sqrt(np.mean(filt_data[start_ind:]**2))/(10**((cal_db+args.correction)/20)*20e-6)*1e3,4)
        keys[channel_ind] = f"Channel {channel_ind}"

        # plt.plot(rms,linewidth = 0.4)
        # plt.scatter(start_ind,rms[start_ind])
        # plt.show()

    # writes out sensitivities to a json file 
    with open(os.path.join(os.getcwd(),'mic_sens.json'),'w') as f:
        json.dump(dict(zip(keys, sens)),f,indent=2)

if __name__ == "__main__":
	main()
	print("Exiting main.py")


