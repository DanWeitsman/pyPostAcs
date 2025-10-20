#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyPostAcsFun import *
import argparse

#%%
def main(argv=None):
    parser = argparse.ArgumentParser("rotor_gust_interaction",description='Simulates a gust interacting with a hovering rotor, only the positive half of the gust profile is considered.')
    parser.add_argument(
          '-m','--mics',                
        nargs='+',
        help="Mic (channel) number to evaluate. If ommited applies to all mics.",
		required=False,
        type=int,
    )
    parser.add_argument(
          '-c','--case',                
        help="Name of cases to compare.",
		required=False,
        type=str,
    )

    parser.add_argument(
        '-df','--frequency_resolution',                
        help="Frequency resolution [Hz] for spectra and spectrograms.",
		required=False,
        type=float
    )
    parser.add_argument(
        '-ovr','--overlap',                
        help="Percentage overlap between sequential records for spectra and spectrograms.",
		required=False,
        default = 0.0,
        type=float
    )

    parser.add_argument(
        '-win','--window',                
        help="Window function for spectra and spectrograms.",
        default= 'boxcar',
		required=False,
        type=str
    )
    parser.add_argument(
        '--start_t',     
        nargs='+',           
        help="Start time for computing spectral density. Can be different for each case just provide sequentially as command line argument for each case. ",
        default= 0.0,
		required=False,
        type=float
    )
    parser.add_argument(
        '--end_t',                
        nargs='+',           
        help="End time for computing spectral density. Can be different for each case just provide sequentially as command line argument for each case",
		required=False,
        type=float
    )
    parser.add_argument(
        '--tonal_separation',                
        help="Include to also perform synchronous averaging to extract tonal components",
        default=False,
		required=False,
		action='store_true'
        )
    parser.add_argument(
          '--filter_harmonics',                
        nargs='+',
        help="List of two elements indicating the minimum and maximum shaft order harmonics to retain when performing the tonal noise separation.",
		required=False,
        type=int,
    )
    parser.add_argument(
          '--align',                
        help="Include this flag in order to align the signals corresponding to each rotor revolution before averaging to extract the tonal noise component when performing the tonal separation.",
		required=False,
        action='store_true',
    )


    args = parser.parse_args(argv)

    if args.case is None:
        args.case = os.getcwd()

    # sets plot flag to false for tonal separation. This script generates plots regardless.
    args.plot = False

    data = import_h5(os.path.join(args.case, 'acs_data.h5'))
    if "Performance_Data" not in data: 
        apply_fun(args.case,[],args)

    if args.end_t is None:
        args.end_t = data['Acoustic Data'].shape[-1]/data['Sampling Rate'] 

    start_ind = int(args.start_t*data['Sampling Rate'])
    end_ind = int(args.end_t*data['Sampling Rate'])

    nperseg = int(data['Sampling Rate']/args.frequency_resolution)

    fs_perf = np.diff(data['Performance_Data']['Time (s)']).mean()**-1
    nperseg_perf = fs_perf/args.frequency_resolution
    start_ind_accel = int(args.start_t*fs_perf)
    end_ind_accel = int(args.end_t*fs_perf)

    f,pxx = welch(data['Acoustic Data'][args.mics,start_ind:end_ind], fs=data['Sampling Rate'], window=args.window, nperseg=nperseg, noverlap=int(args.overlap*nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    f_accel,pxx_accel = welch(data['Performance_Data']['Acceleration_Ch2(m_s2)'][start_ind_accel:end_ind_accel], fs=fs_perf, window=args.window, nperseg=nperseg_perf, noverlap=int(args.overlap*nperseg_perf), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

    for mic_itr,mic in enumerate(args.mics):
        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.15,bottom=0.15)

        ax.plot(f,10*np.log10(pxx[mic_itr]/20e-6**2),linestyle = '-')
        ax.plot(f_accel,10*np.log10(pxx_accel/10e-8),linestyle = '-.')
        ax.set(xlabel = r'$Frequency \ [Hz]$',ylabel = r'$PSD, \ dB/Hz \ (re: \ 20 \mu Pa)$',ylim = [0,None],xlim = [10,1e3],title = f"Mic {mic}")
        # ax.legend(args.legend_labels,borderpad=0.2,handlelength=1,handletextpad=0.3,columnspacing=1.2,loc='lower center',ncol = len(args.legend_labels), bbox_to_anchor=(.5, -0.225),prop={'size': 10})
        ax.legend(['Acoustic Data', 'Accelerometer Data'],prop={'size': 12})
        ax.grid()
        # plt.savefig((f'psd_{"__".join(args.cases)}_m{mic}.png').replace(os.sep,'__'),format = 'png')
        # plt.close()

if __name__ == "__main__":
	main()
	print("Exiting main.py")
