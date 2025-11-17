#!/usr/bin/env python3

import pyPostAcsFun as fun
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser("rotor_gust_interaction",description='Simulates a gust interacting with a hovering rotor, only the positive half of the gust profile is considered.')
    parser.add_argument(
        "--perf_tseries",
		action='store_true',
		help="Plots thrust, torque, and rpm time series",
		default=False,
		required=False
	)
    parser.add_argument(
          '-m','--mics',                
        nargs='+',
        help="Mic (channel) number to evaluate. If ommited applies to all mics.",
		required=False,
        type=int,
    )
    parser.add_argument(
        "--p_tseries",
		action='store_true',
		help="Plot acoustic pressure time series of all the microphones",
		default=False,
		required=False
	)
    parser.add_argument(
        "--psd",
		action='store_true',
		help="Plot power spectral density of all microphones",
		default=False,
		required=False
	)
    parser.add_argument(
        "--spec",
		action='store_true',
		help="Computes and plots the spectrogram of the acoustic measurements for each microphone.",
		default=False,
		required=False
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
        '--start_t',                
        help="Start time for computing the power spectral density in any of the frequency domain analyses.",
		required=False,
        default = 0.0,
        type=float
    )
    parser.add_argument(
        '--end_t',                
        help="End time for computing the power spectral density in any of the frequency domain analyses.",
		required=False,
        type=float
    )
    parser.add_argument(
        '-win','--window',                
        help="Window function for spectra and spectrograms.",
        default= 'hann',
		required=False,
        type=str
    )
    parser.add_argument(
        "--tonal_separation",
		action='store_true',
		help="Separates tonal and broadband noise components based on measured rotational rate (synchronous averaging).",
		default=False,
		required=False
	)
    parser.add_argument(
          '--filter_harmonics',                
        nargs='+',
        help="List of two elements indicating the minimum and maximum shaft order harmonics to retain when performing the tonal noise separation.",
		required=False,
        type=int,
    )
    parser.add_argument(
        "--filter_shaft_order",
		action='store_true',
		help="Include to filter out all shaft order harmonics.",
		default=False,
		required=False
	)

    parser.add_argument(
          '--align',                
        help="Include this flag in order to align the signals corresponding to each rotor revolution before averaging to extract the tonal noise component when performing the tonal separation.",
		required=False,
        action='store_true',
    )

    args = parser.parse_args(argv)
    funcs = []
    if args.perf_tseries:
          funcs.append(lambda a:fun.perf_tseries(a,args) )
    if args.p_tseries:
          funcs.append(lambda a:fun.p_tseries(a,args) )
    if args.spec:
          funcs.append(lambda a:fun.spectrogram(a,args) )
    if args.psd:
          funcs.append(lambda a:fun.psd(a,args) )
    if args.tonal_separation:
          funcs.append(lambda a:fun.tonal_separation(a,args) )

    fun.apply_fun(os.getcwd(),funcs,args)

if __name__ == "__main__":
	main()
	print("Exiting main.py")
