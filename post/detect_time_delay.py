#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyPostAcsFun import *
import argparse
from scipy.signal import find_peaks


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
          '-c','--cases',                
        nargs='+',
        help="Name of cases to compare.",
		required=False,
        type=str,
    )

    parser.add_argument(
          '-l','--legend_labels',                
        nargs='+',
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
    
    if args.cases is None:
        subdirs = np.asarray(os.listdir())
        args.cases = subdirs[[os.path.isdir(subdir) for subdir in subdirs]]

    if args.legend_labels is None:
        args.legend_labels = args.cases
    # sets plot flag to false for tonal separation. This script generates plots regardless.
    args.plot = False

    if not isinstance(args.start_t,list) or not isinstance(args.end_t,list):
        start_t = [args.start_t]*len(args.cases)
        end_t = [args.end_t]*len(args.cases)
    # start_t = args.start_t
    # end_t = args.end_t

    psd = {}
    for i,case in enumerate(args.cases):
        data = import_h5(os.path.join(case, 'acs_data.h5'))
        if "Performance_Data" not in data: 
            apply_fun(case,[],args)
            data = import_h5(os.path.join(case, 'acs_data.h5'))

        if args.end_t is None:
            end_t = [data['Acoustic Data'].shape[-1]/data['Sampling Rate'] for i in end_t if i==None]

        args.start_t = start_t[i]
        args.end_t = end_t[i]
        args.start_ind = int(args.start_t*data['Sampling Rate'])
        args.end_ind = int(args.end_t*data['Sampling Rate'])

        args.nperseg = int(data['Sampling Rate']/args.frequency_resolution)
        t,xn_avg,xn_bb,f_tonal,pxx_tonal,pxx_err,f_bb,pxx_bb = tonal_separation(data,args)
        
        peaks = []
        peak_ind = []
        cross_ind = []
        for mic_itr,mic in enumerate(args.mics):
            peaks_temp,properties = find_peaks(-xn_avg[mic_itr], height=None, threshold=None, distance=None, prominence=(None,None), width=None, wlen=None, rel_height=0.5, plateau_size=None)
            peak_ind.append(peaks_temp[properties['prominences']>properties['prominences'].max()*.75])        
            cross_ind_temp = np.where(np.abs(xn_avg[mic_itr])<=np.sqrt(np.mean(xn_avg[mic_itr]**2))/10)[0]
            cross_ind.append(cross_ind_temp[(cross_ind_temp>peak_ind[mic_itr][0]) & (cross_ind_temp<(peak_ind[mic_itr][-1]-peak_ind[mic_itr][0])*.3+peak_ind[mic_itr][0])])
            peaks.append(peaks_temp)

        psd.update({case:{'peaks':peaks,'peak_ind':peak_ind,'cross_ind':cross_ind,'t':t,'xn_avg':xn_avg,'xn_bb':xn_bb,'f_tonal':f_tonal,'pxx_tonal':pxx_tonal,'pxx_err':pxx_err,'f_bb':f_bb,'pxx_bb':pxx_bb}})
    
    t_shift = [None]*len(args.mics)
    Rxy = [None]*len(args.mics)
    t = [None]*len(args.mics)
    starting_ind = [None]*len(args.mics)
    ending_ind = [None]*len(args.mics)

    fs_avg = np.mean((psd[args.cases[0]]['t'][1]**-1,psd[args.cases[1]]['t'][1]**-1))
    for mic_itr,mic in enumerate(args.mics):

        starting_ind[mic_itr] = [psd[args.cases[0]]['peak_ind'][mic_itr][0],psd[args.cases[1]]['peak_ind'][mic_itr][0]]+np.min((psd[args.cases[0]]['cross_ind'][mic_itr][0]-psd[args.cases[0]]['peak_ind'][mic_itr][0],psd[args.cases[1]]['cross_ind'][mic_itr][0]-psd[args.cases[1]]['peak_ind'][mic_itr][0]))
        ending_ind[mic_itr] = [psd[args.cases[0]]['peak_ind'][mic_itr][0],psd[args.cases[1]]['peak_ind'][mic_itr][0]]+np.max((psd[args.cases[0]]['cross_ind'][mic_itr][-1]-psd[args.cases[0]]['peak_ind'][mic_itr][0],psd[args.cases[1]]['cross_ind'][mic_itr][-1]-psd[args.cases[1]]['peak_ind'][mic_itr][0]))
        t[mic_itr],Rxy[mic_itr],_ =correlation(X = np.concatenate((psd[args.cases[0]]['xn_avg'][mic_itr,starting_ind[mic_itr][0]:ending_ind[mic_itr][0]],np.zeros(ending_ind[mic_itr][0]-starting_ind[mic_itr][0]))),Y = np.concatenate((psd[args.cases[1]]['xn_avg'][mic_itr,starting_ind[mic_itr][1]:ending_ind[mic_itr][1]],np.zeros(ending_ind[mic_itr][1]-starting_ind[mic_itr][1]))),fs =fs_avg ,auto = False)
        t_shift[mic_itr] = t[mic_itr][Rxy[mic_itr].argmax(axis = -1)]


    for mic_itr,mic in enumerate(args.mics):

        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.175,bottom=0.15)
        ax.plot(t[mic_itr],Rxy[mic_itr])
        ax.set(title = rf'$Mic \ {mic}$',xlabel = r'$Time \ Delay \ [s]$',ylabel = r'$R_{xy} \ [Pa^2]$')
        ax.legend(args.legend_labels,prop={'size': 12})
        ax.grid()

    for mic_itr,mic in enumerate(args.mics):

        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.175,bottom=0.15)
        for case_itr,case in enumerate(args.cases):
            ax.plot(psd[case]['t'],psd[case]['xn_avg'][mic_itr],c=np.roll(default_colors,-case_itr)[0], linestyle='-.', label=case)
            ax.plot(psd[case]['t'][starting_ind[mic_itr][case_itr]:ending_ind[mic_itr][case_itr]],psd[case]['xn_avg'][mic_itr,starting_ind[mic_itr][case_itr]:ending_ind[mic_itr][case_itr]],c=np.roll(default_colors,-case_itr)[0], label=case)

        ax.set(title = rf'$Mic \ {mic}$',xlabel = r'$Rotation$',ylabel = r'$Pressure \ [Pa]$')
        ax.legend(args.legend_labels,prop={'size': 12})
        ax.grid()

    for mic_itr,mic in enumerate(args.mics):

        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.175,bottom=0.15)
        for case_itr,case in enumerate(args.cases):
            ax.plot(np.arange(ending_ind[mic_itr][case_itr]-starting_ind[mic_itr][case_itr])*fs_avg**-1,psd[case]['xn_avg'][mic_itr,starting_ind[mic_itr][case_itr]:ending_ind[mic_itr][case_itr]], label=case)
        
        ax.plot(np.arange(ending_ind[mic_itr][case_itr]-starting_ind[mic_itr][case_itr])*fs_avg**-1-t_shift[mic_itr],psd[case]['xn_avg'][mic_itr,starting_ind[mic_itr][case_itr]:ending_ind[mic_itr][case_itr]], label=case)

        ax.set(title = rf'$Mic \ {mic}$',xlabel = r'$Rotation$',ylabel = r'$Pressure \ [Pa]$')
        ax.legend(args.legend_labels,prop={'size': 12})
        ax.grid()











    # for mic_itr,mic in enumerate(args.mics):
    #     fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
    #     plt.subplots_adjust(left=0.15,bottom=0.15)
    #     for case_itr,case in enumerate(args.cases):
    #         if args.tonal_separation:
    #             ax.errorbar(psd[case]['f_tonal'], 10*np.log10(psd[case]['pxx_tonal'][mic_itr]*np.diff(psd[case]['f_tonal'][:2])[0]/20e-6**2), yerr=10*np.log10(psd[case]['pxx_err'][mic_itr]), fmt='o',color = default_colors[case_itr],ecolor = default_colors[case_itr],capsize=6,capthick=2,zorder =len(args.cases)-case_itr-1)
    #             # scatter= ax.scatter(psd[case]['f_tonal'],10*np.log10(psd[case]['pxx_tonal'][mic_itr]*np.diff(psd[case]['f_tonal'][:2])[0]/20e-6**2),label="_nolegend_",color = default_colors[case_itr])
    #         line= ax.plot(psd[case]['f'],10*np.log10(psd[case]['pxx'][mic_itr]*np.diff(psd[case]['f'][:2])[0]/20e-6**2),zorder =len(args.cases)-case_itr-1)
    #         line[0].set(color=np.roll(default_colors,-case_itr)[0], linestyle=np.roll(linestyle,-case_itr)[0], label=case)
    #     ax.set(xlabel = r'$Frequency \ [Hz]$',ylabel = r'$SPL, \ dB \ (re: \ 20 \mu Pa)$',ylim = [0,100],xscale = 'log',xlim = [10,10e3],title = f"Mic {mic}")
    #     # ax.legend(args.legend_labels,borderpad=0.2,handlelength=1,handletextpad=0.3,columnspacing=1.2,loc='lower center',ncol = len(args.legend_labels), bbox_to_anchor=(.5, -0.225),prop={'size': 10})
    #     ax.legend(args.legend_labels,prop={'size': 12})
    #     ax.grid()
    #     plt.savefig((f'psd_{"__".join(args.cases)}_m{mic}.png').replace(os.sep,'__'),format = 'png')
    #     plt.close()

    #     if args.tonal_separation:
    #         fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
    #         plt.subplots_adjust(left=0.175,bottom=0.15)
    #         for case_itr,case in enumerate(args.cases):
    #             ax.plot(psd[case]['t']/psd[case]['t'][-1],psd[case]['xn_avg'][mic_itr],c=np.roll(default_colors,-case_itr)[0], linestyle=np.roll(linestyle,-case_itr)[0], label=case)
    #         ax.set(title = rf'$Mic \ {mic}$',xlabel = r'$Rotation$',ylabel = r'$Pressure \ [Pa]$')
    #         ax.legend(args.legend_labels,prop={'size': 12})
    #         ax.grid()
    #         plt.savefig((f'p_tseries_{"__".join(args.cases)}_m{mic}.png').replace(os.sep,'__'),format = 'png')
    #         plt.close()

if __name__ == "__main__":
	main()
	print("Exiting main.py")
