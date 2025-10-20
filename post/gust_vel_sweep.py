#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyPostAcsFun import *
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser("rotor_gust_interaction",description='Simulates a gust interacting with a hovering rotor, only the positive half of the gust profile is considered.')

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

    args = parser.parse_args(argv)

    if args.cases is None:
        subdirs = np.asarray(os.listdir())
        args.cases = subdirs[[os.path.isdir(subdir) for subdir in subdirs]]

    if args.legend_labels is None:
        args.legend_labels = args.cases

    V = {}
    for case in args.cases:
        data = import_h5(os.path.join(case, 'acs_data.h5'))
        if "Performance_Data" not in data: 
            apply_fun(case,[],args)
        V.update({case:data['Performance_Data']['Wind Speed (m_s)']})
    

    fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
    plt.subplots_adjust(left=0.15,bottom=0.15)
    for case_itr,case in enumerate(args.cases):
        ax.scatter(int(case),V[case].mean())
    ax.set(xlabel = r'$Throttle \ [%]$',ylabel = r'$V_g [m/s]$',ylim = [0,None],xlim = [0,100])
    ax.grid()
    plt.savefig(f'psd_{"__".join(args.cases)}_m{mic}.png',format = 'png')
    plt.close()



if __name__ == "__main__":
	main()
	print("Exiting main.py")
