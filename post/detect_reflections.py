#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyPostAcsFun import *
import argparse

from scipy.fft import fft,ifft
from scipy.signal import csd


def main(argv=None):
    parser = argparse.ArgumentParser("detect_reflection",description='Computes the autospectrum of each microphone to detect reflections.')

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


    args = parser.parse_args(argv)

    if args.case is None:
         args.case  = os.getcwd()

    pxx = {}
    data = import_h5(os.path.join(args.case, 'acs_data.h5'))
    if "Performance_Data" not in data: 
        apply_fun(args.case,[],args)
        data = import_h5(os.path.join(args.case, 'acs_data.h5'))

    if args.mics is None:
        args.mics = np.arange(len(data['Acoustic Data']))

    def correlation_2(X,Y,fs,auto = True):
        if X.ndim ==1:
             X = X[None]
        N = X.shape[-1]

        if auto:
            Xm = fft(X,axis = -1)
            Sxy = 1 /N * np.conj(Xm) * Xm
        else:
            Sxy = np.zeros(np.insert(len(X),1,Y.shape),dtype = complex)
            for i in range(len(X)):
                Sxy[i] = 1 /N * np.conj(fft(np.roll(X,-i,axis = 0),axis = -1)) * fft(Y,axis = -1)
        
        Rxy = np.real(ifft(Sxy,axis = -1))
        Rxy = (np.concatenate((Rxy[...,int(N/2):],Rxy[...,:int(N/2)]),axis = -1)).squeeze()
        Cxy = (Rxy/(np.sqrt(np.mean(X**2,axis = -1))*np.sqrt(np.mean(Y**2,axis = -1)))[:,None]).squeeze()
        t = (np.arange(N)-N/2)*fs**-1
        return t,Rxy,Cxy
    
    t,Rxy,Cxy = correlation_2(data['Acoustic Data'][args.mics],data['Acoustic Data'][args.mics],data['Sampling Rate'],auto = True)

    for i,mic in enumerate(args.mics):
        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.15,bottom=0.15)
        # ax.plot(t,Cxy[i])
        ax.plot(t,Cxy[i])
        ax.set(xlim = [-0.02,0.02],ylim = [-.75,1],xlabel = r'Time Delay [sec]',ylabel = r"$C_{xy}$")
        ax.grid()
        # ax.legend(args.mics)
        plt.savefig(f'Cxy_m{mic}.png',format = 'png')
        plt.close()

        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.15,bottom=0.15)
        # ax.plot(t,Cxy[i])
        ax.plot(np.arange(len(data['Acoustic Data'][i]))/data['Sampling Rate'],data['Acoustic Data'][i])
        ax.set(xlim = [data['Acoustic Data'][i].argmax()/data['Sampling Rate']-0.005,data['Acoustic Data'][i].argmax()/data['Sampling Rate']+0.02],ylim = [None,None],xlabel = r'Time [sec]',ylabel = r"$P \ [Pa]$")
        ax.grid()
        plt.savefig(f'reflect_tseries_m{mic}.png',format = 'png')
        plt.close()


if __name__ == "__main__":
	main()
	print("Exiting main.py")
