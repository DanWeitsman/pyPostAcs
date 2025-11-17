#!/usr/bin/env python3

'''
Acoustic Post Processing Functions
By Daniel Weitsman
1/23/21
'''

import os
import numpy as np
from scipy.fft import fft,ifft
from scipy.signal import lfilter,welch,windows,ShortTimeFFT,csd,butter
from scipy.signal.windows import get_window

import matplotlib.pyplot as plt
import h5py
import re
from bisect import bisect

#%%
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Times New Roman"]
plt.rcParams['font.size'] = 16
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.']
#%%


def import_h5(dir):
    
    def h5_to_dict(h5_obj):
        """
        Recursively converts an HDF5 file/group into a nested dictionary.

        Args:
            h5_obj (h5py.File or h5py.Group): HDF5 file or group object.

        Returns:
            dict: Nested dictionary representation of the HDF5 structure.
        """
        h5_dict = {}
        for key,value in h5_obj.items():
            if isinstance(value, h5py.Group):
                # Recursively process groups
                h5_dict.update({key:h5_to_dict(value)})
            else:
                if isinstance(value[()], bytes):
                    h5_dict.update({key:value[()].decode()})
                else:
                    h5_dict.update({key:value[()]})
        return h5_dict

    with h5py.File(dir, 'r+') as f:
        data = h5_to_dict(f)
    return data

def write_h5(dir, data):
    """
    Exports a nested dictionary to an HDF5 file.

    Args:
        data (dict): The nested dictionary to export.
        file_path (str): Path to the HDF5 file to save.
    """
    def dict_to_h5(h5_group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = h5_group.create_group(key)
                dict_to_h5(subgroup, value)
            else:
                if isinstance(value, np.ndarray):
                #     value = np.asarray([value])
                    h5_group.create_dataset(key, data=value,        
                    compression="gzip",     # compression filter
                    compression_opts=1,     # level (depends on filter)
                    shuffle=True,           # optional filter for better compression
                    fletcher32=True)         # checksum for error detection
                else:
                    h5_group.create_dataset(key, data=value)

    with h5py.File(dir, 'w') as f:
        dict_to_h5(f, data)



def apply_fun(dir,function,args):

    '''
    This function finds all the acs_data.h5 files contained in a directory and all of its subdirectories and applies
    the specified functions to the data set contained in each h5 file.
    :param dir: The parent directory within which to search for the acs_data.h5 files
    to search for the h5
    :param function: A list of functions which to run on each of the detected acs_data.h5 files.
    :param append_perf: Boolean (true/false) if you want to write the load contained in full.txt to the h5 file.

    :return:
    '''

    #   Checks to ensure that functions are provided in a list
    assert isinstance(function, list), 'Functions must be inputed to this function in a list'

    #   obtaines all the subdirectories in dir
    subdir = os.listdir(dir)
    # if there are subfolders, this function recursively searches through each subfolder until the base is found which contains the acs_data.h5 file
    if 'acs_data.h5' not in subdir:
        for item in subdir:
            if os.path.isdir(os.path.join(dir, item)):
                apply_fun(os.path.join(dir, item), function,args)
    else:
        print(f'h5 file exists in {dir}')
        # imports data from the h5 file
        data = import_h5(os.path.join(dir, 'acs_data.h5'))
        
        # if function list is not empty declare the following args, otherwise just append performance data
        if function:
            # if mic numbers are not explicitly provided as an arguments selects all mics
            if args.mics is None:
                args.mics = np.arange(len(data['Acoustic Data'])-1)

            # if frequency resolution is not explicitly provided as an arguments it is set to the narrowband frequency resolution
            if args.frequency_resolution is None:
                args.frequency_resolution = data['Sampling Rate']/data['Acoustic Data'].shape[-1]
            
            args.nperseg = int(data['Sampling Rate']/args.frequency_resolution)

            if args.end_t is None:
                args.end_t = data['Acoustic Data'].shape[-1]/data['Sampling Rate']
            args.start_ind = int(args.start_t*data['Sampling Rate'])
            args.end_ind = int(args.end_t*data['Sampling Rate'])
            # sets to true in order to plot figures
            args.plot = True
        # If the performance data is not yet appended to the data dictionary it does so
        if 'Performance_Data' not in data:
            
            append_perf_dat(data, dir)
            if 'ZSYNC' in data:
            # removes the ZSYNCH dataset since it's not used
                data.pop('ZSYNC')
            # zeros the time array
            data['Performance_Data']['Time (s)'] = data['Performance_Data']['Time (s)']-data['Performance_Data']['Time (s)'][0]
            
            # applies sensitivities to raw acoustic measurements 
            data['Acoustic Data'] =(data['Acoustic Data'].T/(data['Sensitivities']*1e-3)).T

            # writes out the new h5 file containing the performance and acoustics data
            write_h5(os.path.join(dir,'acs_data.h5'), data)

            #   Makes a new 'Figures' folder in the current directory
            # if 'Figures' not in subdir:
            #     # creates a directory for storing results
            #     os.mkdir(os.path.join(dir, 'Figures'))
                # deletes the two lvm files since they are not used
            if 'acs_data.lvm' in subdir:
                os.remove(os.path.join(dir,'acs_data.lvm'))
            if 'SPL_FFT.lvm' in subdir:
                os.remove(os.path.join(dir,'SPL_FFT.lvm'))
        
        data['case_dir'] = dir  
        # applies the provided functions to the dataset
        for f in function:
            f(data)


def append_perf_dat(dat_file, dir,col = 29):
    '''
    This function appends the performance and all the other data contained in Full.txt, which is exported by default
    from the LabVIEW data acquisition vi (UAV Control V4.vi) to the acs_data.h5 file.
    :param dat_file: opened acs_dat.h5 file to which to append the performance data
    :param b: prefix to the path containing the Full.txt file
    :param col: number of columns in the Full.txt file
    :return:
    '''

    #   opens and reads contents of Full.txt file. Any slashes ('/') are replaced with underscores since dataset names cannot contain '/'.
    with open(os.path.join(dir, 'Full.txt'), 'r') as f_txt:
        data = f_txt.read().replace('/','_')
    #   splits the Full.txt file with \t and \n as delimiters
    data_split = re.split('\t|\n', data)
    #   reshapes a single-dimensional list into the right dimensional array
    #   extracts header of each data set in Full.txt
    header = data_split[:col]

    data_split = np.reshape(data_split[col:-1], (int(len(data_split[col:-1]) / col), col)).astype(float)
    #   loops through each dataset contained in Full.txt
    #   appends data in of each column of Full.txt to the data dictionary
    dat_file.update({'Performance_Data':dict(zip(header, data_split.T))})


def p_tseries(data,args):
    
    t = np.arange(data['Acoustic Data'].shape[-1])*data['Sampling Rate']**-1

    if args.plot():
        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.15,bottom=0.15,right = 0.8)
        lines = ax.plot(t,data['Acoustic Data'][args.mics].T)
        for i,mic in enumerate(args.mics):
            lines[i].set(color=np.roll(default_colors,-i)[0], linestyle=np.roll(linestyle,-i)[0], label=f"Mic {mic}")
        ax.grid()
        ax.set(xlabel = r'$Time \ [sec]$',ylabel = r'$Pressure \ [Pa]$')
        ax.legend(loc='center left', bbox_to_anchor=(1.005, 0.5),prop={'size': 12})
        plt.savefig(os.path.join(data['case_dir'],'p_tseries.png'),format = 'png')
        plt.close()

def perf_tseries(data,args):
    
    fs = 1/np.diff(data['Performance_Data']['Time (s)'][:2])[0]
    LE_ind,_,_,rpm_nom,_ = eval_rpm(data['Performance_Data']['Motor2 RPM'],fs,start_t = 0,end_t = data['Performance_Data']['Time (s)'][-1])
    t = (data['Performance_Data']['Time (s)'][LE_ind][1:]+data['Performance_Data']['Time (s)'][LE_ind][:-1])/2
    rpm = np.diff(data['Performance_Data']['Time (s)'][LE_ind])**-1*60
    

    if args.plot:
        fig, ax = plt.subplots(3,1,figsize = (6.4,5))
        plt.subplots_adjust(left=0.2,bottom=0.15,hspace=0.4)
        ax[0].plot(data['Performance_Data']['Time (s)'],data['Performance_Data']['Motor2 Thrust (Nm)'])
        ax[1].plot(data['Performance_Data']['Time (s)'],data['Performance_Data']['Motor2 Torque (Nm)'])
        ax[2].plot(t,rpm)
        ax[0].set(ylabel = r'$Thrust \ [N]$',xticklabels = [],ylim =np.round([data['Performance_Data']['Motor2 Thrust (Nm)'].min()-2,data['Performance_Data']['Motor2 Thrust (Nm)'].max()+2]))
        ax[1].set(ylabel = r'$Torque \ [Nm]$',xticklabels = [],ylim = np.round([data['Performance_Data']['Motor2 Torque (Nm)'].min()-.02,data['Performance_Data']['Motor2 Torque (Nm)'].max()+.02],2))
        ax[-1].set(xlabel = r'$Time \ [sec]$',ylabel = r'$RPM$',ylim = np.round([rpm_nom-25,rpm_nom+25]))
        [ax[i].grid() for i in range(3)]
        plt.savefig(os.path.join(data['case_dir'],'perf_tseries.png'),format = 'png')
        plt.close()


def psd(data,args):

    f,pxx = welch(data['Acoustic Data'][args.mics,args.start_ind:args.end_ind], fs=data['Sampling Rate'], window=args.window, nperseg=args.nperseg, noverlap=int(args.overlap*args.nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    if args.plot:
        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.1,bottom=0.15,right = 0.8)
        lines = ax.plot(f,10*np.log10(pxx.T/20e-6**2))
        for i,mic in enumerate(args.mics):
            lines[i].set(color=np.roll(default_colors,-i)[0], linestyle=np.roll(linestyle,-i)[0], label=f"Mic {mic}")
        ax.set(xlabel = r'$Frequency \ [Hz]$',ylabel = r'$PSD, \ dB/Hz \ (re: \ 20 \mu Pa)$',xlim = [10,10e3],xscale = 'log')
        ax.legend(loc='center left', bbox_to_anchor=(1.005, 0.5),prop={'size': 12})
        ax.grid()
        plt.savefig(os.path.join(data['case_dir'],'psd_spectrum.png'),format = 'png')
        plt.close()
    return f, pxx


def spectrogram(data,args):

    win = get_window(window = args.window, Nx = args.nperseg, fftbins=True)
    SFT = ShortTimeFFT(win = win, hop = int(args.nperseg-args.overlap*args.nperseg) , fs =data['Sampling Rate'] , fft_mode='onesided', scale_to='psd')
    Gxx = SFT.spectrogram(data['Acoustic Data'][args.mics,args.start_ind:args.end_ind])

    if args.plot:
        levels = np.linspace(0, 70, 71)

        for i,mic in enumerate(args.mics):
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
            plt.subplots_adjust(left = 0.15,bottom=0.15,right =1)
            spec = ax.contourf(SFT.t(len(data['Acoustic Data'][0,args.start_ind:args.end_ind])), SFT.f, 10 * np.log10(Gxx[i]* np.diff(SFT.f)[:1]/ 20e-6 ** 2), cmap='hot', levels=levels)
            ax.set(ylabel = r'$Frequency \ [Hz]$',xlabel = r'$Time \ [sec]$',title = rf'$Mic \ {mic}$',xlim =[0,None],ylim =[10,1e3])
            cbar = fig.colorbar(spec)
            cbar.set_label(r'$PSD, \ dB/Hz \ (re: \ 20 \mu Pa)$')
            plt.savefig(os.path.join(data['case_dir'],f'spectrogram_m{mic}.png'),format = 'png')
            plt.close()

    return SFT.t,SFT.f,Gxx

def eval_rpm(ttl,fs,start_t,end_t):
    '''
    This function evaluates the average rotational rate for a segment of data based on the tac pulse signal.
    :param ttl: raw tac time series
    :param fs: sampling rate of the tac signal [Hz]
    :param start_t: start time of the segment under consideration [s]
    :param end_t: end time of the segment under consideration [s]
    :return:
    '''
    # identifies the indices corresponding to the leading edge of each pulse
    LE_ind = np.squeeze(np.where(np.diff(ttl) == 1))
    # returns a list containing the starting and ending index
    lim_ind = list(map(lambda x: bisect(LE_ind, x),[start_t * fs,end_t * fs]))
    # computes the rotational rate over the time interval
    rpm = (np.diff(LE_ind[lim_ind[0]:lim_ind[1]]) / fs / 60) ** -1
    # averages the rpm
    rpm_nom = np.mean(rpm)
    # confidence interval of the the rpm (95% level of certainty)
    u_rpm = 1.94*np.std(rpm)/np.sqrt(len(LE_ind)-1)

    return LE_ind, lim_ind,rpm, rpm_nom, u_rpm

def upsample(xn, fs, N):
    '''
    This function upsamples a time series that was sampled at a sampling rate of fs to a length of N points.
    :param xn: times series
    :param fs: sampling rate [Hz]
    :param N: number of points to upsample to
    :return:s
    '''

    fs_upsample = N*fs/xn.shape[-1]
    if fs != fs_upsample:
        Xm = fft(xn,axis = -1).T*fs**-1
        Xm_upsample = np.zeros((N,xn.shape[0]),dtype = complex)
        if len(Xm)%2 ==0:
            Xm[int(len(Xm)/2)]/=2 
            Xm_upsample[:int(len(Xm)/2)+1] = Xm[:int(len(Xm)/2)+1]
            Xm_upsample[-int(len(Xm)/2):] = Xm[int(len(Xm)/2):]
        else:
            Xm_upsample[:int(len(Xm)/2)+1] = Xm[:int(len(Xm)/2)+1]
            Xm_upsample[-int(len(Xm)/2):] = Xm[int(len(Xm)/2)+1:]

        xn = ifft(Xm_upsample.T,axis =-1)*fs_upsample
    return xn


def tonal_separation(data,args,**kwargs):
    '''
    This function extracts te tonal and broadband components of a signal. The noise component separation is done in
    the frequency domain. The signal is first parsed on a rev-to-rev basis. The linear spectrum of each rev is then
    computed and upsampled by appending zeros to the linear spectrum of each record so that their lengths are equal.
    To determine the tonal noise components the linear spectrum is averaged across all revs. The averaged linear
    spectrum is then subtracted from that of each rev to determine the broadband contributions.

    The averaged linear spectrum can be filtered via the harm_filt BPF_harm parameters.

    :param xn: time series
    :param tac_ind: tachometer indices by which to parse the time series, ensure that the sample rates of the time series and TAC are equivalent or interpolated.
    :param fs: sampling rate of time series [Hz]
    :param rev_skip: number of intermediate revs to skip
    :param harm_filt: the BPF harmonic to retain, specified as a list [lowest BPF harmonic, highest BPF harmonic]
    :param filt_shaft_harm: boolean if set to True the shaft order harmonics will be filtered from the signal
    :param Nb:  number of blades, only needs to be specified when filt_shaft_harm is set to True
    :return:
    '''

    def filt_harmonics(xn,harmonics, filter_shaft_order = False):
        Xm_upsampled = fft(xn,axis = 1)
        
        if harmonics[0]!=0:
            Xm_upsampled[:,:harmonics[0]+1] = 0
            Xm_upsampled[:,-harmonics[0]:] = 0
        else:
            Xm_upsampled[:,:harmonics[0]+1] = 0 
            # Xm_upsampled[:,-1] = 0
        Xm_upsampled[:,harmonics[1]+1:-harmonics[1]] =0
        
        if filter_shaft_order:
            N = Xm_upsampled.shape[1]
            Xm_upsampled[:,:int(N/2)][:,1::2] = 0
            Xm_upsampled[:,int(N/2):][:,::-1][:,::2] = 0

        xn = np.real(ifft(Xm_upsampled,axis = 1))
        return xn

    # fs = 1/np.diff(data['Performance_Data']['Time (s)'][:2])[0]
    # LE_ind,_,rpm,_,_ = eval_rpm(data['Performance_Data']['Motor2 RPM'],fs,start_t = data['Performance_Data']['Time (s)'][0],end_t = data['Performance_Data']['Time (s)'][-1])

    # fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
    # plt.subplots_adjust(left=0.15,bottom=0.15,right = 0.8)
    # ax.plot(np.diff(data['Performance_Data']['Time (s)']))
    
    # t_perf = 0.5*(data['Performance_Data']['Time (s)'][LE_ind[1:]]+data['Performance_Data']['Time (s)'][LE_ind[:-1]])
    t_perf = np.arange(len(data['Performance_Data']['Motor2 RPM']))*data['Performance_Data']['Time (s)'][-1]/(len(data['Performance_Data']['Motor2 RPM'])-1)
    fs = 1/np.diff(t_perf[:2])[0]

    LE_ind,_,rpm,_,_ = eval_rpm(data['Performance_Data']['Motor2 RPM'],fs,start_t = data['Performance_Data']['Time (s)'][0],end_t = data['Performance_Data']['Time (s)'][-1])
    t_perf = data['Performance_Data']['Time (s)'][LE_ind]

    t_perf = t_perf[(t_perf>=args.start_t) & (t_perf<=args.end_t)]
    t_acs = (np.arange(data['Acoustic Data'].shape[-1])*data['Sampling Rate']**-1)[args.start_ind:args.end_ind]

    rev_ind = np.abs(t_perf[...,None]-t_acs).argmin(axis = -1)
    N_upsample = np.diff(rev_ind[1:-1]).max()
    fs_upsample = N_upsample / np.diff(rev_ind) * data['Sampling Rate']

    acs_data_split = np.split(data['Acoustic Data'][args.mics,args.start_ind:args.end_ind], rev_ind,axis = -1)
    N_records = len(acs_data_split)-2
    xn_upsampled= np.asarray([upsample(record,data['Sampling Rate'],N_upsample) for record in acs_data_split[1:-1]]).transpose(1,2,0)
    
    t_upsample = np.arange(N_upsample)[:,None]/fs_upsample


    if args.filter_harmonics is not None:
        xn_upsampled = filt_harmonics(xn_upsampled,args.filter_harmonics,filter_shaft_order=args.filter_shaft_order)


    if args.align:
        
        # if len(args.mics)>1:
        #     mic_ind = 1
        # else:
        mic_ind = 1

        # creates a reference waveform to align each record with
        xn_upsampled_filt = np.concatenate((filt_harmonics(xn_upsampled[mic_ind,:,0][None],[0,2]).squeeze(),np.zeros(N_upsample)),axis = 0)
        ref_wavform = np.zeros(2*N_upsample)
        ref_wavform[:N_upsample] = xn_upsampled_filt.max()*np.sin(4*np.pi*np.arange(N_upsample)/(N_upsample-1))

        # xn_upsampled_filt = filt_harmonics(xn_upsampled[1,:,0][None],[0,2]).squeeze()
        # ref_wavform = xn_upsampled_filt.max()*np.sin(4*np.pi*np.arange(N_upsample)/(N_upsample-1))

        t_ref,Rxy_ref,_ =correlation(X = ref_wavform,Y = xn_upsampled_filt,fs =fs_upsample[0] ,auto = False)
        xn_upsampled[:,:,0] = np.roll(np.real(xn_upsampled[...,0]), -t_ref[Rxy_ref.argmax()]*fs_upsample[0], axis=-1)

        # # appends zeros to time series to avoid performing a circular cross-correlation 
        xn_upsampled_zero_pad = np.concatenate((xn_upsampled[mic_ind],np.zeros(xn_upsampled[mic_ind].shape)),axis = 0)

        # computes the cross-correlation between each record or rotor revolution relative to the first
        t,Rxy,_ =correlation(X = xn_upsampled_zero_pad[:,0].T,Y = xn_upsampled_zero_pad.T,fs =fs_upsample ,auto = False)
        
        # if np.abs(t_ref[Rxy_ref.argmax()])>t_ref[-1]/8:
        #     # shifts the time series in the same direction as the reference waveform
        #     if t_ref[Rxy_ref.argmax()]<0:
        # # Peak of cross-correlation is the time delay by which to shift the other records
        #         t_shift = t[:N_upsample+1][Rxy[:,:N_upsample+1].argmax(axis = -1),np.arange(len(Rxy))]
        #     else:
        #         t_shift = t[N_upsample:][Rxy[:,N_upsample:].argmax(axis = -1),np.arange(len(Rxy))]
        # else:
        t_shift = t[Rxy.argmax(axis = -1),np.arange(len(Rxy))]

       # Creates an array of indicies to shift the original signal by 
        shift_ind = (np.arange(N_upsample) + (t_shift*fs_upsample).astype(int)[:, None]).T % N_upsample
        # shifts the upsampled signal by the computed time delays
        xn_upsampled = np.take_along_axis(xn_upsampled, shift_ind[None], axis=1)

        # fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        # plt.subplots_adjust(left=0.15,bottom=0.15)
        # # ax.plot(t_ref,(Rxy_ref))
        # ax.plot(t[:,100],(Rxy[100]))

        # fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        # plt.subplots_adjust(left=0.15,bottom=0.15)
        # ax.plot(t_upsample[:,0],ref_wavform[:N_upsample])
        # ax.plot(t_upsample[:,0],xn_upsampled_filt[:N_upsample])
        # ax.plot(t_upsample[:,0], np.roll(np.real(xn_upsampled_filt[:N_upsample]), -t_ref[Rxy_ref.argmax()]*fs_upsample[0], axis=-1))

        # ind = 449
        # fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        # plt.subplots_adjust(left=0.15,bottom=0.15)
        # ax.plot(t_upsample[:,0],np.real(xn_upsampled_filt[:N_upsample]))
        # ax.plot(t_upsample[:,0],np.roll(np.real(xn_upsampled_filt[:N_upsample]), -t_ref[Rxy_ref.argmax()]*fs_upsample[0], axis=-1))
        
        # ax.plot(t_upsample[:,ind],np.real(xn_upsampled[0,:,ind]))
        # ax.plot(t_upsample[:,ind],np.roll(np.real(xn_upsampled[0,:,ind]), -t_shift[ind]*fs_upsample[ind], axis=-1))

        # ax.plot(t_upsample[:,0],np.roll(np.real(xn_upsampled_filt[0,:,0]), -t_ref[Rxy_ref.argmax()]*fs_upsample[0], axis=-1))


        # ax.plot(t_upsample[:,ind],np.take_along_axis(np.real(xn_upsampled[0,:,ind]), shift_ind[:,ind],axis = 0))
        # ax.plot(t_upsample[:,-1],np.roll(np.real(xn_upsampled_filt[1,:,-1]), -t_shift[-1]*fs_upsample[-1]))

    # computes the average across all records to extract tonal noise component
    xn_avg = np.real(np.mean(xn_upsampled,axis = -1))


    if not N_upsample%2:
        Xm_upsampled =  fft(xn_upsampled,axis = 1)[:,:int(N_upsample/2)+1]/fs_upsample
    else:
        Xm_upsampled =  fft(xn_upsampled,axis = 1)[:,:int(N_upsample/2)]/fs_upsample
    
    Xm_avg = np.mean(Xm_upsampled,axis = -1)
    pxx_tonal = (fs_upsample.mean()/N_upsample)*np.abs(Xm_avg)**2
    pxx_tonal[:,1:-1] = 2*pxx_tonal[:,1:-1] 

    Gxx = (fs_upsample.mean()/N_upsample)*np.abs(Xm_upsampled)**2
    Gxx[:,1:-1] = 2*Gxx[:,1:-1]

    # # jacobian of single-sided power spectral density (unscaled by period)
    # Gxx_jac = 2*np.asarray([np.real(Xm_avg),np.imag(Xm_avg)])
    # # computes the covariance matrix of the linear spectrum
    # Xm_diff = Xm_upsampled-np.expand_dims(Xm_avg,axis = -1)
    # Xm_cov = np.asarray([np.real(Xm_diff),np.imag(Xm_diff)])
    # # covariance matrix of the average linear spectrum, hence why it is divided by the number of records
    # Xm_cov = 1/(N_records-1)*np.einsum("imfk,jmfk->fmij",Xm_cov,Xm_cov)/N_records
    # # variance in the power spectral density (unscaled by period)
    # Gxx_var = np.einsum("imf,fmij,jmf->mf",Gxx_jac,Xm_cov,Gxx_jac)

    # this is another estimate for the variance, here the linear spectrum are assumed to follow a complex normal distribution
    Gxx_var = 2*np.var(Xm_upsampled,axis = -1)*np.abs(Xm_avg)**2/N_records
    
    Gxx_err =(fs_upsample.mean()/N_upsample)*1.96*np.sqrt(Gxx_var)
    Gxx_err[:,1:-1] = 2*Gxx_err[:,1:-1]

    # lower and upper error bounds of the PSD
    yerr = np.asarray([pxx_tonal/(pxx_tonal-Gxx_err),(pxx_tonal+Gxx_err)/pxx_tonal]).transpose(1,0,-1)
    neg_yerr_ind = np.where(yerr<0)
    yerr[neg_yerr_ind] = 1
    yerr[(neg_yerr_ind[0],np.ones(len(neg_yerr_ind[1]),dtype=int),neg_yerr_ind[2])] = 1

    # # computes the PSD of the average time series using welchs method
    # f_tonal,pxx_tonal = welch(xn_avg, fs=fs_upsample.mean(), window='boxcar', nperseg=N_upsample, noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

    xn_bb= np.real(xn_upsampled.T-xn_avg.T).T.reshape((len(args.mics),int(N_upsample*len(acs_data_split[1:-1]))),order = 'F')
    f_bb,pxx_bb = welch(xn_bb[:,args.start_ind:args.end_ind], fs=data['Sampling Rate'], window=args.window, nperseg=args.nperseg, noverlap=int(args.overlap*args.nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    df_bb = np.diff(f_bb[:2])[0]


    t = np.arange(N_upsample)/fs_upsample.mean()

    f_tonal = np.arange(len(Gxx[0]))*fs_upsample.mean()/N_upsample
    df_tonal = np.diff(f_tonal[:2])[0]


    if args.plot:

        f,pxx = welch(data['Acoustic Data'][args.mics,args.start_ind:args.end_ind], fs=data['Sampling Rate'], window=args.window, nperseg=args.nperseg, noverlap=int(args.overlap*args.nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        df = np.diff(f[:2])[0]

        fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        plt.subplots_adjust(left=0.15,bottom=0.15,right = 0.8)
        lines = ax.plot(t,xn_avg.T)
        for i,mic in enumerate(args.mics):
            lines[i].set(color=np.roll(default_colors,-i)[0], linestyle=np.roll(linestyle,-i)[0], label=f"Mic {mic}")
        ax.grid()
        ax.set(xlabel = r'$Time \ [sec]$',ylabel = r'$Pressure \ [Pa]$')
        ax.legend(loc='center left', bbox_to_anchor=(1.005, 0.5),prop={'size': 12})
        plt.savefig(os.path.join(data['case_dir'],'p_avg_tseries.png'),format = 'png')
        plt.close()

        for i,mic in enumerate(args.mics):
            
            fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
            plt.subplots_adjust(left=0.15,bottom=0.15)
            # ax.scatter(f_tonal,10*np.log10(pxx_tonal[i]*df_tonal/20e-6**2),c = default_colors[3],zorder=3)
            # ax.scatter(f_tonal[:-1],10*np.log10(Gxx.mean(axis = 1)[i]*df_tonal/20e-6**2),c = default_colors[2],zorder=3)

            ax.errorbar(f_tonal, 10*np.log10(pxx_tonal[i]*df_tonal/20e-6**2), yerr=10*np.log10(yerr[i]), fmt='o',color = default_colors[3],ecolor = default_colors[3],capsize=6,capthick=2)

            ax.plot(f,10*np.log10(pxx[i]*df/20e-6**2))
            # ax.plot(f_bb,10*np.log10(pxx_bb[i]*df_bb/20e-6**2))
            ax.set(xlabel = r'$Frequency \ [Hz]$',ylabel = r'$SPL, \ dB \ (re: \ 20 \mu Pa)$',title = rf'$Mic \ {mic}$',xscale = 'log',xlim = [10,15e3],ylim = (0,90))
            ax.grid()
            plt.savefig(os.path.join(data['case_dir'],f'tonal_sep_m{mic}.png'),format = 'png')
            plt.close()

        # import matplotlib.colors as mc
        # import colorsys
        # c = colorsys.rgb_to_hls(*mc.to_rgb(default_colors[3]))
        # c_amount = (np.arange(N_records)/(N_records-1))

        # fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        # plt.subplots_adjust(left=0.175,bottom=0.15)
        # for i in range(int(np.round(N_records/5))):
        #     ax.plot((t_upsample[...,::5]/t_upsample[-1,::5])[:,i],np.real(xn_upsampled[0][...,::5][:,i]),c = colorsys.hls_to_rgb(c[0], c_amount[::5][i], c[2]))
        # ax.set(xlabel = r'$Rotation$',ylabel = r'$Pressure \ [Pa]$',xlim = [0,1])
        # ax.grid()

        # fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        # plt.subplots_adjust(left=0.175,bottom=0.15)
        # ax.plot(t_upsample[:,0]/t_upsample[-1,0],np.real(xn_upsampled[0,:,0]))

        # fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
        # plt.subplots_adjust(left=0.175,bottom=0.15)
        # ax.plot(t_acs,data['Acoustic Data'][0].squeeze())
        # ax.plot(data['Performance_Data']['Time (s)'],data['Performance_Data']['Motor2 RPM'])
        # ax.set(title = rf'$Mic \ {mic}$',xlabel = r'$Time \ [s]$',ylabel = r'$Pressure \ [Pa]$',xlim = [8.96,9])
        # ax.grid()

        for i,mic in enumerate(args.mics):
            fig, ax = plt.subplots(1,1,figsize = (6.4,4.5))
            plt.subplots_adjust(left=0.175,bottom=0.15)
            ax.plot(t_upsample/t_upsample[-1],np.real(xn_upsampled[i]),alpha = 0.3,c = 'gray')
            ax.plot(t/t[-1],xn_avg[i],linewidth = 1,c = 'black',label = 'Averaged')
            # ax.plot(t/t[-1],np.sin(4*np.pi*np.arange(N_upsample)/(N_upsample-1)),linewidth = 1,c = 'red',label = 'Reference')        
            ax.set(title = rf'$Mic \ {mic}$',xlabel = r'$Rotation$',ylabel = r'$Pressure \ [Pa]$',xlim = [0,1],ylim = [None,None])
            ax.grid()
            # ax.legend()
            plt.savefig(os.path.join(data['case_dir'],f'tonal_tseries_m{mic}.png'),format = 'png')
            plt.close()

    return t,xn_avg,xn_bb,f_tonal,pxx_tonal,yerr,f_bb,pxx_bb


def correlation(X,Y,fs,auto = True):
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
    t = ((np.arange(N)-N/2)[:,None]*fs**-1).squeeze()
    return t,Rxy,Cxy





def SD(Xm,fs):
    '''
    This function computes the single-sided spectral density (Gxx) from a linear spectrum.
    :param Xm: complex two-sided linear spectrum [Pa]
    :param fs: sampling rate [Hz]
    :return:
    '''

    # number of points in record
    N = Xm.shape[-1]
    # temporal resolution
    dt = fs ** -1
    # single sided power spectral density [Pa^2/Hz]
    Sxx = (dt * N) ** -1 * abs(Xm) ** 2
    Gxx = Sxx[...,:int(N / 2)]
    Gxx[...,1:-1] = 2 * Gxx[...,1:-1]
    return Gxx

