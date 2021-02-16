import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter
from scipy.fft import fft

#%%
fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%

dir ='/Users/danielweitsman/OneDrive - The Pennsylvania State University/January2021TestCampaign/Runs/2_12_21'
# dir_save = '/Users/danielweitsman/OneDrive - The Pennsylvania State University/January2021TestCampaign/NASA Blade Failure Analysis/Figures'
caseName  = ['RUN51','RUN57']
# caseName  = ['RUN4']
leglab = ''

#%%
def filt_response(bb,aa,fs,N):
    '''
    This function returns the frequency response of a moving average filter by computing the linear spectrum of the impulse response.
    :param bb: output (numerator) coefficients of the frequency response, multiplied by dt
    :param aa: input (denominator) coefficients of the frequency response
    :param fs: sampling frequency [Hz]
    :param N: length of the impulse time series [points]
    :return:
    :param f: frequency vector [Hz]
    :param y: impulse time series
    :param h: frequency response
    :param phase: phase [deg]

    '''
    impulse = np.zeros(int(N))
    impulse[0] = fs
    y = lfilter(bb, aa, impulse)
    h = (fft(y)*fs**-1)[:int(N/2)]
    phase = np.angle(h) * 180 / np.pi
    f = np.arange(N/2)*(N*fs**-1)**-1
    return f,y,h,phase

#%% Design Butterworth filter and check frequency response by applying the filter to an impulse response

# b,a = butter(4,25/(5e3/2), 'lp')
# f,y,h,phase = filt_response(b,a,fs = 5e3,N = 5e3*5)

# fig,ax = plt.subplots(2,1,figsize = (6.4,4.5))
# ax[0].plot(f,20*np.log10(abs(h)))
# ax[0].set_ylabel('Magnitude')
# ax[0].tick_params(axis = 'x', labelsize=0)
# ax[0].grid()
# ax[0].set_xscale('log')
# ax[0].set_xlim(f[0],f[-1])
#
# ax[1].plot(f,phase)
# ax[1].set_ylabel('Phase [$\circ$]')
# ax[1].set_xlabel('Frequency [Hz]')
# ax[1].grid()
# ax[1].set_xscale('log')
# ax[1].set_xlim(f[0],f[-1])

#%%
t_min = 0
t_max = 5
rpm_avg = np.zeros(len(caseName))
T_avg= np.zeros(len(caseName))
Q_avg= np.zeros(len(caseName))

for i,case in enumerate(caseName):
    with h5py.File(os.path.join(dir,caseName[i], 'acs_data.h5'), 'r') as dat_file:

        #%% process data

        dt = np.diff(dat_file['Time (s)'])[0]
        fs = dt ** -1
        t = np.arange(len(dat_file['Motor2 Thrust (Nm)'])) * dt
        rpm = (np.diff(np.squeeze(np.where(np.diff(dat_file['Motor2 RPM']) == 1))) / fs / 60) ** -1
        t_rpm = t[np.squeeze(np.where(np.diff(dat_file['Motor2 RPM']) == 1))]
        rpm_avg[i] = np.mean(rpm)

        b, a = butter(4, 10 / (fs / 2), 'lp')
        T_filt = lfilter(b,a,dat_file['Motor2 Thrust (Nm)'])
        T_avg[i] = np.mean(dat_file['Motor2 Thrust (Nm)'][int(t_min*fs):int(t_max*fs)])
        Q_filt = lfilter(b,a,dat_file['Motor2 Torque (Nm)'])

        Q_avg[i] = np.mean(dat_file['Motor2 Torque (Nm)'][int(t_min*fs):int(t_max*fs)])

        # b, a = butter(4, 2000 / (fs / 2), 'lp')
        a_filt = lfilter(b,a,dat_file['Acceleration_Ch2(m_s2)'])

        #   single sided spectral density of loads
        df = .25
        f,Gxx,Gxx_avg = fun.msPSD(dat_file['Motor2 Thrust (Nm)'][int(t_min*fs):int(t_max*fs)], fs=fs, df=df, win=False, ovr=0, f_lim=[0, 5e3], levels=[0, 100], save_fig=False, save_path='', plot=False)

#%%
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
        plt.subplots_adjust(bottom=0.15)
        ax.set_title(case+': '+ str(round(rpm_avg[i],1))+' rpm')
        ax.plot(f, 10 * np.log10(Gxx_avg * df))
        # ax.set_xscale('log')
        ax.set_xlim([f[0],40])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('$Thrust \ PSD, \ dB \ (re: \  1 \  N^2/Hz)$')
        ax.grid()
        # plt.savefig(os.path.join(dir_save,'thrust_spectra.png'), format='png')

        #%% Thrust & torque time series
        #
        fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.5))
        # plt.subplots_adjust(bottom=0.15)

        ax[0].plot(t,T_filt)
        # ax[0].plot(t,np.ones(len(t))*T_avg[i])
        ax[1].plot(t,Q_filt)
        # ax[2].plot(t,dat_file['Motor2 Throttle'][()]*10)

        # ax[1].plot(t,np.ones(len(t))*Q_avg[i])

        ax[0].set_title(case+': '+ str(round(rpm_avg[i],1))+' rpm')
        ax[0].tick_params(axis='x', labelsize=0)
        ax[0].set_ylabel('Thrust (N)')
        ax[0].grid()
        ax[0].set_xlim([t_min,t_max])
        ax[0].set_ylim([-.08,.08])

        # x[1].set_ylabel('Throttle (%)')
        # ax[1].tick_params(axis='x', labelsize=0)
        ax[1].set_xlim([t_min,t_max])
        ax[1].set_ylim([-.05,.0025])

        ax[1].set_ylabel('Torque (Nm)')
        # ax[1].set_xlabel('Time (sec)')
        ax[1].grid()
        # ax[2].set_xlim([t_min,t_max])
        # ax[2].set_ylabel('Throttle (%)')
        ax[1].set_xlabel('Time (sec)')
        # ax[2].grid()
        # ax[2].set_xlim([t_min,t_max])

        # plt.savefig(os.path.join(dir,case,'Figures','T_Q_tseris_'+case+'.png'),format = 'png')

#%%     rpm time series

        # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
        # # plt.subplots_adjust(bottom=0.15)
        # plt.title(case +': '+str(round(rpm_avg[i],1))+' rpm')
        # ax.plot(t_rpm[:-1],rpm)
        # ax.set_ylabel('RPM')
        # ax.set_xlabel('Time (sec)')
        # ax.grid()
        # ax.set_xlim([0,t_max])
        #
        # # plt.savefig(os.path.join(dir,case,'Figures','rpm_tseris_'+case+'.png'),format = 'png')
        #
        # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
        # plt.title(case+': '+ str(round(rpm_avg[i],1))+' rpm')
        # # plt.subplots_adjust(bottom=0.15)
        # ax.plot(t,a_filt)
        # ax.set_ylabel('$Test \ Stand \ Acceleration \ (m/s^2)$')
        # ax.set_xlabel('Time (sec)')
        # ax.grid()
        # ax.set_xlim([0,t_max])
        # plt.savefig(os.path.join(dir,case,'Figures','accel_tseris_'+case+'.png'),format = 'png')


# %%     Thrust and rpm profile

# fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.5))
# # plt.subplots_adjust(bottom=0.15)
#
# ax[0].scatter(rpm_avg,T_avg)
# ax[1].scatter(rpm_avg,Q_avg)
# ax[0].tick_params(axis='x', labelsize=0)
# ax[0].set_ylabel('Thrust (N)')
# ax[0].grid()
# ax[1].set_ylabel('Torque (Nm)')
# ax[1].set_xlabel('RPM')
# ax[1].grid()

# plt.savefig(os.path.join(dir_save,'T_Q_profile_not_negated.png'), format='png')

# plt.close('all')