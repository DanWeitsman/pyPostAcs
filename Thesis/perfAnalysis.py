import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import bisect
from timeit import default_timer
# %%
fontName = 'Times New Roman'
fontSize = 12
plt.rc('font', **{'family': 'serif', 'serif': [fontName], 'size': fontSize})
plt.rc('mathtext', **{'default': 'regular'})
plt.rc('text', **{'usetex': False})
plt.rc('lines', **{'linewidth': 2})

# %%

# path to directory containing the rpm sweep cases
dir = '/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/h2b69/'

# If you want to compare specific points in this directory their names can be specified in caseName list. Otherwise,
# all the cases in this directory directory would be compared and used to generate the thrust/torque profiles.
caseName = []

# Set equal to "True" in order to plot the thrust/torque/rpm time series for each run. These plots are not saved but are
# useful for determining the averaging interval to use for the thrust/torque profiles.
plot_tseries = True
plot_T_Q_profile = False
plot_OASPL= False

#   save generated figures
save_fig = False
#   save average load data and OASPL into a new h5 file
save_h5 = False

#   moment arm length from LC to rotor hub [m]
d = 0.18542

#   mic number to plot
mics = [1,5,9,10]
#   Start time of averaging interval (s)
t_min = 10
#   End time of averaging interval (s)
t_max = 15

# %%
# def filt_response(bb,aa,fs,N):
#     '''
#     This function returns the frequency response of a moving average filter by computing the linear spectrum of the impulse response.
#     :param bb: output (numerator) coefficients of the frequency response, multiplied by dt
#     :param aa: input (denominator) coefficients of the frequency response
#     :param fs: sampling frequency [Hz]
#     :param N: length of the impulse time series [points]
#     :return:
#     :param f: frequency vector [Hz]
#     :param y: impulse time series
#     :param h: frequency response
#     :param phase: phase [deg]
#
#     '''
#     impulse = np.zeros(int(N))
#     impulse[0] = fs
#     y = lfilter(bb, aa, impulse)
#     h = (fft(y)*fs**-1)[:int(N/2)]
#     phase = np.angle(h) * 180 / np.pi
#     f = np.arange(N/2)*(N*fs**-1)**-1
#     return f,y,h,phase

# %% Design Butterworth filter and check frequency response by applying the filter to an impulse response

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

# %%
#   Determines which cases to use for the thrust/torque profiles
if caseName == []:
    # cases = os.listdir(dir)
    # This line of code rearranges the order of the cases so that they are increasing
    # numerically, however it only applies to the TAMU data. Comment this line and uncomment the previous line if you
    # are working with an alternate dataset.
    cases = [os.path.basename(dir)[:3] + str(x) for x in sorted([int(x[3:]) for x in os.listdir(dir)])]
else:
    cases = caseName

t1 = default_timer()
#   Initializes empty array
rpm_avg =np.zeros(len(cases))
T_avg=  np.zeros(len(cases))
T_err=np.zeros(len(cases))
Q_avg=np.zeros(len(cases))
Q_err=np.zeros(len(cases))
Fx_avg=np.zeros(len(cases))
Fx_err=np.zeros(len(cases))
Fy_avg=np.zeros(len(cases))
Fy_err=np.zeros(len(cases))
Mx_avg=np.zeros(len(cases))
Mx_err=np.zeros(len(cases))
My_avg=np.zeros(len(cases))
My_err = np.zeros(len(cases))
OASPL=np.zeros((len(cases),12))

#   Loops through all the cases
for i, case in enumerate(cases):
    #   Opens the h5 file corresponding to each case
    with h5py.File(os.path.join(dir, case, 'acs_data.h5'), 'r') as dat_file:
        #   temporal resolution (s)
        dt = np.diff(dat_file['Time (s)'])[0]
        #   Sampling rate (Hz)
        fs = dt ** -1
        #   Aranges time vector
        t = np.arange(len(dat_file['Motor2 Thrust (Nm)'])) * dt
        # generates the filter coefficients for a 4th order low pass Butterworth filter
        b, a = butter(4, 10 / (fs / 2), 'lp')

        rpm = (np.diff(np.squeeze(np.where(np.diff(dat_file['Motor1 RPM']) == 1))) / fs / 60) ** -1
        t_rpm = t[np.squeeze(np.where(np.diff(dat_file['Motor1 RPM']) == 1))]
        rpm_avg[i] = np.mean(rpm[bisect.bisect(t_rpm, t_min):bisect.bisect(t_rpm, t_max)])
        N_rev = len(rpm[bisect.bisect(t_rpm, t_min):bisect.bisect(t_rpm, t_max)])

        T_filt = lfilter(b, a, dat_file['Motor1 Thrust (N)'])
        T_avg[i] = np.mean(dat_file['Motor1 Thrust (N)'][int(t_min * fs):int(t_max * fs)])
        T_err[i] = 1.96 * np.std(dat_file['Motor1 Thrust (N)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        Fx_filt = lfilter(b, a, dat_file['Motor1 Fx (N)'])
        Fx_avg[i] = np.mean(dat_file['Motor1 Fx (N)'][int(t_min * fs):int(t_max * fs)])
        Fx_err[i] = 1.96 * np.std(dat_file['Motor1 Fx (N)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        Fy_filt = lfilter(b, a, dat_file['Motor1 Fy (N)'])
        Fy_avg[i] = np.mean(dat_file['Motor1 Fy (N)'][int(t_min * fs):int(t_max * fs)])
        Fy_err[i] = 1.96 * np.std(dat_file['Motor1 Fy (N)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        Q_filt = lfilter(b, a, dat_file['Motor1 Torque (Nm)'])
        Q_avg[i] = np.mean(dat_file['Motor1 Torque (Nm)'][int(t_min * fs):int(t_max * fs)])
        Q_err[i] = 1.96 * np.std(dat_file['Motor1 Torque (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        Mx_filt = lfilter(b, a, dat_file['Motor1 Mx (Nm)'])
        Mx_avg[i] = np.mean(dat_file['Motor1 Mx (Nm)'][int(t_min * fs):int(t_max * fs)])
        Mx_err[i] = 1.96 * np.std(dat_file['Motor1 Mx (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        My_filt = lfilter(b, a, dat_file['Motor1 My (Nm)'])
        My_avg[i] = np.mean(dat_file['Motor1 My (Nm)'][int(t_min * fs):int(t_max * fs)])
        My_err[i] = 1.96 * np.std(dat_file['Motor1 My (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        fs_acs = dat_file['Sampling Rate'][()]
        OASPL[i] = 10*np.log10(np.mean((dat_file['Acoustic Data'][()].transpose()[int(t_min * fs_acs):int(t_max * fs_acs)]/(dat_file['Sensitivities'][:]*1e-3))**2,axis = 0)/20e-6**2)

        # %%

        #   plots the filtered thrust and torque  as well as the rpm time series
        if plot_tseries:
            fig, ax = plt.subplots(3, 1, figsize=(6.4, 4.5))
            plt.subplots_adjust(bottom=0.15)
            ax[0].set_title(case + ': Upper Rotor')

            ax[0].plot(t, T_filt)
            ax[0].plot(t, np.ones(len(t)) * T_avg[i])
            ax[0].tick_params(axis='x', labelsize=0)
            ax[0].set_ylabel('Thrust (N)')
            ax[0].grid()
            ax[0].set_xlim([t_min, t_max])
            # ax[0].set_ylim([-.08,.08])

            ax[1].plot(t, Q_filt)
            ax[1].plot(t, np.ones(len(t)) * Q_avg[i])
            ax[1].tick_params(axis='x', labelsize=0)
            ax[1].set_xlim([t_min, t_max])
            # ax[1].set_ylim([-.05,.0025])
            ax[1].set_ylabel('Torque (Nm)')
            ax[1].grid()

            ax[2].plot(t_rpm[:-1], rpm)
            ax[2].plot(t_rpm[:-1], np.ones(len(t_rpm[:-1])) * rpm_avg[i])
            ax[2].set_ylabel('RPM')
            ax[2].set_xlim([t_min, t_max])
            ax[2].set_xlabel('Time (sec)')
            ax[2].grid()

#%% Resolves measured loads and moments to the hub frame
Mx_2 = (Fy_avg*np.cos(22.5*np.pi/180)+Fx_avg*np.sin(22.5*np.pi/180))*d
My_2 = (Fx_avg * np.cos(22.5 * np.pi / 180) - Fy_avg * np.sin(22.5 * np.pi / 180)) * d
Fx_2 = My_avg/d*np.cos(22.5*np.pi/180)+Mx_avg/d*np.sin(22.5*np.pi/180)
Fy_2 = My_avg/d*np.sin(22.5*np.pi/180)+Mx_avg/d*np.cos(22.5*np.pi/180)

#%%

T_fit =np.poly1d(np.polyfit(rpm_avg,T_avg,2))
Q_fit = np.poly1d(np.polyfit(rpm_avg,Q_avg,2))

rpm_run = np.arange(2000,6500,500)

#%%
if save_fig:
    #   Creates a new figures folder in the parent directory where to save the thrust and torque profiles
    if not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures')):
        os.mkdir(os.path.join(os.path.dirname(dir), 'Figures'))
    if not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir))):
        os.mkdir(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir)))

# %%     Thrust and rpm profile

if plot_T_Q_profile:
    #   initializes figures
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
    ax.set_title(os.path.basename(dir))
    ax.errorbar(rpm_avg, T_avg, yerr=T_err, fmt='-.o')
    plt.plot(rpm_run,T_fit(rpm_run))
    ax.set_ylabel('Thrust, T (N)')
    ax.set_xlabel('RPM')
    # ax.set_xlim([1000, 5250])
    # ax.set_ylim([0, 70])
    ax.grid()
    if save_fig:
        plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'T_profile.png'), format='png')

# %%
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
    ax.set_title(os.path.basename(dir))
    ax.errorbar(rpm_avg, Q_avg, yerr=Q_err, fmt=':o')
    ax.set_ylabel('Torque, Q (Nm)')
    ax.set_xlabel('RPM')
    # ax.set_xlim([1000, 5250])
    # ax.set_ylim([-1.5, 1.5])
    ax.grid()
    if save_fig:
        plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'Q_profile.png'), format='png')

# %%
elif plot_OASPL:
    for m in mics:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
        ax.set_title(os.path.basename(dir)+': Mic '+str(m))
        ax.plot(My_2, OASPL[:, m], marker ='o')
        ax.set_ylabel('$OASPL, \ dB \ (re: 20 \mu Pa)$')
        ax.set_xlabel('Pitching Moment, My (Nm)')
        # ax.set_xlim([-.15, .07])
        # ax.set_ylim([86, 89])
        ax.grid()
        if save_fig:
            plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'My_OASPL_m'+str(m)+'.png'), format='png')

        #%%
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
        ax.set_title(os.path.basename(dir)+': Mic '+str(m))
        ax.plot(Mx_2, OASPL[:, m], marker ='o')
        ax.set_ylabel('$OASPL, \ dB \ (re: 20 \mu Pa)$')
        ax.set_xlabel('Rolling Moment, Mx (Nm)')
        # ax.set_xlim([-.15, .07])
        # ax.set_ylim([86, 89])
        ax.grid()
        if save_fig:
            plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'Mx_OASPL_m'+str(m)+'.png'), format='png')

    # %%
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
        ax.set_title(os.path.basename(dir)+': Mic '+str(m))
        ax.plot(Fy_2, OASPL[:, m], marker ='o')
        ax.set_ylabel('$OASPL, \ dB \ (re: 20 \mu Pa)$')
        ax.set_xlabel('Hub Force, Fy (N)')
        # ax.set_xlim([-.75, .75])
        # ax.set_ylim([86, 89])
        ax.grid()

        if save_fig:
            plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'Fy_OASPL_m'+str(m)+'.png'), format='png')

    # %%
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
        ax.set_title(os.path.basename(dir)+': Mic '+str(m))
        ax.plot(Fx_2, OASPL[:, m], marker ='o')
        ax.set_ylabel('$OASPL, \ dB \ (re: 20 \mu Pa)$')
        ax.set_xlabel('Side Force, Fx (N)')
        # ax.set_xlim([-.75, .75])
        # ax.set_ylim([86, 89])
        ax.grid()

        if save_fig:
            plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'Fx_OASPL_m'+str(m)+'.png'), format='png')

# %%

# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.
if save_h5:
    save_dat = {'rpm':rpm,'t_rpm':t_rpm,'rpm_avg':rpm_avg,'N_rev':N_rev,'T_filt':T_filt,'T_avg':T_avg,'T_err':T_err,
                'Fx_filt':Fx_filt,'Fx_avg':Fx_avg,'Fx_err':Fx_err,'Fy_filt':Fy_filt,'Fy_avg':Fy_avg,'Fy_err':Fy_err,
                'Q_filt':Q_filt,'Q_avg':Q_avg,'Q_err':Q_err,'Mx_filt':Mx_filt,'Mx_avg':Mx_avg,'Mx_err':Mx_err,
                'My_filt':My_filt,'My_avg':My_avg,'My_err':My_err,'OASPL':OASPL}

    if os.path.exists(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5')):
        os.remove(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5'))

    with h5py.File(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5'), 'a') as f:
        for k, dat in save_dat.items():
            f.create_dataset(k, shape=np.shape(dat), data=dat)

