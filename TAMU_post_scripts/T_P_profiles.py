import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import bisect

# %%
fontName = 'Times New Roman'
fontSize = 12
plt.rc('font', **{'family': 'serif', 'serif': [fontName], 'size': fontSize})
plt.rc('mathtext', **{'default': 'regular'})
plt.rc('text', **{'usetex': False})
plt.rc('lines', **{'linewidth': 2})

# %%

# path to directory containing the rpm sweep cases
dir = '/Users/danielweitsman/Box/Jan21Test/TAMU/runs/rpm_sweeps'

# If you want to compare specific points in this directory their names can be specified in caseName list. Otherwise,
# all the cases in this directory directory would be compared and used to generate the thrust/torque profiles.
caseName = ['ushb/ushb8','cshb/cshb10']

# Set equal to "True" in order to plot the thrust/torque/rpm time series for each run. These plots are not saved but are
# useful for determining the averaging interval to use for the thrust/torque profiles.
plot_tseries = False
save_h5 = False
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
    cases = [os.path.basename(dir) + str(x) for x in sorted([int(x[4:]) for x in os.listdir(dir)])]
else:
    cases = caseName

#   Initializes empty array
rpm_avg_1 = np.zeros(len(cases))
T_avg_1 = np.zeros(len(cases))
T_err_1 = np.zeros(len(cases))
Q_avg_1 = np.zeros(len(cases))
Q_err_1 = np.zeros(len(cases))
rpm_avg_2 = np.zeros(len(cases))
T_avg_2 = np.zeros(len(cases))
T_err_2 = np.zeros(len(cases))
Q_avg_2 = np.zeros(len(cases))
Q_err_2 = np.zeros(len(cases))

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

        #   Computes the rpm time series for the lower rotor
        rpm_2 = (np.diff(np.squeeze(np.where(np.diff(dat_file['Motor2 RPM']) == 1))) / fs / 60) ** -1
        #   Determines the time indeceis corresponding to each rpm pulse
        t_rpm_2 = t[np.squeeze(np.where(np.diff(dat_file['Motor2 RPM']) == 1))]
        rpm_avg_2[i] = np.mean(rpm_2[bisect.bisect(t_rpm_2, t_min):bisect.bisect(t_rpm_2, t_max)])
        #   Determines the number of revolutions in the averaging interval
        N_rev_2 = len(rpm_2[bisect.bisect(t_rpm_2, t_min):bisect.bisect(t_rpm_2, t_max)])

        #   filters thrust time series of lower rotor
        T_filt_2 = lfilter(b, a, dat_file['Motor2 Thrust (Nm)'])
        #   Computes the mean of the thrust time series of lower rotor
        T_avg_2[i] = np.mean(dat_file['Motor2 Thrust (Nm)'][int(t_min * fs):int(t_max * fs)])
        #   Computes the confidence interval of the thrust time series of lower rotor
        T_err_2[i] = 1.96 * np.std(dat_file['Motor2 Thrust (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev_2)

        #   filters thrust torque series of lower rotor
        Q_filt_2 = lfilter(b, a, dat_file['Motor2 Torque (Nm)'])
        #   Computes the mean of the torque time series of lower rotor
        Q_avg_2[i] = np.mean(dat_file['Motor2 Torque (Nm)'][int(t_min * fs):int(t_max * fs)])
        #   Computes the confidence interval of the torque time series of lower rotor
        Q_err_2[i] = 1.96 * np.std(dat_file['Motor2 Torque (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev_2)

        #   Repeats all the previous calculations for the upper rotor
        rpm_1 = (np.diff(np.squeeze(np.where(np.diff(dat_file['Motor1 RPM']) == 1))) / fs / 60) ** -1
        t_rpm_1 = t[np.squeeze(np.where(np.diff(dat_file['Motor1 RPM']) == 1))]
        rpm_avg_1[i] = np.mean(rpm_1[bisect.bisect(t_rpm_1, t_min):bisect.bisect(t_rpm_1, t_max)])
        N_rev_1 = len(rpm_1[bisect.bisect(t_rpm_1, t_min):bisect.bisect(t_rpm_1, t_max)])

        T_filt_1 = lfilter(b, a, dat_file['Motor1 Thrust (N)'])
        T_avg_1[i] = np.mean(dat_file['Motor1 Thrust (N)'][int(t_min * fs):int(t_max * fs)])
        T_err_1[i] = 1.96 * np.std(dat_file['Motor1 Thrust (N)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev_1)

        Q_filt_1 = lfilter(b, a, dat_file['Motor1 Torque (Nm)'])
        Q_avg_1[i] = np.mean(dat_file['Motor1 Torque (Nm)'][int(t_min * fs):int(t_max * fs)])
        Q_err_1[i] = 1.96 * np.std(dat_file['Motor1 Torque (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev_1)

        # %%

        #   plots the filtered thrust and torque  as well as the rpm time series
        if plot_tseries:
            fig, ax = plt.subplots(3, 1, figsize=(6.4, 4.5))
            plt.subplots_adjust(bottom=0.15)
            ax[0].set_title(case + ': Upper Rotor')

            ax[0].plot(t, T_filt_1)
            ax[0].plot(t, np.ones(len(t)) * T_avg_1[i])
            ax[0].tick_params(axis='x', labelsize=0)
            ax[0].set_ylabel('Thrust (N)')
            ax[0].grid()
            ax[0].set_xlim([t_min, t_max])
            # ax[0].set_ylim([-.08,.08])

            ax[1].plot(t, Q_filt_1)
            ax[1].plot(t, np.ones(len(t)) * Q_avg_1[i])
            ax[1].tick_params(axis='x', labelsize=0)
            ax[1].set_xlim([t_min, t_max])
            # ax[1].set_ylim([-.05,.0025])
            ax[1].set_ylabel('Torque (Nm)')
            ax[1].grid()

            ax[2].plot(t_rpm_1[:-1], rpm_1)
            ax[2].plot(t_rpm_1[:-1], np.ones(len(t_rpm_1[:-1])) * rpm_avg_1[i])
            ax[2].set_ylabel('RPM')
            ax[2].set_xlim([t_min, t_max])
            ax[2].set_xlabel('Time (sec)')
            ax[2].grid()

            fig, ax = plt.subplots(3, 1, figsize=(6.4, 4.5))
            plt.subplots_adjust(bottom=0.15)
            ax[0].set_title(case + ': Lower Rotor')

            ax[0].plot(t, T_filt_2)
            ax[0].plot(t, np.ones(len(t)) * T_avg_2[i])
            ax[0].tick_params(axis='x', labelsize=0)
            ax[0].set_ylabel('Thrust (N)')
            ax[0].grid()
            ax[0].set_xlim([t_min, t_max])
            # ax[0].set_ylim([-.08,.08])

            ax[1].plot(t, Q_filt_2)
            ax[1].plot(t, np.ones(len(t)) * Q_avg_2[i])
            ax[1].tick_params(axis='x', labelsize=0)
            ax[1].set_xlim([t_min, t_max])
            # ax[1].set_ylim([-.05,.0025])
            ax[1].set_ylabel('Torque (Nm)')
            ax[1].grid()

            ax[2].plot(t_rpm_2[:-1], rpm_2)
            ax[2].plot(t_rpm_2[:-1], np.ones(len(t_rpm_2[:-1])) * rpm_avg_2[i])
            ax[2].set_ylabel('RPM')
            ax[2].set_xlim([t_min, t_max])
            ax[2].set_xlabel('Time (sec)')
            ax[2].grid()

            # plt.savefig(os.path.join(dir,case,'Figures','T_Q_tseris_'+case+'.png'),format = 'png')

# %%     Thrust and rpm profile

#   Creates a new figures folder in the parent directory where to save the thrust and torque profiles
if not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures')):
    os.mkdir(os.path.join(os.path.dirname(dir), 'Figures'))
elif not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir))):
    os.mkdir(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir)))

T1_fit = np.poly1d(np.polyfit(rpm_avg_1, T_avg_1, 4))
Q1_fit = np.poly1d(np.polyfit(rpm_avg_1, Q_avg_1, 4))
T_tot = T_avg_2 + T1_fit(rpm_avg_2)
T_tot_err = np.sqrt(T_err_2 ** 2 + T_err_1 ** 2)
Q_tot = Q_avg_2 + Q1_fit(rpm_avg_2)
Q_tot_err = np.sqrt(Q_err_2 ** 2 + Q_err_1 ** 2)
#   initializes figures
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.set_title(os.path.basename(dir))

ax.errorbar(rpm_avg_2, T_avg_2, yerr=T_err_2, fmt='-.o')
ax.errorbar(rpm_avg_1, T_avg_1, yerr=T_err_1, fmt=':o')
ax.errorbar(rpm_avg_2, T_tot, yerr=T_tot_err, fmt='-o')

ax.set_ylabel('Thrust, T (N)')
ax.set_xlabel('RPM')
ax.set_xlim([1000, 5250])
ax.set_ylim([0, 70])
ax.legend(['Lower Rotor', 'Upper Rotor', 'Total'])
ax.grid()

plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'T_profile.png'), format='png')

# %%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.set_title(os.path.basename(dir))

ax.errorbar(rpm_avg_2, Q_avg_2, yerr=Q_err_2, fmt='-.o')
ax.errorbar(rpm_avg_1, Q_avg_1, yerr=Q_err_1, fmt=':o')
ax.errorbar(rpm_avg_2, Q_tot, yerr=Q_tot_err, fmt='-o')

ax.set_ylabel('Torque, Q (Nm)')
ax.set_xlabel('RPM')
ax.set_xlim([1000, 5250])
ax.set_ylim([-1.5, 1.5])
ax.legend(['Lower Rotor', 'Upper Rotor', 'Total'])
ax.grid()

plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'Q_profile.png'), format='png')

# %%
if save_h5:
# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.

    load_dat = {'rpm_avg_2': rpm_avg_2, 'T_avg_2': T_avg_2,
                'T_err_2': T_err_2, 'Q_avg_2': Q_avg_2, 'Q_err_2': Q_err_2,
                'rpm_avg_1': rpm_avg_1, 'T_avg_1': T_avg_1,
                'T_err_1': T_err_1, 'Q_avg_1': Q_avg_1, 'Q_err_1': Q_err_1, 'T_tot': T_tot, 'T_tot_err': T_tot_err,
                'Q_tot': Q_tot, 'Q_tot_err': Q_tot_err}

    if os.path.exists(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5')):
        os.remove(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5'))

    with h5py.File(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5'), 'a') as f:
        for k, dat in load_dat.items():
            f.create_dataset(k, shape=np.shape(dat), data=dat)
