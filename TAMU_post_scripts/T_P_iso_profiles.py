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
dir = '/Users/danielweitsman/Box/May21Test/rpm_sweep/uohb'

# If you want to compare specific points in this directory their names can be specified in caseName list. Otherwise,
# all the cases in this directory directory would be compared and used to generate the thrust/torque profiles.
caseName = []

# Set equal to "True" in order to plot the thrust/torque/rpm time series for each run. These plots are not saved but are
# useful for determining the averaging interval to use for the thrust/torque profiles.
plot_tseries = True
save_h5 = False
#   Start time of averaging interval (s)
t_min = 10
#   End time of averaging interval (s)
t_max = 15


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
            # plt.savefig(os.path.join(dir,case,'Figures','T_Q_tseris_'+case+'.png'),format = 'png')

# %%     Thrust and rpm profile

#   Creates a new figures folder in the parent directory where to save the thrust and torque profiles
if not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures')):
    os.mkdir(os.path.join(os.path.dirname(dir), 'Figures'))
elif not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir))):
    os.mkdir(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir)))

#%%
#   initializes figures
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.set_title(os.path.basename(dir))

ax.errorbar(rpm_avg_1, T_avg_1, yerr=T_err_1, fmt=':o')

ax.set_ylabel('Thrust, T (N)')
ax.set_xlabel('RPM')
# ax.set_xlim([1000, 5250])
# ax.set_ylim([0, 70])
ax.legend(['Lower Rotor', 'Upper Rotor', 'Total'])
ax.grid()

plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'T_profile.png'), format='png')

# %%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.set_title(os.path.basename(dir))

ax.errorbar(rpm_avg_1, Q_avg_1, yerr=Q_err_1, fmt=':o')

ax.set_ylabel('Torque, Q (Nm)')
ax.set_xlabel('RPM')
# ax.set_xlim([1000, 5250])
# ax.set_ylim([-1.5, 1.5])
ax.legend(['Lower Rotor', 'Upper Rotor', 'Total'])
ax.grid()

plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'Q_profile.png'), format='png')

# %%
if save_h5:
# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.

    load_dat = {
                'rpm_avg_1': rpm_avg_1, 'T_avg_1': T_avg_1,
                'T_err_1': T_err_1, 'Q_avg_1': Q_avg_1, 'Q_err_1': Q_err_1,
                }

    if os.path.exists(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5')):
        os.remove(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5'))

    with h5py.File(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5'), 'a') as f:
        for k, dat in load_dat.items():
            f.create_dataset(k, shape=np.shape(dat), data=dat)
