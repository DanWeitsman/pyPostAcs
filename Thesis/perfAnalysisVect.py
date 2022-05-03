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

def getLoads(cases):
    #   Loops through all the cases
    with h5py.File(os.path.join(dir, cases, 'acs_data.h5'), 'r') as dat_file:
        #   temporal resolution (s)
        dt = np.diff(dat_file['Time (s)'])[0]
        #   Sampling rate (Hz)
        fs = dt ** -1
        #   Aranges time vector
        t = np.arange(len(dat_file['Motor2 Thrust (Nm)'])) * dt
        # generates the filter coefficients for a 4th order low pass Butterworth filter
        b, a = butter(4, 10 / (fs / 2), 'lp')

        rpm    = (np.diff(np.squeeze(np.where(np.diff(dat_file['Motor1 RPM']) == 1))) / fs / 60) ** -1
        t_rpm   = t[np.squeeze(np.where(np.diff(dat_file['Motor1 RPM']) == 1))]
        rpm_avg = np.mean(rpm[bisect.bisect(t_rpm, t_min):bisect.bisect(t_rpm, t_max)])
        N_rev = len(rpm[bisect.bisect(t_rpm, t_min):bisect.bisect(t_rpm, t_max)])

        T_filt = lfilter(b, a, dat_file['Motor1 Thrust (N)'])
        T_avg = np.mean(dat_file['Motor1 Thrust (N)'][int(t_min * fs):int(t_max * fs)])
        T_err = 1.96 * np.std(dat_file['Motor1 Thrust (N)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        Fx_filt = lfilter(b, a, dat_file['Motor1 Fx (N)'])
        Fx_avg = np.mean(dat_file['Motor1 Fx (N)'][int(t_min * fs):int(t_max * fs)])
        Fx_err = 1.96 * np.std(dat_file['Motor1 Fx (N)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        Fy_filt = lfilter(b, a, dat_file['Motor1 Fy (N)'])
        Fy_avg = np.mean(dat_file['Motor1 Fy (N)'][int(t_min * fs):int(t_max * fs)])
        Fy_err = 1.96 * np.std(dat_file['Motor1 Fy (N)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        Q_filt = lfilter(b, a, dat_file['Motor1 Torque (Nm)'])
        Q_avg = np.mean(dat_file['Motor1 Torque (Nm)'][int(t_min * fs):int(t_max * fs)])
        Q_err = 1.96 * np.std(dat_file['Motor1 Torque (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        Mx_filt = lfilter(b, a, dat_file['Motor1 Mx (Nm)'])
        Mx_avg = np.mean(dat_file['Motor1 Mx (Nm)'][int(t_min * fs):int(t_max * fs)])
        Mx_err = 1.96 * np.std(dat_file['Motor1 Mx (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        My_filt = lfilter(b, a, dat_file['Motor1 My (Nm)'])
        My_avg = np.mean(dat_file['Motor1 My (Nm)'][int(t_min * fs):int(t_max * fs)])
        My_err = 1.96 * np.std(dat_file['Motor1 My (Nm)'][int(t_min * fs):int(t_max * fs)]) / np.sqrt(N_rev)

        # Resolves measured moments and forces into hub frame
        Mx_avg = (Fy_avg * np.cos(22.5 * np.pi / 180) + Fx_avg * np.sin(22.5 * np.pi / 180)) * d
        My_avg = (Fx_avg * np.cos(22.5 * np.pi / 180) - Fy_avg * np.sin(22.5 * np.pi / 180)) * d
        Fx_avg = My_avg / d * np.cos(22.5 * np.pi / 180) + Mx_avg / d * np.sin(22.5 * np.pi / 180)
        Fy_avg = My_avg / d * np.sin(22.5 * np.pi / 180) + Mx_avg / d * np.cos(22.5 * np.pi / 180)

        fs_acs = dat_file['Sampling Rate'][()]
        OASPL = 10*np.log10(np.mean((dat_file['Acoustic Data'][()].transpose()[int(t_min * fs_acs):int(t_max * fs_acs)]/(dat_file['Sensitivities'][:]*1e-3))**2,axis = 0)/20e-6**2)

    return {'rpm':rpm,'t_rpm':t_rpm,'rpm_avg':rpm_avg,'N_rev':N_rev,'T_filt':T_filt,'T_avg':T_avg,'T_err':T_err,'Fx_filt':Fx_filt,'Fx_avg':Fx_avg,'Fx_err':Fx_err,'Fy_filt':Fy_filt,'Fy_avg':Fy_avg,'Fy_err':Fy_err,'Q_filt':Q_filt,'Q_avg':Q_avg,'Q_err':Q_err,'Mx_filt':Mx_filt,'Mx_avg':Mx_avg,'Mx_err':Mx_err,'My_filt':My_filt,'My_avg':My_avg,'My_err':My_err,'OASPL':OASPL}

#%%

# path to directory containing the rpm sweep cases
dir = '/Users/danielweitsman/Box/Jan21Test/dan_thesis/runs/rpm_sweep/h2b'

# If you want to compare specific points in this directory their names can be specified in caseName list. Otherwise,
# all the cases in this directory directory would be compared and used to generate the thrust/torque profiles.
caseName = []

# Set equal to "True" in order to plot the thrust/torque/rpm time series for each run. These plots are not saved but are
# useful for determining the averaging interval to use for the thrust/torque profiles.
plot_tseries = False
save_fig = False

#   moment arm length from LC to rotor hub [m]
d = 0.18542

#   mic number
mic = 10
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
    cases = [os.path.basename(dir) + str(x) for x in sorted([int(x[3:]) for x in os.listdir(dir)])]
else:
    cases = caseName

output = np.array(list(map(getLoads,cases)))


# %%     T and Q profile

# #   initializes figures
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
# ax.set_title(os.path.basename(dir))
#
# ax.errorbar(rpm_avg, T_avg, yerr=T_err, fmt='-.o')
# ax.set_ylabel('Thrust, T (N)')
# ax.set_xlabel('RPM')
# # ax.set_xlim([1000, 5250])
# # ax.set_ylim([0, 70])
# ax.grid()
#
# if save_fig:
#
#     #   Creates a new figures folder in the parent directory where to save the thrust and torque profiles
#     if not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures')):
#         os.mkdir(os.path.join(os.path.dirname(dir), 'Figures'))
#     elif not os.path.exists(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir))):
#         os.mkdir(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir)))
#
#     plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'T_profile.png'), format='png')
#
# # %%
#
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
# ax.set_title(os.path.basename(dir))
# ax.errorbar(rpm_avg, Q_avg, yerr=Q_err, fmt=':o')
# ax.set_ylabel('Torque, Q (Nm)')
# ax.set_xlabel('RPM')
# # ax.set_xlim([1000, 5250])
# # ax.set_ylim([-1.5, 1.5])
# ax.grid()
#
# if save_fig:
#     plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'Q_profile.png'), format='png')

# %%
#
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
# ax.set_title(os.path.basename(dir))
# ax.errorbar(rpm_avg, Mx_avg, yerr=Q_err, fmt=':o')
# ax.set_ylabel('Rolling Moment, Mx (Nm)')
# ax.set_xlabel('RPM')
# # ax.set_xlim([1000, 5250])
# # ax.set_ylim([-1.5, 1.5])
# ax.grid()
#
# if save_fig:
#     plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'Mx_profile.png'), format='png')
#
# # %%
#
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
# ax.set_title(os.path.basename(dir))
# ax.errorbar(rpm_avg, My_avg, yerr=Q_err, fmt=':o')
# ax.set_ylabel('Pitching Moment, My (Nm)')
# ax.set_xlabel('RPM')
# # ax.set_xlim([1000, 5250])
# # ax.set_ylim([-1.5, 1.5])
# ax.grid()
#
# if save_fig:
#     plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'My_profile.png'), format='png')

# %%
Fx = []
[Fx.append(x['Fx_avg']) for x in output]
Fy = []
[Fy.append(x['Fy_avg']) for x in output]
Mx = []
[Mx.append(x['Mx_avg']) for x in output]
My = []
[My.append(x['My_avg']) for x in output]
OASPL = []
[My.append(x['OASPL']) for x in output]

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.set_title(os.path.basename(dir))
ax.plot(My,OASPL[:,mic],marker = 'o')
ax.set_ylabel('$OASPL, \ dB \ (re: 20 \mu Pa)$')
ax.set_xlabel('Pitching Moment, My (Nm)')
ax.set_xlim([-.15, .07])
ax.set_ylim([86, 87.5])
ax.grid()

if save_fig:
    plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'My_profile.png'), format='png')

#%%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.set_title(os.path.basename(dir))
ax.plot(Mx,OASPL[:,mic],marker = 'o')
ax.set_ylabel('$OASPL, \ dB \ (re: 20 \mu Pa)$')
ax.set_xlabel('Rolling Moment, Mx (Nm)')
ax.set_xlim([-.15, .07])
ax.set_ylim([86, 87.5])
ax.grid()

if save_fig:
    plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'My_profile.png'), format='png')

# %%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.set_title(os.path.basename(dir))
ax.plot(Fy,OASPL[:,mic],marker = 'o')
ax.set_ylabel('$OASPL, \ dB \ (re: 20 \mu Pa)$')
ax.set_xlabel('Hub Force, Fy (N)')
ax.set_xlim([-.75, .75])
ax.set_ylim([86, 87.5])
ax.grid()

if save_fig:
    plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'My_profile.png'), format='png')

# %%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.set_title(os.path.basename(dir))
ax.plot(Fx,OASPL[:,mic],marker = 'o')
ax.set_ylabel('$OASPL, \ dB \ (re: 20 \mu Pa)$')
ax.set_xlabel('Side Force, Fx (N)')
ax.set_xlim([-.75, .75])
ax.set_ylim([86, 87.5])
ax.grid()

if save_fig:
    plt.savefig(os.path.join(os.path.dirname(dir), 'Figures', os.path.basename(dir), 'My_profile.png'), format='png')

# %%

# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.

# load_dat = {'rpm_avg_2': rpm_avg_2, 'T_avg_2': T_avg_2,
#             'T_err_2': T_err_2, 'Q_avg_2': Q_avg_2, 'Q_err_2': Q_err_2,
#             'rpm_avg_1': rpm_avg, 'T_avg_1': T_avg,
#             'T_err_1': T_err, 'Q_avg_1': Q_avg, 'Q_err_1': Q_err, 'T_tot': T_tot, 'T_tot_err': T_tot_err,
#             'Q_tot': Q_tot, 'Q_tot_err': Q_tot_err}
#
# if os.path.exists(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5')):
#     os.remove(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5'))
#
# with h5py.File(os.path.join(os.path.dirname(dir), os.path.basename(dir) + '.h5'), 'a') as f:
#     for k, dat in load_dat.items():
#         f.create_dataset(k, shape=np.shape(dat), data=dat)
