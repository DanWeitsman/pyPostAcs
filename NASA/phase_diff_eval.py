import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt


#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
#   Directory of data files.
dir ='/Users/danielweitsman/Downloads/NASA_data/ddhcs10'

#   Starting time from which to compute the spectra
start_t = 1
#   End time to which to compute the spectra
end_t = 29

#%%
#   Opens and reads in the acoustic and TAC data from the h5 files

with h5py.File(os.path.join(dir, 'acs_data.h5'), 'r') as dat_file:
    ttl_1 = dat_file['Motor1 RPM'][()]
    ttl_2 = dat_file['Motor2 RPM'][()]
    fs_acs = dat_file['Sampling Rate'][()]
    fs_ttl = round((np.mean(np.diff(dat_file['Time (s)'])))**-1)

#%%
# generates time vectors for the tac and acoustic data
t = np.arange(len(ttl_1))/fs_ttl

# evaluates the leading edge of the tac pulses, start/end time rpm indices, nominal rpm, and rpm confidence limit for
# each rotor.
LE_ind1, lim_ind1, rpm_nom1, u_rpm1 = fun.rpm_eval(ttl_1,fs_ttl,start_t,end_t)
LE_ind2, lim_ind2, rpm_nom2, u_rpm2 = fun.rpm_eval(ttl_2,fs_ttl,start_t,end_t)

if abs(rpm_nom1-rpm_nom2)/rpm_nom2 > 0.01:
    print('Caution: The difference between the rotational rates of the upper and lower rotors are significant. This '
          'technique may not yield accurate results.')

# averages the nominal rpm between the upper and lower rotor for the separated case
rpm_avg = np.mean((rpm_nom1,rpm_nom2))

# number of total revs considered
Nrev = int(np.floor((end_t-start_t)*rpm_avg/60))

# computes the azimuthal offset between the upper and lower rotor for the separated cases (lower-upper)
dphi = (LE_ind1[lim_ind1[0]:lim_ind1[0]+Nrev]-LE_ind2[lim_ind2[0]:lim_ind2[0]+Nrev])/fs_ttl*rpm_avg/60*360
# uses the indices of the upper rotor to generate the resulting time vector
t_rpm = t[LE_ind1[lim_ind1[0]:lim_ind1[0]+Nrev]]

#%%
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
ax.plot(t_rpm,dphi)
ax.set_ylabel('$Azimuthal \ Offset,\ \Delta \psi \ [\circ]$')
ax.set_xlabel('Time (sec)')
ax.set_xlim([start_t, end_t])
# ax.set_ylim([4.4e3, 4.5e3])
ax.grid()
