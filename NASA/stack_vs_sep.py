#%%

# Xn_ext = np.concatenate((xn_ref_filt,np.zeros(len(acs_sep) - len(Xn_avg))))
# Xn_ext_ms = np.mean(Xn_ext**2)
#
# Rxy=fun.xCorr(Xn_ext, acs_sep, fs_acs, fs_acs)
# Rxy_ind = np.squeeze(np.where(np.equal(Rxy <= 1.01*Xn_ext_ms,Rxy >= 0.99*Xn_ext_ms)))
# # Rxy_ind = np.squeeze(np.where(Rxy>=np.max(Rxy)*0.2))
# min_ind = f[1]**-1*fs_acs
# Rxy_ind2= np.squeeze(np.where(np.diff(Rxy_ind) >= min_ind))
# Rxy_ind2 = np.insert(Rxy_ind2,[0,len(Rxy_ind2)],[0,-1])
# # Rxy_ind3 = np.array([np.squeeze(np.where(Rxy == np.max(Rxy[Rxy_ind[ind]:Rxy_ind[Rxy_ind2[i+1]]]))) for i,ind in enumerate(Rxy_ind2[:-1])])
# Rxy_ind3 = np.array([np.squeeze(np.where(abs(Rxy-Xn_ext_ms) == np.min(abs(Rxy[Rxy_ind[ind]:Rxy_ind[Rxy_ind2[i+1]]]-Xn_ext_ms)))) for i,ind in enumerate(Rxy_ind2[:-1])])
#
# #%%
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
# ax.plot(t_acs,Rxy)
# ax.scatter(t_acs[Rxy_ind],Rxy[Rxy_ind])
# ax.scatter(t_acs[Rxy_ind3],Rxy[Rxy_ind3])
# ax.set_ylabel('$R_{xy}$')
# ax.set_xlabel('Time (sec)')
# # ax.set_xlim([-.15, .07])
# # ax.set_ylim([86, 89])
# ax.grid()
#
# #%%
# # ind_acs = list(map(lambda x: bisect.bisect(t_acs,x),t_rpm))
# # Rxy_ind_acs = list(map(lambda x: bisect.bisect(ind_acs,x),Rxy_ind3))
# # Rxy_ind_acs=list(filter(lambda x: x !=0, Rxy_ind_acs))
# acs_data_list = np.array([acs_sep[i:int(i + N_avg)] for i in Rxy_ind3[:-1]])
# acs_list_flat = np.concatenate(acs_data_list)
# # N2 = np.max([len(x) for x in acs_data_list])
#
# Xn2_avg = np.mean(acs_data_list,axis = 0)
# # Xn2_avg = np.mean(upsampled_Xm_xn[:,1,:],axis = 0)
#
# #%%
# # xn2_filt = filtfilt(b,a,Xn2_avg)
# xn2_filt = list(map(lambda x:filtfilt(b,a,x),acs_data_list ))
# #%%
# f_sep_ms,Gxx_sep_ms,Gxx_sep_avg_ms = fun.msPSD(acs_sep[int(fs_acs * start_t):int(fs_acs * end_t)], fs = fs_acs, df = df_ms, save_fig = False, plot = False)
# f2_ms,Gxx2_ms,Gxx2_avg_ms = fun.msPSD(acs_list_flat, fs = fs_acs, df = df_ms, save_fig = False, plot = False)
#
# # N2_avg = np.floor(np.mean(list(map(lambda x: len(x),acs_data_list))))
# # fs2 = N/N2_avg*fs_acs
# # dt2 = fs2**-1
# # df2 = (N*dt2)**-1
# #
# # f2 =np.arange(int(N/2))*df
# # Sxx2 = (dt*N)**-1 * abs(Xm2_avg) ** 2
# # Gxx2 = Sxx2[:int(N/2)]
# # Gxx2[1:-1]= 2*Gxx2[1:-1]
#
# #%%
#
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
# # ax.plot(dt*np.arange(N),Xn_avg)
# ax.plot(dt*np.arange(N),xn_ref_filt,linewidth = 5, c = 'black')
# # ax.plot(dt*np.arange(1081),Xn2_avg)
# # ax.plot(dt*np.arange(N_avg),xn2_filt)
# for x in xn2_filt[20:30]:
#     ax.plot(dt*np.arange(N_avg),x)
# ax.set_ylabel('Pressure [Pa]')
# ax.set_xlabel('Time (sec)')
# # ax.set_xlim([-.15, .07])
# # ax.set_ylim([86, 89])
# ax.grid()
#
# #%%
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5))
# # ax.stem(f,10*np.log10(Gxx*df/20e-6**2))
# # ax.stem(f,10*np.log10(Gxx2*df2/20e-6**2))
#
# ax.plot(f_ms,10*np.log10(Gxx_avg_ms*df_ms/20e-6**2),label = 'Stacked')
# ax.plot(f_ms,10*np.log10(Gxx_sep_avg_ms*df_ms/20e-6**2),label = 'Seperated')
#
# ax.plot(f2_ms,10*np.log10(Gxx2_avg_ms*df_ms/20e-6**2),label ='Corrected' )
#
# ax.set_xscale('log')
# ax.set_ylabel('$\overline{X_m}, \ dB \ (re: 20 \mu Pa)$')
# ax.set_xlabel('Frequency (Hz)')
# ax.set_xlim([30, 15e3])
# # ax.set_ylim([86, 89])
# plt.legend()
# ax.grid()