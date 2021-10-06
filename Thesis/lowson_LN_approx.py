import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/danielweitsman/Desktop/Masters_Research/py scripts/WOPWOP_PostProcess/pyWopwop')
import matplotlib.colors as mcolors

#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
pred_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b69/'
mics = [3,16,29]
save_h5= True
#%%

UserIn = {}
geomParams = {}
loadParams = {}
#   imports several performance quantities from the MainDict.h5 file.
with h5py.File(os.path.join(pred_dir, 'MainDict.h5'), "r") as f:

    for k, v in f[list(f.keys())[0]]['geomParams'].items():
        geomParams = {**geomParams, **{k: v[()]}}

    for k, v in f[list(f.keys())[0]]['loadParams'].items():
        loadParams = {**loadParams, **{k: v[()]}}

    for k,v in f[list(f.keys())[1]].items():
        UserIn={**UserIn,**{k:v[()]}}

#%%

dphi = (UserIn['psimax']-UserIn['psimin'])/(UserIn['nbpsi']-1)*np.pi/180
phi = (np.arange(-int(UserIn['nbpsi']/2),int(UserIn['nbpsi']/2)+1)*dphi)[::-1]
x = np.array([UserIn['radius']*np.cos(phi)*np.cos(UserIn['thetamax']*np.pi/180),UserIn['radius']*np.cos(phi)*np.sin(UserIn['thetamax']*np.pi/180),UserIn['radius']*np.sin(phi)])

#%%

a0 = 340
psi = np.arange(361)*np.pi/180
dt = (loadParams['omega']/(2*np.pi))**-1/(len(psi)-1)
ts = np.arange(len(psi))*dt

y = np.array([geomParams['R']*np.cos(psi),geomParams['R']*np.sin(psi),np.zeros(len(psi))])
r = np.expand_dims(x.transpose(),axis = 2)-y

r_shift = np.empty(np.insert(np.shape(r),0,UserIn['Nb']))
# tr = np.empty((UserIn['Nb'],np.shape(x)[1],len(psi)))
for Nb in range(UserIn['Nb']):
    r_shift[Nb] = np.concatenate((r[:,:, int(360 / UserIn['Nb']*Nb):], r[:, :,:int(360 / UserIn['Nb']*Nb)]), axis=2)

t_shift = np.linalg.norm(r_shift, axis=2)/a0
t_shift_ind = (t_shift*loadParams['omega'])%(2*np.pi)*180/np.pi
tr_ind = np.empty(np.shape(t_shift_ind))
for Nb in range(UserIn['Nb']):
    tr_ind[Nb] = np.round((psi*180/np.pi+360/UserIn['Nb']*Nb)%360-t_shift_ind[Nb])
tr_ind = tr_ind.astype(int)
# tr_ind = np.round((tr*loadParams['omega'])%(2*np.pi)*180/np.pi).astype(int)
#
# for Nb in range(UserIn['Nb']):
#     tr_ind[Nb]= (tr_ind[Nb]+360/UserIn['Nb']*Nb)%360
#
#%%
t1 = np.linalg.norm(r, axis=1) / a0
r2 = np.concatenate((r[:,:, int(360 / UserIn['Nb']*1):], r[:, :,:int(360 / UserIn['Nb']*1)]), axis=2)
t2 =np.linalg.norm(r2, axis=1) / a0
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(1,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
# ax.plot(tr[0,15,:])
# ax.plot(tr[1,15,:])
ax.plot(t1[15,:])
ax.plot(t2[15,:])
# ax.plot((-t1+2*np.expand_dims(np.linalg.norm(x, axis=0) / a0,axis =1))[15,:])

ax.legend(['B1','B2','B1','B2'])
# r1 = np.sqrt(x[0]**2+x[-1]**2)
# r2 = r1-(x[0]*geomParams['R']/r1*np.expand_dims(np.cos(psi),axis = 1))

#%%
dr = geomParams['R']/geomParams['nXsecs']

if len(np.shape(loadParams['dFz'])) ==1:
    dFz = np.expand_dims(loadParams['dFz'],axis = 1)*np.ones(len(psi))
    dFx = np.expand_dims(loadParams['dFx'],axis = 1)*np.ones(len(psi))

Fx = dFx*np.cos(loadParams['th'][0])+dFz*np.sin(loadParams['th'][0])
Fz = -dFx*np.sin(loadParams['th'][0])+dFz*np.cos(loadParams['th'][0])

Fz_comp = 0.4*np.trapz(Fz,x=geomParams['r'],axis = 0)
Fx_comp = 0.4*np.trapz(Fx,x=geomParams['r'],axis = 0)

Fi = np.array([Fx_comp*np.sin(psi),-Fx_comp*np.cos(psi),Fz_comp])
li = -Fi
li_dt = np.array([-Fx_comp*loadParams['omega']*np.cos(psi),-Fx_comp*loadParams['omega']*np.sin(psi),np.zeros(len(psi))])

#%%

M = loadParams['omega']*geomParams['R']/a0
Mi = np.array([-M*np.sin(psi),M*np.cos(psi),np.zeros(len(psi))])
Mr = np.sum(r*Mi,axis = 1)/np.linalg.norm(r,axis =1)

dMi_dt = np.array([-M*loadParams['omega']*np.cos(psi),-M*loadParams['omega']*np.sin(psi),np.zeros(len(psi))])
dMr_dt = np.sum(r*dMi_dt,axis = 1)/np.linalg.norm(r,axis =1)

#%%
lr = np.sum(r*li,axis = 1)/np.linalg.norm(r,axis =1)
dlr_dt = np.sum(r*li_dt,axis = 1)/np.linalg.norm(r,axis =1)


#%%

# Mr_r2 = np.empty(np.shape(tr_ind))
# for Nb in range(UserIn['Nb']):
#     for m in range(9):
#         Mr_r2[Nb,m] = np.array([Mr[m,tr_ind[Nb,m]]])

# for Nb in range(UserIn['Nb']):
#     Mr_r,dMr_dt_r,lr_r,dlr_dt_r = list(map(lambda f: np.array([f[m,tr_ind[Nb,m]] for m in range(np.shape(x)[1])]),[Mr,dMr_dt,lr,dlr_dt]))

Mr_r,dMr_dt_r,lr_r,dlr_dt_r = [var.reshape(np.shape(tr_ind)) for var in list(map(lambda f: np.array([f[m,tr_ind[Nb,m]] for Nb in range(UserIn['Nb']) for m in range(np.shape(x)[1])]),[Mr,dMr_dt,lr,dlr_dt]))]

#%%

term1 = dlr_dt_r
term2 = lr_r/(1-Mr_r)*dMr_dt_r

p_term1 = np.sum(1/(4*np.pi*a0*np.linalg.norm(r,axis =1)*(1-Mr_r)**2)*(term1),axis = 0)
p_term1 = p_term1-np.expand_dims(np.mean(p_term1,axis = 1),axis = 1)
p_term2 = np.sum(1/(4*np.pi*a0*np.linalg.norm(r,axis =1)*(1-Mr_r)**2)*(term2),axis =0)
p_term2 = p_term2-np.expand_dims(np.mean(p_term2,axis = 1),axis = 1)

p = 1/(4*np.pi*a0*np.linalg.norm(r_shift,axis =2)*(1-Mr_r)**2)*(term1+term2)
p_total = np.sum(p,axis = 0)
p_total = p_total-np.expand_dims(np.mean(p_total,axis = 1),axis = 1)
OASPL = 10*np.log10(np.mean(p_total**2,axis = 1)/20e-6**2)

#%%
p2 = np.empty(np.shape(t_shift_ind))
for Nb in range(UserIn['Nb']):
    p2[Nb] = np.concatenate((p[0,:, int(360 / UserIn['Nb']*Nb):], p[0,:,:int(360 / UserIn['Nb']*Nb)]), axis=1)
p2_tot =  np.sum(p,axis = 0)
p2_total = p_total-np.expand_dims(np.mean(p_total,axis = 1),axis = 1)

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(ts/(dt*360), p_term1[m-1])
            ax[i].plot(ts/(dt*360), p_term2[m-1])
            ax[i].plot(ts/(dt*360), p_total[m-1])
        else:
            ax.plot(ts/(dt*360), p_term1[m-1])
            ax.plot(ts/(dt*360), p_term2[m-1])
            ax.plot(ts/(dt*360), p_total[m-1])

for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1]*180/np.pi)}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xlim([0,1])
    # ax[i].set_ylim([-0.015,0.015])
    ax[i].grid('on')

ax[int(len(mics)/2)].set_ylabel('Pressure [Pa]')
ax[- 1].set_xlabel('Roatation')
ax[-1].legend(['$\partial l_r/\partial t$','$l_r(1-M_r)^{-1}(\partial M_r/\partial t)$','$\partial l_r/\partial t+l_r(1-M_r)^{-1}(\partial M_r/\partial t)$'], ncol=3,loc='center',bbox_to_anchor=(0.5, -0.55))

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(ts/(dt*360), p_total[m-1])
        else:
            ax.plot(ts/(dt*360), p_total[m-1])

for i, m in enumerate(mics):
    ax[i].set_title(f'$Mic\ {m} \ ( \phi = {round(phi[m-1]*180/np.pi)}^\circ)$')
    if i!=len(mics)-1:
        ax[i].tick_params(axis='x', labelsize=0)
    ax[i].set_xlim([0,1])
    # ax[i].set_ylim([-0.015,0.015])
    ax[i].grid('on')

ax[int(len(mics)/2)].set_ylabel('Pressure [Pa]')
ax[- 1].set_xlabel('Roatation')


#%%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
ax.plot(phi , OASPL)
ax.set_thetamax(phi[0]*180/np.pi+2)
ax.set_thetamin(phi[-1]*180/np.pi-2)
ax.set_ylim([0,50])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.25),  labelpad = -20, rotation = phi[-1]*180/np.pi-3)
# ax.legend(['Measured','Predicted'], ncol=1,loc='center',bbox_to_anchor=(.25, 0.9))

#%%
# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.
if save_h5:
    sdata = {'phi':phi,'dt':dt,'x':x,'y':y,'r':r,'ts':ts,'tr':tr,'tr_ind':tr_ind,'Mr_r':Mr_r,'dMr_dt_r':dMr_dt_r,'lr_r':lr_r,'dlr_dt_r':dlr_dt_r,'term1':term1,'term2':term2,'p_total':p_total,'OASPL':OASPL}

    if os.path.exists(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_lowson_LN_sdata.h5')):
        os.remove(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_lowson_LN_sdata.h5'))

    with h5py.File(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_lowson_LN_sdata.h5'), 'a') as h5_f:
        for k, dat in sdata.items():
            h5_f.create_dataset(k, shape=np.shape(dat), data=dat)