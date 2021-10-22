import os
import h5py
import numpy as np
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/danielweitsman/Desktop/Masters_Research/py scripts/WOPWOP_PostProcess/pyWopwop')
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
import scipy.optimize  as opt
from time import process_time
#%% Sets font parameters

fontName = 'Times New Roman'
fontSize = 12
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%
def eval_pressure(f):
    '''
    This function takes in an arbitrary quantity that is already resolved in retarted time for a single blade and
    computes the corresponding total pressure time series due to this quantity. This pressure time series takes into
    account the contributions from all the blades.
    :param f: noise source strength evaluated at the retarded time around the rotor disk.
    :return:
    :param p: total pressure time series, which includes the contributions from each blade.
    '''

    #   computes the pressure time series attributed to this quantity
    p = 1 / (4 * np.pi * a0 ) * f
    p = np.trapz(p,dx = dr,axis =1)
    # p = np.sum(p,axis = 1)*dr
    # p=np.sum(p, axis=1) * dr
    #    subtract our the DC offset
    p_tot = p - np.expand_dims(np.mean(p, axis=1), axis=1)
    #   rearranges and appends an additional array, which accounts for the contributions from the remaining rotor blades.
    p_tot = np.array([np.concatenate((p_tot[:,:-1][:, int(360 / UserIn['Nb']*Nb):], p_tot[:,:-1][:,:int(360 / UserIn['Nb']*Nb)]), axis=1) for Nb in range(UserIn['Nb'])])
    #   sums contributions from each rotor blade to determine the total pressure time series.
    p_tot = np.sum(p_tot,axis = 0)
    return p_tot

def linterp_ind(ind,quant):
    '''
    This function linearly interpolates a quantity with the first three dimensions being [N_obs x N_r x N_psi] and having an arbitrary number of additional dimensions.
    :param ind:  matrix of indicies at which to evaluate the quantitiy. It must have the same dimensions as "qunt".
    :param quant: values of a quantity at every observer, radial position, and azimuthal angle. The remaining dimensions can be an arbitrary length.
    :return:
    :param y: linearly interpolated array having the same shape as "quant".
    '''

    dim_diff =len(np.shape(quant)) - len(np.shape(ind))
    if dim_diff !=0 :
        for i in range(dim_diff):
            ind = np.expand_dims(ind,axis = len(np.shape(ind))+i)

    x0 = np.floor(ind).astype(int)
    x1 = np.ceil(ind).astype(int)
    y0,y1 = list(map(lambda x: np.array([quant[m_ind,r_ind,x[m_ind,r_ind]] for r_ind in range(np.shape(quant)[1]) for m_ind in range(len(quant))]).reshape((np.shape(quant)),order = 'F'), [x0,x1]))
    y = ((y0*(x1-ind)+y1*(ind-x0))/(x1-x0))
    return y

#%%
pred_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b69/'
mics = [5,16,-4]
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
UserIn['rotation'] == 2
loadParams['omega'] = loadParams['omega']*2*np.pi/60
dphi = (UserIn['psimax']-UserIn['psimin'])/(UserIn['nbpsi']-1)*np.pi/180
phi = (np.arange(-int(UserIn['nbpsi']/2),int(UserIn['nbpsi']/2)+1)*dphi)[::-1]
x = np.array([UserIn['radius']*np.cos(phi)*np.cos(UserIn['thetamax']*np.pi/180),UserIn['radius']*np.cos(phi)*np.sin(UserIn['thetamax']*np.pi/180),UserIn['radius']*np.sin(phi)])

#%%
dr = (geomParams['R']-geomParams['e'])/(geomParams['nXsecs']-1)
psi = (np.arange(361) * np.pi / 180)
a0 = 340
dt = (loadParams['omega']/(2*np.pi))**-1/(len(psi)-1)
ts = np.arange(len(psi))*dt

if UserIn['rotation'] != 1:
    psi = psi[::-1]

y = np.array([np.expand_dims(geomParams['rdim'],axis = 1)*np.cos(psi),np.expand_dims(geomParams['rdim'],axis = 1)*np.sin(psi),np.zeros((len(geomParams['rdim']),len(psi)))])

r = np.expand_dims(np.expand_dims(x.transpose(),axis = 2),axis = 3)-np.expand_dims(y,axis = 0)
r_mag = np.linalg.norm(r, axis=1)

#%%

if len(np.shape(loadParams['dFz2'])) ==1:
    Fz = np.expand_dims(loadParams['dFz'],axis = 1)*np.ones(len(psi)+1)
    Fx = np.expand_dims(loadParams['dFx'],axis = 1)*np.ones(len(psi)+1)
else:
    Fx = loadParams['dFx2'].transpose()
    Fz = loadParams['dFz2'].transpose()


if UserIn['rotation'] == 1:
    li = -np.array([Fx*np.sin(psi),-Fx*np.cos(psi),Fz])
else:
    li = -np.array([Fx * np.sin(psi), Fx * np.cos(psi), Fz])

dD_dt = np.gradient(Fx,axis = 1,edge_order = 2)/dt
dT_dt = np.gradient(Fz,axis = 1,edge_order = 2)/dt

if UserIn['rotation'] == 1:
    dli_dt =  -np.array([(dD_dt * np.sin(psi) + Fx * loadParams['omega'] * np.cos(psi)), (-dD_dt * np.cos(psi) + Fx * loadParams['omega'] * np.sin(psi)), dT_dt])
else:
    dli_dt =  -np.array([(dD_dt * np.sin(psi) + Fx * loadParams['omega'] * np.cos(psi)), (dD_dt * np.cos(psi) - Fx * loadParams['omega'] * np.sin(psi)), dT_dt])

#%%

M = np.expand_dims(loadParams['omega']*geomParams['rdim']/a0,axis = 1)
if UserIn['rotation'] == 1:
    Mi = np.array([-M*np.sin(psi),M*np.cos(psi),np.zeros((len(geomParams['rdim']),len(psi)))])
else:
    Mi = np.array([-M*np.sin(psi),-M*np.cos(psi),np.zeros((len(geomParams['rdim']),len(psi)))])

if UserIn['rotation'] == 1:
    dMi_dt = np.array([-M*loadParams['omega']*np.cos(psi),-M*loadParams['omega']*np.sin(psi),np.zeros((len(geomParams['rdim']),len(psi)))])
else:
    dMi_dt = np.array([-M*loadParams['omega']*np.cos(psi),M*loadParams['omega']*np.sin(psi),np.zeros((len(geomParams['rdim']),len(psi)))])

#%%

x0 = ts-r_mag/a0
ts_exp = (np.ones(np.shape(x0))*ts).flatten()
start_t = process_time()
f = lambda x: x - ts_exp + linterp_ind(quant=r_mag, ind= (x.reshape(np.shape(r_mag))* loadParams['omega'] * 180 / np.pi) % 360).flatten() / a0
t_ret = opt.newton(func = f , x0 = x0.flatten()).reshape(np.shape(r_mag))
# out2 = np.array([opt.newton(func = min_res , x0 = t.flatten(),args = [ts[i]]) for i,t in enumerate(x0)]).transpose().reshape(31,48,361)
print(process_time() - start_t)
t_ret_ind =(t_ret*loadParams['omega']*180/np.pi)%360

#%%
Mr,dMr_dt,lr,dlr_dt = list(map(lambda f: np.sum(r * f, axis = 1)/r_mag, [Mi,dMi_dt,li,dli_dt]))
r_r,Mr_r,dMr_dt_r,lr_r,dlr_dt_r= list(map(lambda f:linterp_ind(t_ret_ind,f), [r.transpose(0,2,3,1),Mr,dMr_dt,lr,dlr_dt]))
r_r_mag =  np.linalg.norm(r_r, axis=-1)

#%%
term1 = dlr_dt_r/(r_r_mag*(1-Mr_r)**2)
term2 = lr_r/(r_r_mag*(1-Mr_r)**3)*dMr_dt_r
p_term1, p_term2, p_tot = list(map(lambda quant: eval_pressure(quant),[term1,term2,term1+term2]))
OASPL_term1, OASPL_term2, OASPL_tot = list(map(lambda f: 10*np.log10(np.mean(f**2,axis = 1)/20e-6**2),[p_term1, p_term2, p_tot]))

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(len(mics),1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)

#   Loops through each mic
for i,m in enumerate(mics):
#   Plots the resulting spectra in dB
        if len(mics)>1:
            ax[i].plot(ts[:-1]/(dt*360), p_term1[m-1])
            ax[i].plot(ts[:-1]/(dt*360), p_term2[m-1])
            ax[i].plot(ts[:-1]/(dt*360), p_tot[m-1])
        else:
            ax.plot(ts[:-1]/(dt*360), p_term1[m-1])
            ax.plot(ts[:-1]/(dt*360), p_term2[m-1])
            ax.plot(ts[:-1]/(dt*360), p_tot[m-1])

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
            ax[i].plot(ts[:-1]/(dt*360), p_tot[m-1])
        else:
            ax.plot(ts[:-1]/(dt*360), p_tot[m-1])

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
ax.plot(phi , OASPL_term1)
ax.plot(phi , OASPL_term2)
ax.plot(phi , OASPL_tot)
ax.set_thetamax(phi[0]*180/np.pi+2)
ax.set_thetamin(phi[-1]*180/np.pi-2)
ax.set_ylim([0,60])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.25),  labelpad = -20, rotation = phi[-1]*180/np.pi-3)
# ax.legend(['Measured','Predicted'], ncol=1,loc='center',bbox_to_anchor=(.25, 0.9))
ax.legend(['$\partial l_r/\partial t$','$l_r(1-M_r)^{-1}(\partial M_r/\partial t)$','$\partial l_r/\partial t+l_r(1-M_r)^{-1}(\partial M_r/\partial t)$'], ncol=1,loc='center',bbox_to_anchor=(-.115, 0.9))

#%%
# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.
if save_h5:
    sdata = {'phi':phi,'dt':dt,'x':x,'y':y,'r':r,'ts':ts,'tr':t_ret,'tr_ind':t_ret_ind,'Mr_r':Mr_r,'dMr_dt_r':dMr_dt_r,'lr_r':lr_r,'dlr_dt_r':dlr_dt_r,'term1':term1,'term2':term2,'p_total':p_tot,'OASPL_term1':OASPL_term1,'OASPL_term2':OASPL_term2,'OASPL_tot':OASPL_tot}

    if os.path.exists(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_form1a_LN_sdata.h5')):
        os.remove(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_form1a_LN_sdata.h5'))

    with h5py.File(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_form1a_LN_sdata.h5'), 'a') as h5_f:
        for k, dat in sdata.items():
            h5_f.create_dataset(k, shape=np.shape(dat), data=dat)

            #%%

# fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
# quant =Fx[:,:-1].transpose()
# # levels = np.linspace(-.005, .005, 50)
# levels = np.linspace(np.min(quant), np.max(quant), 50)
# dist = ax.contourf(psi, geomParams['rdim'], quant.transpose(), levels=levels)
# ax.set_ylim(geomParams['rdim'][0], geomParams['rdim'][-1])
# cbar = fig.colorbar(dist)
# cbar.ax.set_ylabel('$dFz \: [N]$')

#%%
#   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(1,1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
# ax.plot(psi,dlr_dt[5,30])
# ax.plot(psi[:-1],dlr_dt_r[5,30])
# ax.plot(psi,t_ret_ind[5,30])

# ax.plot(psi[:-1],p_tot2[0,5])
# ax.plot(psi[:-1],p_tot2[1,5])

#%%
# #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(1,1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
# ax.plot(psi,Fz[30,:-1])
# #%%
# #   Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
# fig,ax = plt.subplots(1,1,figsize = (8,6))
# #   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
# plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
# ax.plot(psi,np.diff(Fz[30]))