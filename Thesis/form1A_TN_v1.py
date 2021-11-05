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
import pyvista
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
    p = np.trapz(np.trapz(f,x = np.insert(abs(np.diff(surfNodes[:,0,0],axis = -1)),24,np.diff(surfNodes[:,0,0])[-1]),axis = 1),x = geomParams['rdim'],axis =1)
    # p = np.trapz(np.trapz(f,dx = np.mean(abs(np.diff(surfNodes[:,0,0]))),axis = 1),dx = dr,axis =1)

    # p = np.sum(np.sum(f, axis=1), axis=1)*dc*dr
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
    if dim_diff > 0 :
        for i in range(abs(dim_diff)):
            ind = np.expand_dims(ind,axis = len(np.shape(ind))+i)
    # elif dim_diff <0 :
    #     for i in range(abs(dim_diff)):
    #         print(len(np.shape(quant)))
    #         quant = np.expand_dims(quant,axis = len(np.shape(quant)))

    x0 = np.floor(ind).astype(int)
    x1 = np.ceil(ind).astype(int)
    y0,y1 = list(map(lambda x: np.array([quant[m_ind,c_ind,r_ind,x[m_ind,c_ind,r_ind]] for r_ind in range(np.shape(quant)[2]) for c_ind in range(np.shape(quant)[1]) for m_ind in range(len(quant))]).reshape((np.shape(quant)),order = 'F'), [x0,x1]))
    y = (y0*(x1-ind)+y1*(ind-x0))/(x1-x0)
    return y

#%%
pred_dir ='/Users/danielweitsman/Desktop/Masters_Research/lynx/h2b69_Fz/'
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

loadParams['omega'] = loadParams['omega']*2*np.pi/60
dphi = (UserIn['psimax']-UserIn['psimin'])/(UserIn['nbpsi']-1)*np.pi/180
phi = (np.arange(-int(UserIn['nbpsi']/2),int(UserIn['nbpsi']/2)+1)*dphi)[::-1]
x = np.array([UserIn['radius']*np.cos(phi)*np.cos(UserIn['thetamax']*np.pi/180),UserIn['radius']*np.cos(phi)*np.sin(UserIn['thetamax']*np.pi/180),UserIn['radius']*np.sin(phi)])

#%%
psi_len =len(loadParams['phi'])
dr = (geomParams['R']-geomParams['e'])/(geomParams['nXsecs']-1)
dc = geomParams['chordDist'][0]/(geomParams['pntsPerXsec']/2-1)
# np.diff(abs(surfNodes[:,0,0]))
a0 = UserIn['c']
dt = (loadParams['omega']/(2*np.pi))**-1/(psi_len-1)
ts = np.arange(psi_len)*dt

if UserIn['rotation'] == 1:
    psi =np.arange(psi_len) * np.pi / 180
else:
    psi = ((np.arange(psi_len)-180)%(psi_len-1))*np.pi/180
    # psi = ((np.arange(psi_len)[::-1] - 180) % (psi_len-1))*np.pi/180

#%%
surfNodes = geomParams['surfNodes'].reshape(geomParams['pntsPerXsec'],geomParams['nXsecs'],3,order = 'F')
surfNorms = (geomParams['surfNorms']/np.expand_dims(np.linalg.norm(geomParams['surfNorms'],axis = 1),axis = 1)).reshape(geomParams['pntsPerXsec'],geomParams['nXsecs'],3,order = 'F')

# surfNodes[:, :, 0] = -surfNodes[:, :, 0]
# surfNorms[:, :, 0] = -surfNorms[:, :, 0]

# mesh = pyvista.read('/Users/danielweitsman/Desktop/Masters_Research/lynx/lynx.vtk')
# mesh = pyvista.PolyData(geomParams['surfNodes'])
# mesh.compute_normals(inplace=True).plot()
# pnts = mesh.points
# mesh.points[:,0] = -mesh.points[:,0]
# norms =mesh.point_normals
# wrap = mesh.warp_by_scalar()
# wrap.plot()
surfNorms = np.array((np.expand_dims(surfNorms[:, :, 1], axis=-1) * np.cos(psi) + np.expand_dims(surfNorms[:, :, 0],axis=-1) * np.sin(psi),
              np.expand_dims(-surfNorms[:, :, 0], axis=-1) * np.cos(psi) + np.expand_dims(surfNorms[:, :, 1], axis=-1) * np.sin(psi),
              np.expand_dims(surfNorms[:, :, -1], axis=-1) * np.ones(len(psi))))

y = np.array((np.expand_dims(surfNodes[:, :, 1], axis=2) * np.cos(psi) + np.expand_dims(surfNodes[:, :, 0],axis=-1) * np.sin(psi),
              np.expand_dims(-surfNodes[:, :, 0], axis=2) * np.cos(psi) + np.expand_dims(surfNodes[:, :, 1], axis=-1) * np.sin(psi),
              np.expand_dims(surfNodes[:, :, -1], axis=2) * np.ones(len(psi))))
# y = np.array([np.expand_dims(geomParams['rdim'],axis = 1)*np.cos(psi),np.expand_dims(geomParams['rdim'],axis = 1)*np.sin(psi),np.zeros((len(geomParams['rdim']),psi_len))])

r = np.expand_dims(np.expand_dims(np.expand_dims(x.transpose(),axis = 2),axis = 3),axis =4)-y
r_mag = np.linalg.norm(r, axis=1)

#%%

#try multiplying by r_mag to find v
# v = loadParams['omega']*geomParams['rdim']
v = loadParams['omega']*np.linalg.norm(y[:1,:,:,0],axis = 0)
M = np.expand_dims(v/a0,axis =-1 )
Mi = np.array([-M*np.sin(psi),M*np.cos(psi),np.zeros((geomParams['pntsPerXsec'],geomParams['nXsecs'],psi_len))])
dMi_dt = np.array([-M*loadParams['omega']*np.cos(psi),-M*loadParams['omega']*np.sin(psi),np.zeros((geomParams['pntsPerXsec'],geomParams['nXsecs'],psi_len))])

# vn = np.sum(np.expand_dims(surfNorms.transpose(-1,0,1),axis = -1)*np.expand_dims(Mi*a0,axis = 1),axis = 0)
# vn2 = np.sum(a0*M*surfNorms,axis = -1)
# vn = np.sum(surfNorms*np.expand_dims(Mi,axis = 1)*a0,axis = 0)
# vn = np.repeat(np.expand_dims(np.repeat(np.expand_dims(vn,axis = 0),31,axis = 0),axis = -1),361,axis = -1)
vn = surfNorms[1,:,:,0]*v
dvn_dt = np.sum(dMi_dt*a0*surfNorms,axis = 0)+np.sum(a0*Mi*np.cross([0,0,loadParams['omega']],surfNorms,axis = 0),axis = 0)

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
# # r_r,Mi_r = linterp_ind(t_ret_ind,r.transpose(0,2,3,4,1))
# # r_r_mag =  np.linalg.norm(r_r, axis=-1)
# start_t = process_time()
# r_r,Mi_r = list(map(lambda f: linterp_ind(t_ret_ind,f), [r.transpose(0,2,3,4,1),np.repeat(np.expand_dims(Mi,axis = 0),31,axis = 0).transpose(0,2,3,4,1)]))
# r_r_mag =  np.linalg.norm(r_r, axis=-1)
# Mr_r = np.sum(r_r * Mi_r, axis = -1)/r_r_mag
# print(process_time() - start_t)

#%%

# Mr =np.sum(r * np.expand_dims(Mi,axis = 0), axis = 1)/r_mag
Mr,dMr_dt = list(map(lambda f: np.sum(r * np.expand_dims(f,axis = 0), axis = 1)/r_mag,[Mi,dMi_dt]))
# vn_r=linterp_ind(t_ret_ind,np.expand_dims(np.expand_dims(vn, axis = 0),axis = -1))
# r_r,Mr_r,vn_r= list(map(lambda f:linterp_ind(t_ret_ind,f), [r.transpose(0,2,3,4,1),Mr,vn]))
r_r,Mr_r,dMr_dt_r= list(map(lambda f:linterp_ind(t_ret_ind,f), [r.transpose(0,2,3,4,1),Mr,dMr_dt]))

# r_r,Mr_r,vn_r= list(map(lambda f:linterp_ind(t_ret_ind,f), [r.transpose(0,2,3,4,1),Mr,np.repeat(np.expand_dims(vn,axis = 0),len(t_ret_ind),axis = 0)]))
r_r_mag =  np.linalg.norm(r_r, axis=-1)


#%%

term1 = UserIn['rho']*dvn_dt/(r_r_mag*(1-Mr_r)**2)+UserIn['rho']*np.expand_dims(np.expand_dims(vn,axis = 0),axis = -1)*dMr_dt_r/(r_r_mag*(1-Mr_r)**3)
term2 = UserIn['rho']*a0*np.expand_dims(np.expand_dims(vn,axis = 0),axis = -1)*(Mr_r-M**2)/(r_r_mag**2*(1-Mr_r)**3)

p_term1, p_term2, p_tot = list(map(lambda f:(4*np.pi)**-1*eval_pressure(f), [term1,term2,term1+term2]))
oaspl_term1, oaspl_term2, oaspl_tot = list(map(lambda f:10*np.log10(np.mean(f**2,axis = 1)/20e-6**2), [p_term1, p_term2, p_tot]))

# p = eval_pressure(UserIn['rho']*np.expand_dims(np.expand_dims(vn,axis = 0),axis = -1)/(r_r_mag*(1-Mr_r)))
# p = eval_pressure(UserIn['rho']*vn/(r_r_mag*(1-Mr_r)))
# OASPL = 10*np.log10(np.mean(p**2,axis = 1)/20e-6**2)

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
ax[-1].legend(['far-field','near-field','total'],ncol=3,loc='center',bbox_to_anchor=(0.5, -0.55))

#%%

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.5),subplot_kw=dict(polar=True))
ax.plot(phi , oaspl_term1)
ax.plot(phi , oaspl_term2)
ax.plot(phi , oaspl_tot)
ax.set_thetamax(phi[0]*180/np.pi+2)
ax.set_thetamin(phi[-1]*180/np.pi-2)
ax.set_ylim([0,60])
ax.set_ylabel(' OASPL (dB, re:20$\mu$Pa)',position = (1,.25),  labelpad = -20, rotation = phi[-1]*180/np.pi-3)
# ax.legend(['Measured','Predicted'], ncol=1,loc='center',bbox_to_anchor=(.25, 0.9))
# ax.legend(['$\partial l_r/\partial t$','$l_r(1-M_r)^{-1}(\partial M_r/\partial t)$','$\partial l_r/\partial t+l_r(1-M_r)^{-1}(\partial M_r/\partial t)$'], ncol=1,loc='center',bbox_to_anchor=(-.115, 0.9))
ax.legend(['far-field','near-field','total'])
#%%
# Saves the data to a new h5 file, which could later be referenced tp compare the thrust/torque profiles of different
# rotor configurations.
if save_h5:
    sdata = {'phi':phi,'dt':dt,'x':x,'y':y,'r':r,'ts':ts,'tr':t_ret,'tr_ind':t_ret_ind,'r_r':r_r,'Mr_r':Mr_r,'vn':vn,'p_term1':p_term1,'p_term2':p_term2,'p':p_tot,'oaspl_tot':oaspl_tot}

    if os.path.exists(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_form1a_TN_sdata.h5')):
        os.remove(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_form1a_TN_sdata.h5'))

    with h5py.File(os.path.join(os.path.dirname(pred_dir), os.path.basename(os.path.dirname(pred_dir)) + '_form1a_TN_sdata.h5'), 'a') as h5_f:
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
  # Initializes figure with the number of subplots equal to the number of mics specified in the "mics" list
fig,ax = plt.subplots(1,1,figsize = (8,6))
#   Adds a space in between the subplots and at the bottom for the subplot titles and legend, respectfully.
plt.subplots_adjust(hspace = 0.35,bottom = 0.15)
ax.plot(psi*180/np.pi,Mr[-5,10,-1,:])

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
#%%
# pnts = surfNodes.reshape(np.shape(geomParams['surfNodes']),order = 'F')
# norms = surfNorms.reshape(np.shape(geomParams['surfNodes']),order = 'F')
# pnts[:,0] = -surfNodes_temp[:,0]
# surfNorms_temp[:,0] = -surfNorms_temp[:,0]
# scale = .001
# fig = plt.figure()
# ax = fig.gca(projection = '3d')
# ax.auto_scale_xyz([-2, 2], [10, 60], [-1, 1])
# ax.pbaspect = [.09, 1, .05]
# ax.set(xlabel = 'x',ylabel = 'y',zlabel = 'z')
# ax.scatter3D(geomParams['surfNodes'][:,0], geomParams['surfNodes'][:,1], geomParams['surfNodes'][:,2],c = 'red',linewidths = 1)
# ax.quiver(geomParams['surfNodes'][:,0], geomParams['surfNodes'][:,1], geomParams['surfNodes'][:,2],geomParams['surfNorms'][:,0]*scale, geomParams['surfNorms'][:,1]*scale, geomParams['surfNorms'][:,2]*scale)

# # fig.show()

scale = .005
fig = plt.figure()
ax = fig.gca(projection = '3d')
# ax.auto_scale_xyz([-2, 2], [2, 2], [-2, 2])
# ax.pbaspect = [1, 1, 1]
ax.set(xlabel = 'x',ylabel = 'y',zlabel = 'z')
azi = 45
ax.scatter3D(y[0,:,:,azi], y[1,:,:,azi], y[2,:,:,azi],c = 'red',linewidths = 1)
ax.quiver(y[0,:,:,azi], y[1,:,:,azi], y[2,:,:,azi],surfNorms[0,:,:,azi]*scale, surfNorms[1,:,:,azi]*scale, surfNorms[2,:,:,azi]*scale)
ax.set_xlim([-geomParams['R']*1.25,geomParams['R']*1.25])
ax.set_ylim([-geomParams['R']*1.25,geomParams['R']*1.25])
ax.set_zlim([-geomParams['R']*1.25,geomParams['R']*1.25])