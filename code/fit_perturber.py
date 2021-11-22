from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import sys
sys.path.append('../code/')
import orbit_fitting

import os
import glob

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic
from gala.dynamics.nbody import DirectNBody

import astropy.coordinates as coord
import astropy.units as u
from pyia import GaiaData

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt

import emcee
import tqdm
from contextlib import closing
from multiprocessing import Pool
import scipy
from sklearn.neighbors import KernelDensity

# my top choice and the one to use for the spur modeling
#creates the gaps around -20 and -5
df = ms.FardalStreamDF(random_state=np.random.RandomState(42))
gd1_init = gc.GD1Koposov10(phi1 = -13*u.degree, phi2=-0.2*u.degree, distance=8.73*u.kpc,
                      pm_phi1_cosphi2=-10.6*u.mas/u.yr,
                      pm_phi2=-2.52*u.mas/u.yr,
                     radial_velocity = -185*u.km/u.s)
rep = gd1_init.transform_to(coord.Galactocentric).data
gd1_w0 = gd.PhaseSpacePosition(rep)
gd1_mass = 1e4 * u.Msun
gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
mw = gp.MilkyWayPotential()
gen_gd1 = ms.MockStreamGenerator(df, mw, progenitor_potential=gd1_pot)
gd1_stream, gd1_nbody = gen_gd1.run(gd1_w0, gd1_mass,
                                dt=-1 * u.Myr, n_steps=3000)
gd1 = gd1_stream.to_coord_frame(gc.GD1)

gd1_short = gd1_stream[(-50<gd1.phi1.value) & (gd1.phi1.value<-20)]
w0_now = gd.PhaseSpacePosition(gd1_short.data, gd1_short.vel)

#to help with initialization
orbit = mw.integrate_orbit(w0_now, dt=-1*u.Myr, n_steps=500)
old_gd1 = orbit[-1]

after = GaiaData('../data/member_prob_all.fits')
model_output = after[after.post_member_prob > 0.3]

def make_gifs(t_int, orbits, core):
    for i in range(t_int+30):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
        if i<(30*2):
            orbits[i, 0].plot(c='r', s=200, axes=[ax1, ax2, ax3])
        orbits[i, 1:].plot(alpha=0.3, c='b', s=5, axes=[ax1, ax2, ax3])
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)
        plt.savefig('../image_folders/xyz_core{}/image_{}'.format(str(core), str(i)), dpi=100)
        fig, ax4 = plt.subplots(1, 1, figsize=(10,3))
        back0 = orbits[i, 1:].to_coord_frame(gc.GD1)
        ax4.set_ylim(np.min(back0.phi2.value)-2, np.max(back0.phi2.value)+2)
        ax4.scatter(back0.phi1, back0.phi2, s = 1, c='b', alpha=0.3)
        plt.savefig('../image_folders/gd1_coord_core{}/image_{}'.format(str(core), str(i)), dpi=100)

    # Create the frames
    frames = []
    imgs = sorted(glob.glob('../image_folders/xyz_core{}/*.png'.format(str(core))), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save('../image_gifs/xyz_core{}.gif'.format(str(core)), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=40, loop=0)

    # Create the frames
    frames1 = []
    imgs1 = sorted(glob.glob('../image_folders/gd1_coord_core{}/*.png'.format(str(core))), key=os.path.getmtime)
    for i in imgs1:
        new_frame = Image.open(i)
        frames1.append(new_frame)

    frames1[0].save('../image_gifs/gd1_coord_core{}.gif'.format(str(core)), format='GIF',
                   append_images=frames1[1:],
                   save_all=True,
                   duration=40, loop=0)

def lnprior(vals):
    xpert, ypert, zpert, vxpert, vypert, vzpert, logmasspert, t_int = vals
    
    orbit = mw.integrate_orbit(w0_now, dt=-1*u.Myr, n_steps=int(t_int))
    old_gd1 = orbit[-1]
    
    minx, maxx = np.min(old_gd1.pos.x.value), np.max(old_gd1.pos.x.value)
    miny, maxy = np.min(old_gd1.pos.y.value), np.max(old_gd1.pos.y.value)
    minz, maxz = np.min(old_gd1.pos.z.value), np.max(old_gd1.pos.z.value)
    
    
    if minx<xpert<maxx and miny<ypert<maxy and minz<zpert<maxz and -500<vxpert<500 and -500<vypert<500 and -500<vzpert<500 and 4<logmasspert<9 and 0<t_int<1200:
        return 0.0
    return -np.inf

def loglik(vals, core):
    xpert, ypert, zpert, vxpert, vypert, vzpert, logmasspert, t_int = vals
    
    orbit = mw.integrate_orbit(w0_now, dt=-1*u.Myr, n_steps=int(t_int))
    old_gd1 = orbit[-1]
     
    w0_old_stream = gd.PhaseSpacePosition(pos=old_gd1.pos, 
                                      vel=old_gd1.vel)
    w0_pert = gd.PhaseSpacePosition(pos=[xpert, ypert, zpert] * u.kpc,
                                    vel=[vxpert, vypert, vzpert] * u.km/u.s)
    orbit_stream = mw.integrate_orbit(w0_old_stream, dt=-1*u.Myr, n_steps=30)
    orbit_pert   = mw.integrate_orbit(w0_pert, dt=-1*u.Myr, n_steps=30)
    orig_stream = orbit_stream[-1]
    orig_pert   = orbit_pert[-1]
    
    w0_orig_stream = gd.PhaseSpacePosition(pos=orig_stream.pos, 
                                           vel=orig_stream.vel)

    # perturber parameters at the moment of "impact" lets call this when it is "most embedded" in the stream
    x_pert,y_pert,z_pert= orig_pert.pos.x, orig_pert.pos.y, orig_pert.pos.z
    vx_pert,vy_pert,vz_pert = orig_pert.vel.d_x, orig_pert.vel.d_y, orig_pert.vel.d_z
    vx_pert,vy_pert,vz_pert = vx_pert.to(u.km/u.s).value, vy_pert.to(u.km/u.s).value, vz_pert.to(u.km/u.s).value
    logmass_pert = logmasspert

    w0_orig_pert = gd.PhaseSpacePosition(pos=[x_pert, y_pert, z_pert] * u.kpc, 
                                     vel=[vx_pert, vy_pert, vz_pert] * u.km/u.s)
    perturber_pot = gp.HernquistPotential(m=10**logmass_pert*u.Msun, c=core*u.pc, units=galactic)

    # all potentials of the orbit (for nbody simulation)
    w0 = gd.combine((w0_orig_pert, w0_orig_stream))
    particle_pot = [list([perturber_pot]) + [None] * orig_stream.shape[0]][0]

    nbody = DirectNBody(w0, particle_pot, external_potential=mw)
    total_time = int(t_int) + 30
    orbits = nbody.integrate_orbit(dt=1*u.Myr, t1=0*u.Gyr, t2=total_time*u.Myr)
    
    # what should be compared to present time
    back = orbits[-1, 1:].to_coord_frame(gc.GD1)
    
    model_window = back[(back.phi1.value > -50) & (back.phi1.value < -20)]
    #cut output of model to window, kernel density estimate (phi1 vs phi2, pm) of the model, evaluate the data 
    #relative to the model
    #kde for pm has to give scales to units and dimensions...
    #Bandwidth is completely arbitrary for now but meant to be approx the width of stream and the error on pm
    data = model_output[(model_output.phi1[:,0] > -50) & (model_output.phi1[:,0] < -20)]
    kde_phi2 = KernelDensity(kernel='gaussian', 
                             bandwidth=0.1).fit(np.array([model_window.phi1, model_window.phi2]).T)
    loglike_phi2 = kde_phi2.score(np.array([data.phi1.flatten(), data.phi2.flatten()]).T)
    
    kde_pm1 = KernelDensity(kernel='gaussian',
                            bandwidth=0.05).fit(np.array([model_window.phi1,model_window.pm_phi1_cosphi2]).T)
    loglike_pm1 = kde_pm1.score(np.array([data.phi1.flatten(), data.pm1.flatten()]).T)
    
    kde_pm2 = KernelDensity(kernel='gaussian', 
                            bandwidth=0.05).fit(np.array([model_window.phi1, model_window.pm_phi2]).T)
    loglike_pm2 = kde_pm2.score(np.array([data.phi1.flatten(), data.pm2.flatten()]).T)
    ll = loglike_phi2 + loglike_pm1 + loglike_pm2
    return ll, kde_phi2, kde_pm1, kde_pm2

def logprob(vals, core):
    lp = lnprior(vals)
    if not np.isfinite(lp):
        return -np.inf 
    ll, kde_phi2, kde_pm1, kde_pm2 = loglik(vals, core)
    return lp + ll


def min_logprob_spur(vals, core):
    return -logprob(vals, core)


if __name__ == '__main__':
    for core in np.array([0.1, 1, 10, 100]):
        os.mkdir('../image_folders/xyz_core{}'.format(core))
        os.mkdir('../image_folders/gd1_coord_core{}'.format(core))
        res = scipy.optimize.minimize(min_logprob_spur, 
                                      x0=np.array([np.mean(old_gd1.pos.x.value),
                                                   np.mean(old_gd1.pos.y.value),
                                                   np.mean(old_gd1.pos.z.value),
                                                   50, -50, 50,
                                                   6.5,
                                                   500]), 
                                      args=(core),
                                      method='Nelder-Mead',
                                      options={'disp':True,'maxiter':500, 'adaptive':True})
        print(res)
        
        # make images and a movie of the best fit
        xpert, ypert, zpert, vxpert, vypert, vzpert, logmasspert, t_int = res.x
        
        orbit = mw.integrate_orbit(w0_now, dt=-1*u.Myr, n_steps=int(t_int))
        old_gd1 = orbit[-1]

        w0_old_stream = gd.PhaseSpacePosition(pos=old_gd1.pos, 
                                          vel=old_gd1.vel)
        w0_pert = gd.PhaseSpacePosition(pos=[xpert, ypert, zpert] * u.kpc,
                                        vel=[vxpert, vypert, vzpert] * u.km/u.s)
        orbit_stream = mw.integrate_orbit(w0_old_stream, dt=-1*u.Myr, n_steps=30)
        orbit_pert   = mw.integrate_orbit(w0_pert, dt=-1*u.Myr, n_steps=30)
        orig_stream = orbit_stream[-1]
        orig_pert   = orbit_pert[-1]

        w0_orig_stream = gd.PhaseSpacePosition(pos=orig_stream.pos, 
                                               vel=orig_stream.vel)

        # perturber parameters at the moment of "impact" lets call this when it is "most embedded" in the stream
        x_pert,y_pert,z_pert= orig_pert.pos.x, orig_pert.pos.y, orig_pert.pos.z
        vx_pert,vy_pert,vz_pert = orig_pert.vel.d_x, orig_pert.vel.d_y, orig_pert.vel.d_z
        vx_pert,vy_pert,vz_pert = vx_pert.to(u.km/u.s).value, vy_pert.to(u.km/u.s).value, vz_pert.to(u.km/u.s).value
        logmass_pert = logmasspert

        w0_orig_pert = gd.PhaseSpacePosition(pos=[x_pert, y_pert, z_pert] * u.kpc, 
                                         vel=[vx_pert, vy_pert, vz_pert] * u.km/u.s)
        perturber_pot = gp.HernquistPotential(m=10**logmass_pert*u.Msun, c=core*u.pc, units=galactic)

        # all potentials of the orbit (for nbody simulation)
        w0 = gd.combine((w0_orig_pert, w0_orig_stream))
        particle_pot = [list([perturber_pot]) + [None] * orig_stream.shape[0]][0]

        nbody = DirectNBody(w0, particle_pot, external_potential=mw)
        total_time = int(t_int) + 30
        orbits = nbody.integrate_orbit(dt=1*u.Myr, t1=0*u.Gyr, t2=total_time*u.Myr)
        
        make_gifs(int(t_int), orbits, core)