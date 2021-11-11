import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import sys
sys.path.append('../code/')

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic

import astropy.coordinates as coord
import astropy.units as u
from pyia import GaiaData

import emcee

rnd = np.random.RandomState(seed=42)

# basic code that will be optimized
df = ms.FardalStreamDF()
phi1_prog, phi2_prog = -13, -0.2

after = GaiaData('../data/member_prob_3_all.fits')
pm_model_output = after[after.post_member_prob_pm > 0.5]
full_cov = pm_model_output.pm_cov

def lnprior(vals):
    pm1 = vals[0]
    pm2 = vals[1]
    rv = vals[2]
    dist = vals[3]
    mass = vals[4]
    
    lnp = 0
    
    if pm1 < -15 or pm1 > -7:
        return -np.inf
    elif pm2 < -5 or pm2 > 0:
        return -np.inf
    elif rv < -500 or rv > 300:
        return -np.inf
    elif dist < 4 or dist > 13:
        return -np.inf
    elif mass < 1e3 or mass > 1e5
        return -np.inf
    else:
        return lnp
    
def loglik(vals):
    pm1 = vals[0]
    pm2 = vals[1]
    rv = vals[2]
    dist = vals[3]
    mass = vals[4]
    
    gd1_stream = gc.GD1Koposov10(phi1 = phi1_prog*u.degree, phi2 = phi2_prog*u.degree, distance=dist*u.kpc,
                          pm_phi1_cosphi2=pm1*u.mas/u.yr,
                          pm_phi2=pm2*u.mas/u.yr,
                         radial_velocity = rv*u.km/u.s)
    rep = gd1_stream.transform_to(coord.Galactocentric).data
    gd1_w0 = gd.PhaseSpacePosition(rep)
    gd1_mass = mass * u.Msun
    gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
    mw = gp.MilkyWayPotential()
    gen_gd1 = ms.MockStreamGenerator(df, mw, progenitor_potential=gd1_pot)
    gd1_stream, _ = gen_gd1.run(gd1_w0, gd1_mass,
                                    dt=-1 * u.Myr, n_steps=2500)
    gd1 = gd1_stream.to_coord_frame(gc.GD1)
    
    spline_pm1 = UnivariateSpline(gd1.phi1, gd1.pm_phi1_cosphi2)
    spline_pm2 = UnivariateSpline(gd1.phi1, gd1.pm_phi2)
    spline_rv = UnivariateSpline(gd1.phi1, gd1.radial_velocity)
    spline_dist = UnivariateSpline(gd1.phi1, gd1.distance)
    
    diff_pm1 = pm_model_output.pm1 - spline_pm1(pm_model_ouput.phi1)
    diff_pm2 = pm_model_output.pm2 - spline_pm2(pm_model_ouput.phi1)
    #diff_rv = pm_model_output.pm1 - spline_rv(pm_model_ouput.phi1)
    #diff_dist = pm_model_output.pm1 - spline_dist(pm_model_ouput.phi1)
    diff_pm = np.vstack([diff_pm1, diff_pm2]).T
    
    a = full_cov[:, 0, 0]
    b = c = full_cov[:, 0, 1]
    d = full_cov[:, 1, 1]
    det = a * d - b * c
    
    numer = (
            d * diff_pm[:, 0] ** 2 
            + a * diff_pm[:, 1] ** 2
            - (b + c) * diff_pm[:, 0] * diff_pm[:, 1]
        )
    
    quad = numer / det
    ll = -0.5 * (quad + tt.log(det) + 2 * tt.log(2*np.pi))
    return ll
    
nsteps = 2000
ndim, nwalkers = 5, 100

def create_IC():
    
    init_pm1 = np.random.normal(-10.8, 1.5)
    init_pm2 = np.random.normal(-2.5, 1)
    init_rv = np.random.normal(-185, 50)
    init_dist = np.random.normal(8.7, 1)
    init_mass = np.random.uniform(5e3, 5e4)
    
    IC = [init_pm1, init_pm2, init_rv, init_dist, init_mass]
    
    return IC

def create_IC_walkers():
    return [create_IC() for i in range(nwalkers)]

if __name__ == '__main__':
    ICs = rand_array_walkers()
    
    outfile = '../data/orbit_samples/samples0.h5'
    
    backend = emcee.backends.HDFBackend(outfile)
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=loglik, backend=backend)
    pos, prob, state = sampler.run_mcmc(ICs, 500)
    sampler.reset()
    
    samples = sampler.chain[:, :, :].reshape((-1, ndim))


