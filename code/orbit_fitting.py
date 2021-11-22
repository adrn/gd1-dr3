import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import sys
sys.path.append('../code/')
import warnings
warnings.filterwarnings('ignore')

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic

import astropy.coordinates as coord
import astropy.units as u
from pyia import GaiaData

import arviz as az
import emcee
import scipy
from astropy.io import fits

rnd = np.random.RandomState(seed=42)

# basic code that will be optimized
df = ms.FardalStreamDF()
phi1_prog, phi2_prog = -13, -0.2

after = GaiaData('../data/member_prob_all.fits')
model_output = after[after.post_member_prob > 0.3]
fn = '../data/sample_outputs/trace0.netcdf'
d = az.from_netcdf(fn)
ln_std_pm = np.apply_over_axes(np.mean, a=d.posterior.ln_std_pm_stream, axes=[0,1]).reshape(2)
var_pm = np.exp(ln_std_pm)**2
#HACK: choose a constant width of 0.25 for the stream (for now)
var_phi2 = 0.25**2
full_cov = model_output.pm_cov + var_pm

rv_bonaca_data = fits.open('../data/rv_catalog.fits')[1].data
gd1_rv_bonaca = rv_bonaca_data[rv_bonaca_data.pmmem & rv_bonaca_data.cmdmem & rv_bonaca_data.vrmem & rv_bonaca_data.fehmem]
var_rv = gd1_rv_bonaca.std_Vrad**2


def lnprior(vals):
    pm1, pm2, rv = vals
    
    if -12.5<pm1<-9 and -3.5<pm2<-1.5 and -400<rv<0:
        return 0.0
    return -np.inf

    
def loglik(vals):
    pm1, pm2, rv = vals
    
    gd1_stream = gc.GD1Koposov10(phi1 = phi1_prog*u.degree, phi2 = phi2_prog*u.degree, distance=8.7*u.kpc,
                          pm_phi1_cosphi2=pm1*u.mas/u.yr,
                          pm_phi2=pm2*u.mas/u.yr,
                         radial_velocity = rv*u.km/u.s)
    rep = gd1_stream.transform_to(coord.Galactocentric).data
    gd1_w0 = gd.PhaseSpacePosition(rep)
    gd1_mass = 2e4 * u.Msun
    gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
    mw = gp.MilkyWayPotential()
    gen_gd1 = ms.MockStreamGenerator(df, mw, progenitor_potential=gd1_pot)
    gd1_stream, _ = gen_gd1.run(gd1_w0, gd1_mass,
                                    dt=-1 * u.Myr, n_steps=3000)
    gd1 = gd1_stream.to_coord_frame(gc.GD1)
    gd1 = gd1[gd1.phi1.argsort()]
    
    spline_pm1 = UnivariateSpline(gd1.phi1, gd1.pm_phi1_cosphi2)
    spline_pm2 = UnivariateSpline(gd1.phi1, gd1.pm_phi2)
    spline_phi2 = UnivariateSpline(gd1.phi1, gd1.phi2)
    spline_rv = UnivariateSpline(gd1.phi1, gd1.radial_velocity)
    spline_dist = UnivariateSpline(gd1.phi1, gd1.distance)
    
    diff_pm1 = model_output.pm1 - spline_pm1(model_output.phi1)
    diff_pm2 = model_output.pm2 - spline_pm2(model_output.phi1)
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
    ll_pm = np.sum(-0.5 * (quad + np.log(det) + 2 * np.log(2*np.pi)))
    
    
    diff_phi2 = model_output.phi2 - spline_phi2(model_output.phi1)
    #need the phi2 uncertainties from the 
    ll_phi2 = np.sum(-0.5 * (np.log(var_phi2) + ((diff_phi2**2)/var_phi2) + np.log(2*np.pi)))
    
    diff_rv = gd1_rv_bonaca.Vrad - spline_rv(gd1_rv_bonaca.phi1)
    ll_rv = np.sum(-0.5 * (np.log(var_rv) + ((diff_rv**2)/var_rv) + np.log(2*np.pi)))
    ll = ll_pm + ll_phi2 + ll_rv
    
    return ll
# add phi2 with dispersion, get rvs from ana, debug by printing intermediately


def logprob(vals):
    lp = lnprior(vals)
    if not np.isfinite(lp):
        return -np.inf 
    return lp + loglik(vals)

def min_logprob(vals):
    return -logprob(vals)

if __name__ == '__main__':
    res = scipy.optimize.minimize(min_logprob, x0=np.array([-10.7, -2.5, -185]), 
                                  method='Nelder-Mead',
                                  #method='L-BFGS-B',
                                  bounds = ((-11.5, -9.5), (-3.5, -1.5), (-300, -100)),
                                  options={'disp':True, 'maxiter':100, 'iprint':99})
    print(res)

'''
nsteps = 1500
ndim, nwalkers = 3, 8

def create_IC():
    
    ret_array = np.zeros(ndim)
    
    ret_array[0] = np.random.normal(-10.8, 0.5)
    ret_array[1] = np.random.normal(-2.5, 0.5)
    ret_array[2] = np.random.normal(-185, 30)
    ret_array[3] = np.random.normal(8.7, 0.5)
    #ret_array[4] = np.random.uniform(3.5, 4.5)
    
    return ret_array

def create_IC_walkers():
    return [create_IC() for i in range(nwalkers)]

if __name__ == '__main__':
    ICs = create_IC_walkers()
    
    outfile = '../data/orbit_output/samples3.h5'
    
    backend = emcee.backends.HDFBackend(outfile)
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=logprob, backend=backend)
    pos, prob, state = sampler.run_mcmc(ICs, nsteps, progress=True)
    
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
'''