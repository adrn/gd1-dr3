import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
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

class OrbitFit:

    def __init__(self):
        # basic code that will be optimized
        self.df = ms.FardalStreamDF()
        self.phi1_prog = -13

        after = GaiaData('../data/member_prob_all.fits')
        self.model_output = after[(after.post_member_prob > 0.3) &
                             (-60 < after.phi1.flatten()) & (after.phi1.flatten() < -20) & #just the region of the spur
                             (after.phi2.flatten() < 0.5)] # don't include the spur in the fit
        fn = '../data/sample_outputs/trace0.netcdf'
        d = az.from_netcdf(fn)
        ln_std_pm = np.apply_over_axes(np.mean, a=d.posterior.ln_std_pm_stream, axes=[0,1]).reshape(2)
        var_pm = np.exp(ln_std_pm)**2
        #HACK: choose a constant width of 0.25 for the stream (for now)
        self.var_phi2 = 0.25**2
        self.full_cov = self.model_output.pm_cov + var_pm

        rv_bonaca_data = fits.open('../data/rv_catalog.fits')[1].data
        gd1_rv_bonaca = rv_bonaca_data[rv_bonaca_data.pmmem & rv_bonaca_data.cmdmem & rv_bonaca_data.vrmem & rv_bonaca_data.fehmem]
        self.var_rv = gd1_rv_bonaca.std_Vrad**2


        # Given a phi1 value, get the other values of the progenitor from the proper motion model
        from_pm = fits.open('../data/pm_model_output.fits')[1].data
        self.spline_phi2_means = InterpolatedUnivariateSpline(from_pm['phi1'][::5], from_pm['phi2_means'][::5])
        self.spline_width_means = InterpolatedUnivariateSpline(from_pm['phi1'][::5], from_pm['width_means'][::5])

        self.spline_pm1_means = InterpolatedUnivariateSpline(from_pm['phi1'][::5], from_pm['pm1_means'][::5])
        self.spline_pm1_std = InterpolatedUnivariateSpline(from_pm['phi1'][::5], from_pm['pm1_std'][::5])

        self.spline_pm2_means = InterpolatedUnivariateSpline(from_pm['phi1'][::5], from_pm['pm2_means'][::5])
        self.spline_pm2_std = InterpolatedUnivariateSpline(from_pm['phi1'][::5], from_pm['pm2_std'][::5])

        sections = np.arange(-100,15,5)
        dm = np.concatenate([[14.7, 14.6, 14.5, 14.45, 14.4, 14.35, 14.3, 14.3],
                             np.linspace(14.3, 14.6, 9),
                             [14.71, 14.75, 14.8, 15, 15.2, 15.4]])
        self.spline_dm_true = UnivariateSpline(sections, dm, k=5)

        rv_bonaca_data = fits.open('../data/rv_catalog.fits')[1].data
        gd1_rv_bonaca = rv_bonaca_data[rv_bonaca_data.pmmem & rv_bonaca_data.cmdmem &
                                       rv_bonaca_data.vrmem & rv_bonaca_data.fehmem]
        self.gd1_rv_bonaca = gd1_rv_bonaca[gd1_rv_bonaca.phi1.argsort()]

        self.spline_rv_bon = UnivariateSpline(self.gd1_rv_bonaca.phi1, self.gd1_rv_bonaca.Vrad, k=1, s = np.inf)


    def lnprior(self, vals):
        lnp = 0

        phi2_prog, pm1_prog, pm2_prog, rv_prog, dist_prog, halo_m = vals

        dm_pm_model = self.spline_dm_true(self.phi1_prog)
        dist_pm_model = 10**((dm_pm_model + 5) / 5) / 1000

        lnp += -((phi2_prog - self.spline_phi2_means(self.phi1_prog)) / self.spline_width_means(self.phi1_prog))**2. / 2.
        lnp += -((pm1_prog - self.spline_pm1_means(self.phi1_prog)) / self.spline_pm1_std(self.phi1_prog))**2. / 2.
        lnp += -((pm2_prog - self.spline_pm2_means(self.phi1_prog)) / self.spline_pm2_std(self.phi1_prog))**2. / 2.
        lnp += -((rv_prog - self.spline_rv_bon(self.phi1_prog)) / 10)**2. / 2.
        lnp += -((dist_prog - dist_pm_model) / 0.2)**2. / 2.

        return lnp


    def loglik(self, vals):
        phi2_prog, pm1_prog, pm2_prog, rv_prog, dist_prog, halo_mass = vals
        mhalo = halo_mass*10**11
        print(f'{phi2_prog:f}',f'{pm1_prog:.3f}',f'{pm2_prog:.3f}',f'{rv_prog:.1f}',f'{dist_prog:.3f}',f'{halo_mass:.2f}')

        gd1_stream = gc.GD1Koposov10(phi1 = self.phi1_prog*u.degree, phi2 = phi2_prog*u.degree, distance=dist_prog*u.kpc,
                              pm_phi1_cosphi2=pm1_prog*u.mas/u.yr,
                              pm_phi2=pm2_prog*u.mas/u.yr,
                             radial_velocity = rv_prog*u.km/u.s)
        rep = gd1_stream.transform_to(coord.Galactocentric).data
        gd1_w0 = gd.PhaseSpacePosition(rep)
        gd1_mass = 5e3 * u.Msun
        gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
        mw = gp.MilkyWayPotential(halo={'m': mhalo*u.Msun, 'r_s': 15.62*u.kpc})
        gen_gd1 = ms.MockStreamGenerator(self.df, mw, progenitor_potential=gd1_pot)
        gd1_stream, _ = gen_gd1.run(gd1_w0, gd1_mass,
                                        dt=-1 * u.Myr, n_steps=3000)
        gd1 = gd1_stream.to_coord_frame(gc.GD1)
        gd1 = gd1[gd1.phi1.argsort()]

        spline_pm1 = UnivariateSpline(gd1.phi1, gd1.pm_phi1_cosphi2)
        spline_pm2 = UnivariateSpline(gd1.phi1, gd1.pm_phi2)
        spline_phi2 = UnivariateSpline(gd1.phi1, gd1.phi2)
        spline_rv = UnivariateSpline(gd1.phi1, gd1.radial_velocity)
        spline_dist = UnivariateSpline(gd1.phi1, gd1.distance)

        diff_pm1 = self.model_output.pm1 - spline_pm1(self.model_output.phi1)
        diff_pm2 = self.model_output.pm2 - spline_pm2(self.model_output.phi1)
        diff_pm = np.vstack([diff_pm1, diff_pm2]).T

        a = self.full_cov[:, 0, 0]
        b = c = self.full_cov[:, 0, 1]
        d = self.full_cov[:, 1, 1]
        det = a * d - b * c

        numer = (
                d * diff_pm[:, 0] ** 2
                + a * diff_pm[:, 1] ** 2
                - (b + c) * diff_pm[:, 0] * diff_pm[:, 1]
            )

        quad = numer / det
        ll_pm = np.sum(-0.5 * (quad + np.log(det) + 2 * np.log(2*np.pi)))


        diff_phi2 = self.model_output.phi2 - spline_phi2(self.model_output.phi1)
        #need the phi2 uncertainties from the
        ll_phi2 = np.sum(-0.5 * (np.log(self.var_phi2) + ((diff_phi2**2)/self.var_phi2) + np.log(2*np.pi)))

        diff_rv = self.gd1_rv_bonaca.Vrad - spline_rv(self.gd1_rv_bonaca.phi1)
        ll_rv = np.sum(-0.5 * (np.log(self.var_rv) + ((diff_rv**2)/self.var_rv) + np.log(2*np.pi)))
        ll = ll_pm + ll_phi2 + ll_rv

        return ll
    # add phi2 with dispersion, get rvs from ana, debug by printing intermediately


    def logprob(self, vals):
        lp = self.lnprior(vals)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglik(vals)

    def min_logprob(self, vals):
        return -self.logprob(vals)

if __name__ == '__main__':
    orbitfit = OrbitFit()
    res = scipy.optimize.minimize(orbitfit.min_logprob,x0=np.array([0, -10, -2.3, -180, 8.8, 5.4]),
                                  method='Nelder-Mead',
                                  #method='L-BFGS-B',
                                  options={'disp':True})
    print('From -60 to -20')
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
