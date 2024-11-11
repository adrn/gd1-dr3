# When I have a bit of time I should write this more efficiently (have class take Fitpert class instead of rewriting)
import sys
sys.path.append('../code/')
import fit_perturber as fp

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pyia import GaiaData
import gala.coordinates as gc
import gala.potential as gp
from gala.potential import NullPotential
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic
from Nbody_gala import DirectNBody

import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
import scipy
from sklearn.neighbors import KernelDensity


class PerturbOpt:

    def __init__(self):
        #########################
        ## DATA FOR COMPARISON ##
        #########################
        after = GaiaData('../data/member_prob_all.fits')
        model_output = after[after.post_member_prob > 0.25]
        self.data = model_output[(model_output.phi1[:,0] > -65) & (model_output.phi1[:,0] < -22)]


        ##########################################
        ## CURRENT STREAM WITHOUT THE PERTURBER ##
        ##########################################
        df = ms.FardalStreamDF(random_state=np.random.RandomState(42))
        gd1_init = gc.GD1Koposov10(phi1 = -13*u.degree, phi2=0*u.degree, distance=8.836*u.kpc,
                                  pm_phi1_cosphi2=-10.575*u.mas/u.yr,
                                  pm_phi2=-2.439*u.mas/u.yr,
                                  radial_velocity = -189.6*u.km/u.s)
        rep = gd1_init.transform_to(coord.Galactocentric).data
        gd1_w0 = gd.PhaseSpacePosition(rep)
        gd1_mass = 5e3 * u.Msun
        gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
        self.mw = gp.MilkyWayPotential(halo={'m': 5.43e11*u.Msun, 'r_s': 15.78*u.kpc})
        gen_gd1 = ms.MockStreamGenerator(df, self.mw, progenitor_potential=gd1_pot)
        gd1_stream, gd1_nbody = gen_gd1.run(gd1_w0, gd1_mass,
                                        dt=-1 * u.Myr, n_steps=3000)
        gd1 = gd1_stream.to_coord_frame(gc.GD1)

        gd1_short = gd1_stream[(-65<gd1.phi1.value) & (gd1.phi1.value<-22)]
        self.w0_now = gd.PhaseSpacePosition(gd1_short.data, gd1_short.vel)

        rv_bonaca_data = fits.open('../data/rv_catalog.fits')[1].data
        self.gd1_rv_bonaca = rv_bonaca_data[rv_bonaca_data.pmmem & rv_bonaca_data.cmdmem & rv_bonaca_data.vrmem & rv_bonaca_data.fehmem]


    def pre_fitting(self, vals):

        self.b, self.psi, self.z, self.v_z, self.vpsi, self.t_int, self.logm = vals

        self.core = 1.05 * (10**self.logm / (10**8))**0.5

        ##############################################
        ## STREAM PROPERTIES AT TIME OF INTERACTION ##
        ##############################################
        orbit = self.mw.integrate_orbit(self.w0_now, dt=-1*u.Myr, n_steps=int(self.t_int))
        old_gd1 = orbit[-1]

        # Converting from xyz to relative-to-stream coordinates and back again
        #take the velocities of the stream particles where the pertuber will cross the stream
        center = old_gd1[(np.abs(np.mean(old_gd1.pos.x.value) - old_gd1.pos.x.value) < 0.5) &
                     (np.abs(np.mean(old_gd1.pos.y.value) - old_gd1.pos.y.value) < 0.5) &
                     (np.abs(np.mean(old_gd1.pos.z.value) - old_gd1.pos.z.value) < 0.5)]

        vxstream = np.mean(center.vel.d_x).to(u.km/u.s).value
        vystream = np.mean(center.vel.d_y).to(u.km/u.s).value
        vzstream = np.mean(center.vel.d_z).to(u.km/u.s).value

        # make the location of impact the mean value for now. This has been chosen so that a feature will
        #  appear around phi1 = -40 so it's probably pretty good
        pos_pert = np.mean(old_gd1.pos)

        self.site_at_impact_w0 = gd.PhaseSpacePosition(pos=np.mean(old_gd1.pos), vel=[vxstream, vystream, vzstream]*u.km/u.s)

        ##########################################
        ## STREAM PROPERTIES BEFORE INTERACTION ##
        ##########################################
        w0_old_stream = gd.PhaseSpacePosition(pos=old_gd1.pos,
                                              vel=old_gd1.vel)
        orbit_stream = self.mw.integrate_orbit(w0_old_stream, dt=-1*u.Myr, n_steps=30)
        self.orig_stream = orbit_stream[-1]
        self.w0_orig_stream = gd.PhaseSpacePosition(pos=self.orig_stream.pos,
                                                    vel=self.orig_stream.vel)

        self.perturber_pot = gp.HernquistPotential(m=10**self.logm*u.Msun, c=self.core*u.pc, units=galactic)
        #self.perturber_pot = gp.KeplerPotential(m=10**self.logm*u.Msun, units=galactic)

        return self.site_at_impact_w0

    def get_cyl_rotation(self): #borrowed from Adrian Price-Whelan's streampunch github repo
        L = self.site_at_impact_w0.angular_momentum()
        v = self.site_at_impact_w0.v_xyz

        new_z = v / np.linalg.norm(v, axis=0)
        new_x = L / np.linalg.norm(L, axis=0)
        new_y = -np.cross(new_x, new_z)
        R = np.stack((new_x, new_y, new_z))
        return R

    def get_perturber_w0_at_impact(self):

        # Get the rotation matrix to rotate from Galactocentric to cylindrical
        # impact coordinates at the impact site along the stream
        R = self.get_cyl_rotation()

        b, psi, z, v_z, vpsi = self.b * u.pc, self.psi * u.deg, self.z * u.kpc, self.v_z * u.km/u.s, self.vpsi * u.km/u.s

        # Define the position of the perturber at the time of impact in the
        # cylindrical impact coordinates:
        perturber_pos = coord.CylindricalRepresentation(rho=b,
                                                        phi=psi,
                                                        z=z)
        # z=0 is fixed by definition: This is the impact site

        # Define the velocity in the cylindrical impact coordinates:
        perturber_vel = coord.CylindricalDifferential(
            d_rho=0*u.km/u.s,  # Fixed by definition: b is closest approach
            d_phi=(vpsi / b).to(u.rad/u.Myr, u.dimensionless_angles()),
            d_z=v_z)

        # Transform from the cylindrical impact coordinates to Galactocentric
        perturber_rep = perturber_pos.with_differentials(perturber_vel)
        perturber_rep = perturber_rep.represent_as(
            coord.CartesianRepresentation, coord.CartesianDifferential)
        perturber_rep = perturber_rep.transform(R.T)

        pos = perturber_rep.without_differentials() + self.site_at_impact_w0.pos
        vel = perturber_rep.differentials['s'] + self.site_at_impact_w0.vel

        # This should be in Galactocentric Cartesian coordinates now!
        return gd.PhaseSpacePosition(pos, vel)


    def nbody(self):
        #################################################
        ## PERTURBER PROPERTIES AT TIME OF INTERACTION ##
        #################################################
        w0_pert = self.get_perturber_w0_at_impact()
        pert_energy = gp.hamiltonian.Hamiltonian(self.mw).energy(w0_pert)

        if pert_energy.value[0] > 0:
            self.current, self.orbits = False, False
        else:
            #############################################
            ## PERTURBER PROPERTIES BEFORE INTERACTION ##
            #############################################
            orbit_pert  = self.mw.integrate_orbit(w0_pert, dt=-1*u.Myr, n_steps=30)
            orig_pert   = orbit_pert[-1]

            x_pert,y_pert,z_pert= orig_pert.pos.x, orig_pert.pos.y, orig_pert.pos.z
            vx_pert = orig_pert.vel.d_x.to(u.km/u.s).value
            vy_pert = orig_pert.vel.d_y.to(u.km/u.s).value
            vz_pert = orig_pert.vel.d_z.to(u.km/u.s).value

            w0_orig_pert = gd.PhaseSpacePosition(pos=[x_pert, y_pert, z_pert] * u.kpc,
                                             vel=[vx_pert, vy_pert, vz_pert]*u.km/u.s)

            # all potentials of the orbit (for nbody simulation)
            w0 = gd.combine((w0_orig_pert, self.w0_orig_stream))
            particle_pot = [list([self.perturber_pot]) + [NullPotential(units=self.perturber_pot.units)] * self.orig_stream.shape[0]][0]

            ##############################
            ## LAUNCH N-BODY SIMULATION ##
            ##############################
            nbody = DirectNBody(w0, particle_pot, external_potential=self.mw, save_all=True)
            total_time = int(self.t_int) + 30
            self.orbits = nbody.integrate_orbit(dt=1*u.Myr, t1=0, t2=total_time*u.Myr)

            # what should be compared to present time
            self.current = self.orbits[-1, 1:].to_coord_frame(gc.GD1)
        return self.current, self.orbits

    def lnprior(self, params):
        self.lnp = 0

        self.pre_fitting(params)

        if (self.b < 0) | (self.b > 100):
            self.lnp += -np.inf
            return self.lnp
        elif (self.psi < 0) | (self.psi > 360):
            self.lnp += -np.inf
            return self.lnp
        elif (self.z < -0.2) | (self.z > 0.6):
            self.lnp += -np.inf
            return self.lnp
        elif (self.v_z < 50) | (self.v_z > 250):
            self.lnp += -np.inf
            return self.lnp
        elif (self.vpsi < 1) | (self.vpsi > 150):
            self.lnp += -np.inf
            return self.lnp
        elif (self.t_int < 50) | (self.t_int > 1200):
            self.lnp += -np.inf
            return self.lnp
        elif (self.logm < 5.8) | (self.logm > 6.8):
            self.lnp += -np.inf
            return self.lnp

    def loglik_model_kde(self, params):

        #self.pre_fitting(params)
        current, orbits = self.nbody()
        if not current:
            self.ll = -np.inf

        else:
            #############################
            ## EVALUATE LOG-LIKELIHOOD ##
            #############################

            #cut output of model to window
            model_window = self.current[(self.current.phi1.value > -65) & (self.current.phi1.value < -22)]

            # evaluate the ll using KDE
            kde_phi2 = KernelDensity(kernel='gaussian',
                                     bandwidth=0.11).fit(np.array([(model_window.phi1.value)/10,
                                                                   model_window.phi2]).T)
            loglike_phi2 = kde_phi2.score_samples(np.array([(self.data.phi1.flatten())/10,
                                                             self.data.phi2.flatten()]).T)
            loglike_phi2 = np.sum(loglike_phi2)

            kde_pm1 = KernelDensity(kernel='gaussian',
                                    bandwidth=0.21).fit(np.array([(model_window.phi1.value)/15,
                                                                  model_window.pm_phi1_cosphi2]).T)
            loglike_pm1 = kde_pm1.score_samples(np.array([(self.data.phi1.flatten())/15,
                                                           self.data.pm1.flatten()]).T)
            loglike_pm1 = np.sum(loglike_pm1)

            kde_pm2 = KernelDensity(kernel='gaussian',
                                    bandwidth=0.285).fit(np.array([(model_window.phi1.value)/15,
                                                                  model_window.pm_phi2]).T)
            loglike_pm2 = kde_pm2.score_samples(np.array([(self.data.phi1.flatten())/15,
                                                           self.data.pm2.flatten()]).T)
            loglike_pm2 = np.sum(loglike_pm2)

            kde_rv = KernelDensity(kernel='gaussian',
                                   bandwidth=0.82).fit(np.array([model_window.phi1.value * 5,
                                                                model_window.radial_velocity]).T)
            loglike_rv = kde_rv.score_samples(np.array([self.gd1_rv_bonaca.phi1 * 5,
                                                        self.gd1_rv_bonaca.Vrad]).T)
            loglike_rv = np.sum(loglike_rv)

            self.ll = loglike_phi2 + loglike_pm1 + loglike_pm2 + loglike_rv
        print(params, self.ll)

        return self.ll

    def logprob(self, params):
        self.lnprior(params)
        if not np.isfinite(self.lnp):
            return -np.inf
        return self.lnp + self.loglik_model_kde(params)

    def min_logprob(self, params):
        return -self.logprob(params)


if __name__ == '__main__':
    params = [1, 0, 0.2, 170, 10, 820, 6.2]
    PertOpt = PerturbOpt()
    #PertOpt.loglik_model_kde(params)
    res = scipy.optimize.minimize(PertOpt.min_logprob,x0=np.array(params),
                                  method='Nelder-Mead',
                                  #method='L-BFGS-B',
                                  options={'disp':True})
