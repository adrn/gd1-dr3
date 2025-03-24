import warnings
warnings.filterwarnings("ignore")
import numpy as np

import gala.coordinates as gc
import gala.potential as gp
from gala.potential import NullPotential
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic
#from gala.dynamics.nbody import DirectNBody
from Nbody_gala import DirectNBody
#from gravhopper import Simulation, IC

import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from pyia import GaiaData

from sklearn.neighbors import KernelDensity

class FitPert:

    def __init__(self, data,data_short, mw, w0_now):

        self.data = data
        self.data_short = data_short
        self.mw = mw
        self.w0_now = w0_now

        rv_bonaca_data = fits.open('../data/rv_catalog.fits')[1].data
        self.gd1_rv_bonaca = rv_bonaca_data[rv_bonaca_data.pmmem & rv_bonaca_data.cmdmem & rv_bonaca_data.vrmem & rv_bonaca_data.fehmem]
        self.gd1_rv_bonaca_short = self.gd1_rv_bonaca[(self.gd1_rv_bonaca.phi1 > -43) & (self.gd1_rv_bonaca.phi1 < -25)]

    def pre_fitting(self, vals):

        self.b, self.psi, self.z, self.v_z, self.vpsi, self.t_int, self.logm, self.logcore = vals

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

        self.perturber_pot = gp.HernquistPotential(m=10**self.logm*u.Msun, c=(10**self.logcore)*u.pc, units=galactic)
        #perturber_pot = gp.KeplerPotential(m=10**logm*u.Msun, units=galactic)

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

        ## CHECK THAT THE PERTURBER IS BOUND TO THE MILKY WAY
        pert_energy = gp.hamiltonian.Hamiltonian(self.mw).energy(w0_pert)
        if pert_energy.value[0] > 0:
            self.current = False
            self.orbits = False
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

    def loglik(self, params):
        self.pre_fitting(params)
        current, orbits = self.nbody()

        if not current:
            ll_model, ll_model_short, ll_phi2_short, ll_data, model_dens_ratio = np.nan, np.nan, np.nan, np.nan, np.nan
            pert_apo, pert_peri = np.nan, np.nan
        else:
            #############################
            ## EVALUATE LOG-LIKELIHOOD ##
            #############################

            #cut output of model to window
            model_window = self.current[(self.current.phi1.value > -65) & (self.current.phi1.value < -22)]

            # evaluate the ll using KDE
            kde_phi2_model = KernelDensity(kernel='gaussian',
                                     bandwidth=0.11).fit(np.array([(model_window.phi1.value + 42)/10,
                                                                   model_window.phi2]).T)
            loglike_phi2_model = kde_phi2_model.score_samples(np.array([(self.data.phi1.flatten()+42)/10,
                                                             self.data.phi2.flatten()]).T)
            loglike_phi2_model = np.sum(loglike_phi2_model)

            kde_pm1 = KernelDensity(kernel='gaussian',
                                    bandwidth=0.21).fit(np.array([(model_window.phi1.value +42)/15,
                                                                  model_window.pm_phi1_cosphi2]).T)
            loglike_pm1 = kde_pm1.score_samples(np.array([(self.data.phi1.flatten()+42)/15,
                                                           self.data.pm1.flatten()]).T)
            loglike_pm1 = np.sum(loglike_pm1)

            kde_pm2 = KernelDensity(kernel='gaussian',
                                    bandwidth=0.285).fit(np.array([(model_window.phi1.value+42)/15,
                                                                  model_window.pm_phi2]).T)
            loglike_pm2 = kde_pm2.score_samples(np.array([(self.data.phi1.flatten()+42)/15,
                                                           self.data.pm2.flatten()]).T)
            loglike_pm2 = np.sum(loglike_pm2)

            kde_rv = KernelDensity(kernel='gaussian',
                                   bandwidth=0.82).fit(np.array([model_window.phi1.value*5,
                                                                model_window.radial_velocity]).T)
            loglike_rv = kde_rv.score_samples(np.array([self.gd1_rv_bonaca.phi1*5,
                                                        self.gd1_rv_bonaca.Vrad]).T)
            loglike_rv = np.sum(loglike_rv)

            ll = loglike_phi2_model + loglike_pm1 + loglike_pm2 + loglike_rv

            ###############################################
            ## EVALUATE LOG-LIKELIHOOD IN SMALLER REGION ##
            ###############################################

            #cut output of model to window
            model_window_short = self.current[(self.current.phi1.value > -43) & (self.current.phi1.value < -25)]

            # evaluate the ll using KDE
            kde_phi2_short = KernelDensity(kernel='gaussian',
                                     bandwidth=0.11).fit(np.array([(model_window_short.phi1.value)/10,
                                                                   model_window_short.phi2]).T)
            loglike_phi2_short = kde_phi2_short.score_samples(np.array([(self.data_short.phi1.flatten())/10,
                                                             self.data_short.phi2.flatten()]).T)
            loglike_phi2_short = np.sum(loglike_phi2_short)

            kde_pm1_short = KernelDensity(kernel='gaussian',
                                    bandwidth=0.21).fit(np.array([(model_window_short.phi1.value)/15,
                                                                  model_window_short.pm_phi1_cosphi2]).T)
            loglike_pm1_short = kde_pm1_short.score_samples(np.array([(self.data_short.phi1.flatten())/15,
                                                           self.data_short.pm1.flatten()]).T)
            loglike_pm1_short = np.sum(loglike_pm1_short)

            kde_pm2_short = KernelDensity(kernel='gaussian',
                                    bandwidth=0.285).fit(np.array([(model_window_short.phi1.value)/15,
                                                                  model_window_short.pm_phi2]).T)
            loglike_pm2_short = kde_pm2_short.score_samples(np.array([(self.data_short.phi1.flatten())/15,
                                                           self.data_short.pm2.flatten()]).T)
            loglike_pm2_short = np.sum(loglike_pm2_short)

            kde_rv_short = KernelDensity(kernel='gaussian',
                                   bandwidth=0.82).fit(np.array([model_window_short.phi1.value * 5,
                                                                model_window_short.radial_velocity]).T)
            loglike_rv_short = kde_rv_short.score_samples(np.array([self.gd1_rv_bonaca_short.phi1 * 5,
                                                        self.gd1_rv_bonaca_short.Vrad]).T)
            loglike_rv_short = np.sum(loglike_rv_short)

            ll_short = loglike_phi2_short + loglike_pm1_short + loglike_pm2_short + loglike_rv_short
            ll_phi2_short = loglike_phi2_short

            #########################
            ## TESTING FOR THE GAP ##
            #########################
            #no_spur_model = model_window[(model_window.phi2.value < 0.6)]
            no_spur_model = model_window
            #take the average count in the range -45 to -55, which seems to be heavily populated and consistent
            high_dens_model = no_spur_model[(no_spur_model.phi1.value < -45) & (no_spur_model.phi1.value > -54)]

            low_dens_model = no_spur_model[(no_spur_model.phi1.value < -39) & (no_spur_model.phi1.value > -43)]
            gap_ratio = (len(low_dens_model)/4) / (len(high_dens_model)/9 + 0.01)

            ## Calculate the apocenter and pericenter
            orbit_pert = self.mw.integrate_orbit(orbits[-1,0], dt=-1*u.Myr, n_steps=12000)
            pert_apo, pert_peri = orbit_pert.apocenter().value, orbit_pert.pericenter().value

            # Doing KDE on the data
            #############################
            ## EVALUATE LOG-LIKELIHOOD ##
            #############################
            kde_phi2_data = KernelDensity(kernel='gaussian', bandwidth=0.15).fit(
                                np.array([(self.data.phi1.flatten() + 42)/10, self.data.phi2.flatten()]).T)
            loglike_phi2_data = kde_phi2_data.score_samples(np.array([(model_window.phi1.value+42)/10,
                                                             model_window.phi2.value]).T)
            loglike_phi2_data = np.sum(loglike_phi2_data)
            ll_data = loglike_phi2_data

        return ll, ll_short, ll_phi2_short, ll_data, gap_ratio, pert_apo, pert_peri
