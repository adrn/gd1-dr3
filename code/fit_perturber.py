import numpy as np

import gala.coordinates as gc
import gala.potential as gp
from gala.potential import NullPotential
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic
#from gala.dynamics.nbody import DirectNBody
from Nbody_gala import DirectNBody

import astropy.coordinates as coord
import astropy.units as u
from pyia import GaiaData
#from gravhopper import Simulation, IC
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

class FitPert:
    
    def __init__(self):
        
        #########################
        ## DATA FOR COMPARISON ##
        #########################
        after = GaiaData('../data/member_prob_all.fits')
        model_output = after[after.post_member_prob > 0.3]
        self.data = model_output[(model_output.phi1[:,0] > -60) & (model_output.phi1[:,0] < -25)]
        
        
        ##########################################
        ## CURRENT STREAM WITHOUT THE PERTURBER ##
        ##########################################
        df = ms.FardalStreamDF(random_state=np.random.RandomState(42))
        gd1_init = gc.GD1Koposov10(phi1 = -13*u.degree, phi2=0*u.degree, distance=8.84*u.kpc,
                                  pm_phi1_cosphi2=-10.28*u.mas/u.yr,
                                  pm_phi2=-2.43*u.mas/u.yr,
                                  radial_velocity = -182*u.km/u.s)
        rep = gd1_init.transform_to(coord.Galactocentric).data
        gd1_w0 = gd.PhaseSpacePosition(rep)
        gd1_mass = 5e3 * u.Msun
        gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
        self.mw = gp.MilkyWayPotential(halo={'m': 5.35e11*u.Msun, 'r_s': 15.27*u.kpc})
        gen_gd1 = ms.MockStreamGenerator(df, self.mw, progenitor_potential=gd1_pot)
        gd1_stream, gd1_nbody = gen_gd1.run(gd1_w0, gd1_mass,
                                        dt=-1 * u.Myr, n_steps=3000)
        gd1 = gd1_stream.to_coord_frame(gc.GD1)

        gd1_short = gd1_stream[(-60<gd1.phi1.value) & (gd1.phi1.value<-24)]
        self.w0_now = gd.PhaseSpacePosition(gd1_short.data, gd1_short.vel)
        
    

    def pre_fitting(self, vals):
        
        self.b, self.psi, self.v_z, self.vpsi, self.t_int, self.logm, self.core = vals
        
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
        
        b, psi, v_z, vpsi = self.b * u.pc, self.psi * u.deg, self.v_z * u.km/u.s, self.vpsi * u.km/u.s

        # Define the position of the perturber at the time of impact in the
        # cylindrical impact coordinates:
        perturber_pos = coord.CylindricalRepresentation(rho=b,
                                                        phi=psi,
                                                        z=0*u.pc)
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

        vxpert = w0_pert.vel.d_x.to(u.km/u.s)
        vypert = w0_pert.vel.d_y.to(u.km/u.s)
        vzpert = w0_pert.vel.d_z.to(u.km/u.s)

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
        
        '''
        sim = Simulation(dt = 0.1*u.Myr, eps = 0.9 * u.pc)
        sim.add_external_force(self.mw)
        perturber_IC = IC.Hernquist(N = 1, a = self.core*u.pc, cutoff = 10, totmass = 10**self.logm*u.Msun, center_pos = w0_orig_pert.pos.xyz.T, center_vel = w0_orig_pert.vel.d_xyz.T, force_origin = False)
        stream_IC = {'pos': self.w0_orig_stream.pos.xyz.T, 'vel': self.w0_orig_stream.vel.d_xyz.T, 'mass': np.ones(self.w0_orig_stream.pos.xyz.shape[1]) * (1e-5 / self.w0_orig_stream.pos.xyz.shape[1]) * u.ng}
        sim.add_IC(stream_IC)
        sim.add_IC(perturber_IC)
        get_sim = sim
        sim.run(N = total_time*10)
        current = sim.current_snap()
        current = gd.PhaseSpacePosition(pos = current['pos'].T, vel = current['vel'].T)
        #self.current = current.to_coord_frame(gc.GD1)
        
        # Plot the x-y positions at the beginning and end.
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(131, aspect=1.0)
        ax2 = fig.add_subplot(132, aspect=1.0)
        ax3 = fig.add_subplot(133, aspect=1.0)
        sim.plot_particles(snap='IC', unit=u.pc, ax=ax1)
        sim.plot_particles(snap='final', xlim = [-15000, -10000], ylim = [-5000, 5000], unit=u.pc, ax=ax2)
        sim.plot_particles(snap='final', coords='yz', xlim = [-5000, 5000], ylim = [2000, 12000], unit=u.pc, ax=ax3)
        plt.show()
        
        #fig = plt.figure(figsize=(12,12))
        #sim.movie_particles('Plummer_sim.gif', unit=u.pc, ax = fig.add_subplot(111), xlim=[-17000, 15000], ylim=[-15000, 17000])
        '''
        #return self.back, orbits
        return self.current, self.orbits
        
    def loglik_model_kde(self, params):
        
        self.pre_fitting(params)
        self.nbody()
        
        #############################
        ## EVALUATE LOG-LIKELIHOOD ##
        #############################
        
        #cut output of model to window
        model_window = self.current[(self.current.phi1.value > -60) & (self.current.phi1.value < -25)]
        
        # evaluate the ll using KDE
        kde_phi2 = KernelDensity(kernel='gaussian', 
                                 bandwidth=0.11).fit(np.array([(model_window.phi1.value + 46)/10, 
                                                               model_window.phi2]).T)
        loglike_phi2 = kde_phi2.score_samples(np.array([(self.data.phi1.flatten()+46)/10, 
                                                         self.data.phi2.flatten()]).T)
        loglike_phi2 = np.sum(loglike_phi2)

        kde_pm1 = KernelDensity(kernel='gaussian',
                                bandwidth=0.21).fit(np.array([(model_window.phi1.value +46)/15,
                                                              model_window.pm_phi1_cosphi2]).T)
        loglike_pm1 = kde_pm1.score_samples(np.array([(self.data.phi1.flatten()+46)/15, 
                                                       self.data.pm1.flatten()]).T)
        loglike_pm1 = np.sum(loglike_pm1)

        kde_pm2 = KernelDensity(kernel='gaussian', 
                                bandwidth=0.285).fit(np.array([(model_window.phi1.value+46)/15, 
                                                              model_window.pm_phi2]).T)
        loglike_pm2 = kde_pm2.score_samples(np.array([(self.data.phi1.flatten()+46)/15, 
                                                       self.data.pm2.flatten()]).T)
        loglike_pm2 = np.sum(loglike_pm2)

        ll = loglike_phi2 + loglike_pm1 + loglike_pm2
        
        return ll, kde_phi2, kde_pm1, kde_pm2
    
    def loglik_data_kde(self, params):
        self.prefitting(params)
        self.nbody()
        
        #############################
        ## EVALUATE LOG-LIKELIHOOD ##
        #############################

        #cut output of model to window
        model_window = self.current[(self.current.phi1.value > -60) & (self.current.phi1.value < -25)]
        kde_phi2_data = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(
                            np.array([(data.phi1.flatten() + 46)/10, data.phi2.flatten()]).T)
        loglike_phi2 = kde_phi2_data.score_samples(np.array([(model_window.phi1.value+46)/10, 
                                                         model_window.phi2.value]).T)
        loglike_phi2 = np.sum(loglike_phi2)
        ll = loglike_phi2
        return ll, kde_phi2_data, model_window
        
#if __name__ == '__main__':