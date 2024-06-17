import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic

import astropy.coordinates as coord
import astropy.units as u
from pyia import GaiaData
import glob
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

import sys
import fit_perturber as fp

from PIL import Image

def plot_pretty(dpi=175, fontsize=15, labelsize=15, figsize=(10, 8), tex=True):
    # import pyplot and set some parameters to make plots prettier
    plt.rc('savefig', dpi=dpi)
    plt.rc('text', usetex=tex)
    plt.rc('font', size=fontsize)
    plt.rc('xtick.major', pad=1)
    plt.rc('xtick.minor', pad=1)
    plt.rc('ytick.major', pad=1)
    plt.rc('ytick.minor', pad=1)
    plt.rc('figure', figsize=figsize)
    mpl.rcParams['xtick.labelsize'] = labelsize
    mpl.rcParams['ytick.labelsize'] = labelsize
    mpl.rcParams.update({'figure.autolayout': False})


class Plotting():

    def __init__(self):
        after = GaiaData('../data/member_prob_all.fits')
        self.model_output = after[after.post_member_prob > 0.1]
        self.model_output_pm = after[after.post_member_prob_pm > 0.3]

        phi1_stream_pm_model = np.load('../data/phi1_stream_from_pm_model.npy')
        stream_pm10 = np.load('../data/true_pm1_from_model.npy')
        self.spline_pm1 = InterpolatedUnivariateSpline(phi1_stream_pm_model[::10], stream_pm10[::10])
        stream_pm20 = np.load('../data/true_pm2_from_model.npy')
        self.spline_pm2 = InterpolatedUnivariateSpline(phi1_stream_pm_model[::10], stream_pm20[::10])

        est_track = np.load('../data/gd1_track.npy')
        self.spline_phi2 = UnivariateSpline(phi1_stream_pm_model.reshape(est_track.shape)[::10],
                                                   est_track[::10])

        self.phi1s = np.linspace(-100, 20, 120)

        # Get the Orbit Fit
        df = ms.FardalStreamDF(random_state=np.random.RandomState(42))
        gd1_init = gc.GD1Koposov10(phi1 = -13*u.degree, phi2=0*u.degree, distance=8.88*u.kpc,
                              pm_phi1_cosphi2=-10.245*u.mas/u.yr,
                              pm_phi2=-2.429*u.mas/u.yr,
                             radial_velocity = -182.9*u.km/u.s)
        rep = gd1_init.transform_to(coord.Galactocentric).data
        gd1_w0 = gd.PhaseSpacePosition(rep)
        gd1_mass = 5e3 * u.Msun
        gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
        mw = gp.MilkyWayPotential(halo={'m':5.73e11*u.Msun, 'r_s': 16*u.kpc})
        gen_gd1 = ms.MockStreamGenerator(df, mw, progenitor_potential=gd1_pot)
        gd1_stream, gd1_nbody = gen_gd1.run(gd1_w0, gd1_mass,
                                        dt=-1 * u.Myr, n_steps=3000)
        gd1_stream_c1 = gd1_stream.to_coord_frame(gc.GD1)
        self.gd1 = gd1_stream_c1[gd1_stream_c1.phi1.argsort()]

        self.fitpert = fp.FitPert()


    def plot_orbit_fit(self):

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,6))
        ax1.scatter(self.gd1.phi1.value, self.gd1.phi2.value, s=0.1, c='g', label='Mock Stream')
        ax1.plot(self.phi1s, self.spline_phi2(self.phi1s), c='r', label='Best Fit From From Data')
        ax1.set_xlim(-100, 20)
        ax1.set_ylim(-6, 3)
        ax1.set_ylabel(r'$\phi_2$')
        ax1.legend()

        ax2.scatter(self.gd1.phi1, self.gd1.pm_phi1_cosphi2, s=0.1)
        ax2.plot(self.phi1s, self.spline_pm1(self.phi1s), c='r', lw=1)
        ax2.set_xlim(-100, 20)
        ax2.set_ylim(-15,0)
        ax2.set_ylabel(r'$\mu_{phi_1}$')

        ax3.scatter(self.gd1.phi1, self.gd1.pm_phi2, s=0.1)
        ax3.plot(self.phi1s, self.spline_pm2(self.phi1s), c='r')
        ax3.set_xlim(-100, 20)
        ax3.set_ylim(-4.5,-1.5)
        ax3.set_ylabel(r'$\mu_{phi_2}$')

    def plot_short_orbit_fit(self):


        fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
        ax1.scatter(self.gd1.phi1.value, self.gd1.phi2.value, s=0.1, c='g', label='Mock Stream')
        ax1.plot(self.phi1s, self.spline_phi2(self.phi1s), c='r', label='Best Fit From From Data')
        ax1.set_xlim(-60, -25)
        ax1.set_ylim(-6, 3)
        ax1.set_ylabel(r'$\phi_2$')
        ax1.legend()

        ax2.scatter(self.gd1.phi1, self.gd1.pm_phi1_cosphi2, s=0.1)
        ax2.plot(self.phi1s, self.spline_pm1(self.phi1s), c='r', lw=1)
        ax2.set_xlim(-60, -25)
        ax2.set_ylim(-15,-11)
        ax2.set_ylabel(r'$\mu_{phi_1}$')

        ax3.scatter(self.gd1.phi1, self.gd1.pm_phi2, s=0.1)
        ax3.plot(self.phi1s, self.spline_pm2(self.phi1s), c='r')
        ax3.set_xlim(-60, -25)
        ax3.set_ylim(-4.5,-2)
        ax3.set_ylabel(r'$\mu_{phi_2}$')

    def kde_plot(self, params):
        ll, kde_phi2, kde_pm1, kde_pm2 = self.fitpert.loglik(params)

        data = self.model_output[(self.model_output.phi1[:,0] > -60) & (self.model_output.phi1[:,0] < -25)]


        fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8), sharex=True)
        X, Y = np.meshgrid(np.linspace(-60, -25, 500), np.linspace(-6, 3,100))
        sc1 = kde_phi2.score_samples(np.array([(X.flatten()+46)/10, Y.flatten()]).T).reshape(100, 500)
        ax1.pcolormesh(X, Y, np.e**sc1, cmap='plasma_r', shading='auto')
        ax1.scatter(data.phi1, data.phi2, s=10, alpha=1)
        ax1.set_ylabel(r'$\phi_2$')

        X, Y = np.meshgrid(np.linspace(-60, -25, 500), np.linspace(-15, -11,100))
        sc2 = kde_pm1.score_samples(np.array([(X.flatten()+46)/15, Y.flatten()]).T).reshape(100, 500)
        ax2.pcolormesh(X, Y, np.e**sc2, cmap='plasma_r', shading='auto')
        ax2.scatter(data.phi1, data.pm1, s=10, alpha=1)
        ax2.set_ylabel(r'$\mu_{\phi_1}$')

        X, Y = np.meshgrid(np.linspace(-60, -25, 500), np.linspace(-4.5, -2, 100))
        sc3 = kde_pm2.score_samples(np.array([(X.flatten()+46)/15, Y.flatten()]).T).reshape(100, 500)
        ax3.pcolormesh(X, Y, np.e**sc3, cmap='plasma_r', shading='auto')
        ax3.scatter(data.phi1, data.pm2, s=10, alpha=1)
        ax3.set_xlabel(r'$\phi_1$')
        ax3.set_ylabel(r'$\mu_{\phi_2}$')

    def final_pos(self, orbits, opt):

        plt.figure(figsize=(12,8))
        gs = gridspec.GridSpec(2, 3)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax4 = plt.subplot(gs[1,:])

        orbits[-1, 1:].plot(alpha=0.3, c='b', s=5, axes=[ax1, ax2, ax3])
        ax1.set_xlim(np.sort(orbits[-1, 1:].pos.x.value)[5]-2, np.sort(orbits[-1, 1:].pos.x.value)[-5]+2)
        ax1.set_ylim(np.sort(orbits[-1, 1:].pos.y.value)[5]-2, np.sort(orbits[-1, 1:].pos.y.value)[-5]+2)
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax2.set_xlim(np.sort(orbits[-1, 1:].pos.x.value)[5]-2, np.sort(orbits[-1, 1:].pos.x.value)[-5]+2)
        ax2.set_ylim(np.sort(orbits[-1, 1:].pos.z.value)[5]-2, np.sort(orbits[-1, 1:].pos.z.value)[-5]+2)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax3.set_xlim(np.sort(orbits[-1, 1:].pos.y.value)[5]-2, np.sort(orbits[-1, 1:].pos.y.value)[-5]+2)
        ax3.set_ylim(np.sort(orbits[-1, 1:].pos.z.value)[5]-2, np.sort(orbits[-1, 1:].pos.z.value)[-5]+2)
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)

        back0 = orbits[-1, 1:].to_coord_frame(gc.GD1)
        ax4.set_xlim(-68, -24)
        ax4.set_ylim(-5, 3)
        ax4.scatter(back0.phi1, back0.phi2, s = 3, c='b', alpha=0.3)
        ax4.scatter(self.model_output_pm.phi1, self.model_output_pm.phi2, s = 5, c='r', alpha = 0.4)
        plt.savefig('../image_folders/gd1_coord_grid_search_opt{}/final_pos'.format(opt), dpi=100)




def save_gifs(t_int, orbits, idx):
    opt = idx
    if not os.path.isdir('../image_folders/xyz_grid_search_opt{}'.format(opt)):
        print('making directory')
        os.mkdir('../image_folders/xyz_grid_search_opt{}'.format(opt))
        os.mkdir('../image_folders/gd1_coord_grid_search_opt{}'.format(opt))

        '''
        sim.run(N = t_int+30)
        for i in range(t_int+30):
            fig = plt.figure(figsize=(15,5))
            ax1 = fig.add_subplot(131, aspect=1.0)
            ax2 = fig.add_subplot(132, aspect=1.0)
            ax3 = fig.add_subplot(133, aspect=1.0)
            sim.plot_particles(snap=i, coords='xy', unit=u.pc, ax=ax1, xlim=[-15000, -10000], ylim = [-5000, 5000])
            sim.plot_particles(snap=i, coords='xz', unit=u.pc, ax=ax2, xlim=[-15000, -10000], ylim = [2000, 12000])
            sim.plot_particles(snap=i, coords='yz', unit=u.pc, ax=ax3, xlim=[-5000, 5000], ylim = [2000, 12000])
            plt.savefig('../image_folders/xyz_grid_search_opt{}/image_{}'.format(opt, str(i)), dpi=100)
            plt.close(fig)

            snap = sim.snap(i)
            snap = gd.PhaseSpacePosition(pos = snap['pos'].T, vel = snap['vel'].T)
            snap = snap.to_coord_frame(gc.GD1)
            fig, ax4 = plt.subplots(1, 1, figsize=(10,3))
            plt.ion()
            ax4.set_xlim(np.sort(snap.phi1.value)[2]-2, np.sort(snap.phi1.value)[-3]+2)
            ax4.set_ylim(np.sort(snap.phi2.value)[2]-2, np.sort(snap.phi2.value)[-3]+2)
            ax4.scatter(snap.phi1, snap.phi2, s = 1, c='b', alpha=0.3)
            ax4.set_xlim(-70, -20)
            ax4.set_ylim(-6,3)
            plt.savefig('../image_folders/gd1_coord_grid_search_opt{}/image_{}'.format(opt, str(i)), dpi=100)
            plt.close(fig)
        '''

        for i in range(t_int+30):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
            plt.ion()
            if i<(30*2):
                orbits[i, 0].plot(c='r', s=200, axes=[ax1, ax2, ax3])
            orbits[i, 1:].plot(alpha=0.3, c='b', s=5, axes=[ax1, ax2, ax3])
            ax1.set_xlim(np.sort(orbits[i, 1:].pos.x.value)[5]-2, np.sort(orbits[i, 1:].pos.x.value)[-5]+2)
            ax1.set_ylim(np.sort(orbits[i, 1:].pos.y.value)[5]-2, np.sort(orbits[i, 1:].pos.y.value)[-5]+2)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax2.set_xlim(np.sort(orbits[i, 1:].pos.x.value)[5]-2, np.sort(orbits[i, 1:].pos.x.value)[-5]+2)
            ax2.set_ylim(np.sort(orbits[i, 1:].pos.z.value)[5]-2, np.sort(orbits[i, 1:].pos.z.value)[-5]+2)
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            ax3.set_xlim(np.sort(orbits[i, 1:].pos.y.value)[5]-2, np.sort(orbits[i, 1:].pos.y.value)[-5]+2)
            ax3.set_ylim(np.sort(orbits[i, 1:].pos.z.value)[5]-2, np.sort(orbits[i, 1:].pos.z.value)[-5]+2)
            ax3.xaxis.set_visible(False)
            ax3.yaxis.set_visible(False)
            plt.savefig('../image_folders/xyz_grid_search_opt{}/image_{}'.format(opt, str(i)), dpi=100)
            plt.close(fig)
            fig, ax4 = plt.subplots(1, 1, figsize=(10,3))
            plt.ion()
            back0 = orbits[i, 1:].to_coord_frame(gc.GD1)
            ax4.set_xlim(np.sort(back0.phi1.value)[20]-2, np.sort(back0.phi1.value)[-20]+2)
            ax4.set_ylim(np.median(back0.phi2.value)-6, np.median(back0.phi2.value)+6)
            ax4.scatter(back0.phi1, back0.phi2, s = 1, c='b', alpha=0.3)
            plt.savefig('../image_folders/gd1_coord_grid_search_opt{}/image_{}'.format(opt, str(i)), dpi=100)
            plt.close(fig)

        # Create the frames
        frames = []
        imgs = sorted(glob.glob('../image_folders/xyz_grid_search_opt{}/*.png'.format(opt)), key=os.path.getmtime)
        for i in imgs:
            new_frame = Image.open(i)
            a = new_frame.copy()
            new_frame.close()
            frames.append(a)

        frames[0].save('../image_gifs/xyz_grid_search_opt{}.gif'.format(opt), format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=40, loop=0)

        # Create the frames
        frames1 = []
        imgs1 = sorted(glob.glob('../image_folders/gd1_coord_grid_search_opt{}/*.png'.format(opt)), key=os.path.getmtime)
        for i in imgs1:
            new_frame = Image.open(i)
            a = new_frame.copy()
            new_frame.close()
            frames1.append(a)

        frames1[0].save('../image_gifs/gd1_coord_grid_search_opt{}.gif'.format(opt), format='GIF',
                       append_images=frames1[1:],
                       save_all=True,
                       duration=40, loop=0)
        #Note: opt1 is with vxpert, vypert, vzpert = -21.4, 11.7, 31.5
        #Note: opt2 is with vxpert,vypert, vzpert = -5.2, -33.28, -21.22
        #Note: opt3 is with (b, psi, vz, vpsi, t, logm, core) = (1, 345, 100, 20, 100, 6.8, 1.0)
    else:
        print('Folder already exists')

if __name__ == '__main__':

    # Given the parameter values, create all the relevant plots

    plot_pretty(fontsize=15, labelsize=15)

    Plots = Plotting()

    # First plot: Just the data
    print('Making plot 1...')
    plt.figure(figsize=(14, 3))
    plt.scatter(Plots.model_output.phi1, Plots.model_output.phi2, c='k', s = 5)
    plt.ylim(-7, 3)
    plt.xlim(-100, 20)

    # Second plot: orbit fitting output full stream
    print('Making plot 2...')
    Plots.plot_orbit_fit()

    # Third plot: orbit fitting output area of interest
    print('Making plot 3...')
    Plots.plot_short_orbit_fit()

    # Fourth plot: KDE at present day from parameters
    print('Making plot 4...')
    params = np.array([1, 345, 100, 20, 100, 6.8, 1.0])
    Plots.kde_plot(params)

    # Fifth plot: Particle positions at present day from parameters
    #print('Making plot 5...')
    #Plots.fitpert.pre_fitting(params) # make this come from inputs
    #current = Plots.fitpert.nbody()
    #Plots.final_pos(orbits)

    plt.show()

    save_gifs(100, orbits)
