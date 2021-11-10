import pathlib
import warnings
import warnings
warnings.filterwarnings('ignore')
import os

import sys
sys.path.append('../code/')
import pm_model_func as pmf

# Third-party
import astropy.coordinates as coord
import astropy.table as at
from astropy.table import Table, vstack
from astropy.io import fits
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.lib.recfunctions import stack_arrays
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.ndimage.filters import gaussian_filter

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from pyia import GaiaData
from scipy.stats import binned_statistic

import arviz as az
import pymc3 as pm
import seaborn as sns
from tqdm import trange
from pymc3 import *
import theano.tensor as tt
import pymc3_ext as pmx
from patsy import dmatrix

print(f"Running on PyMC3 v{pm.__version__}")


if __name__ == '__main__':
    #########################
    ## SETTING UP THE DATA ##
    #########################
    gaia = GaiaData('../data/gd1_ps1_with_basic_masks_thin.fits')

    stream_mask = gaia.gi_cmd_mask
    g = gaia[(stream_mask)]# & (gaia.g_0 < 18)]

    dist = g.get_distance(min_parallax=1e-3*u.mas)
    c = g.get_skycoord(distance=dist)
    stream_coord = c.transform_to(gc.GD1)
    phi1 = stream_coord.phi1.degree
    phi2 = stream_coord.phi2.degree
    pm1 = stream_coord.pm_phi1_cosphi2
    pm2 = stream_coord.pm_phi2

    after = GaiaData('../data/sorted_pm_member_prob_all_stars_8comp.fits')

    g_sorted, obs_pm_all, obs_pm_cov_all, phi1_stream_all, phi2_stream_all, bkg_ind = pmf.pre_model(gaia, g, after)
    ln_bg_prob_all = after.pm_ln_bkg_prob.astype('float64')
    ln_bg_prob = ln_bg_prob_all[bkg_ind]

    #######################
    ## WRITING THE MODEL ##
    #######################
    with pm.Model() as model:
        n_pm_nodes, n_track_nodes, n_width_nodes = 8, 8, 5

        # mixture weight
        alpha = pm.Uniform('alpha', lower = 0, upper = 1, testval=5e-4)
        beta = pm.Uniform('beta', lower=0, upper = 1, testval=0.3)

        loglike_fg_pm, loglike_fg_pm_all = pmf.pm_model_spline(model, obs_pm_all, obs_pm_cov_all, 
                                                               phi1_stream_all, bkg_ind, n_pm_nodes)
        ll_fg_pm = pm.Deterministic('ll_fg_pm', tt.log(alpha) + loglike_fg_pm)
        ll_fg_pm_all = pm.Deterministic('ll_fg_pm_all', tt.log(alpha) + loglike_fg_pm_all)

        loglike_fg_phi2, loglike_fg_phi2_all = pmf.phi2_model_spline(model, phi1_stream_all, phi2_stream_all, bkg_ind,
                                                                     n_track_nodes, n_width_nodes)
        loglike_fg_phi2_all = loglike_fg_phi2_all.reshape(loglike_fg_pm_all.shape)
        loglike_fg_phi2 = loglike_fg_phi2.reshape(loglike_fg_pm.shape)
        ll_fg_phi2_all = pm.Deterministic('ll_fg_phi2_all', tt.log(beta) + loglike_fg_phi2_all)
        ll_fg_phi2 = pm.Deterministic('ll_fg_phi2', tt.log(beta) + loglike_fg_phi2)

        loglike_fg_spur, loglike_fg_spur_all = pmf.spur_model(model, phi1_stream_all, phi2_stream_all, bkg_ind)
        loglike_fg_spur_all = loglike_fg_spur_all.reshape(loglike_fg_pm_all.shape)
        loglike_fg_spur = loglike_fg_spur.reshape(loglike_fg_pm.shape)
        ll_fg_phi2_spur_all = pm.Deterministic('ll_fg_phi2_spur_all', 
                                               tt.log(alpha) + tt.log(1-beta) + loglike_fg_spur_all)
        ll_fg_phi2_spur = pm.Deterministic('ll_fg_phi2_spur', tt.log(alpha) + tt.log(1-beta) + loglike_fg_spur)

        #total track likelihood (including spur)
        loglike_fg_phi2_total_all = pm.Deterministic('ll_fg_phi2_total_all', 
                                                     pm.logaddexp(loglike_fg_phi2_all, loglike_fg_spur_all))
        loglike_fg_phi2_total = pm.Deterministic('ll_fg_phi2_total', pm.logaddexp(loglike_fg_phi2, loglike_fg_spur))

        #total foreground likelihood
        loglike_fg_all = loglike_fg_pm_all + loglike_fg_phi2_total_all
        loglike_fg = loglike_fg_pm + loglike_fg_phi2_total
        ll_fg_full_all = pm.Deterministic('ll_fg_full_all', tt.log(alpha) + loglike_fg_all)
        ll_fg_full = pm.Deterministic('ll_fg_full', tt.log(alpha) + loglike_fg)

        ll_bg_full_all = pm.Deterministic('ll_bg_full_all', tt.log(1 - alpha) + ln_bg_prob_all)
        ll_bg_full = pm.Deterministic('ll_bg_full', tt.log(1 - alpha) + ln_bg_prob)

        loglike = pm.logaddexp(ll_fg_full, ll_bg_full)
        pm.Potential("loglike", loglike)

        #######################
        ## OPTIMIZE THE MODEL##
        #######################
        res, logp = pmx.optimize(start={'b4': 0.3,
                                    'ln_std_phi2_spur': np.log(0.1),
                                    'beta': 0.3}, 
                                 return_info = True)

        #######################
        ## SAMPLING THE MODEL##
        #######################
        trace = pmx.sample(draws = 250,tune=250, start=res, chains=2, cores=1)

        data = az.from_pymc3(trace=trace)
        az.to_netcdf(data=data, filename = '../data/sample_outputs/trace0.netcdf')
        print(trace)
        az.plot_trace(trace)
        az.summary()

        az.plot_posterior(trace)


    post_member_prob3_all = np.exp(
                res['ll_fg_full_all'] 
                - np.logaddexp(res['ll_fg_full_all'], res['ll_bg_full_all']))

    post_member_prob3_pm_all = np.exp(
                res['ll_fg_pm_all'] 
                - np.logaddexp(res['ll_fg_pm_all'], res['ll_bg_full_all']))

    post_member_prob3_phi2_all = np.exp(
                res['ll_fg_phi2_total_all'] 
                - np.logaddexp(res['ll_fg_phi2_total_all'], res['ll_bg_full_all']))

    print('# among sel stars with total member prob > 0.1: {}'.format((post_member_prob3 > 0.1).sum()))
    print('# among sel stars with PM member prob > 0.1: {}'.format((post_member_prob3_pm > 0.1).sum()))
    print('# among sel stars with track member prob > 0.1: {}'.format((post_member_prob3_phi2 > 0.1).sum()))
    print('-------------------------------------------')
    print('# among all stars with total member prob > 0.1: {}'.format((post_member_prob3_all > 0.1).sum()))
    print('# among all stars with PM member prob > 0.1: {}'.format((post_member_prob3_pm_all > 0.1).sum()))
    print('# among all stars with track member prob > 0.1: {}'.format((post_member_prob3_phi2_all > 0.1).sum()))


    tbl = at.Table()
    tbl['phi1'] = phi1_stream_all
    tbl['phi2'] = phi2_stream_all
    tbl['g_0'] = g_sorted.g_0
    tbl['i_0'] = g_sorted.i_0
    tbl['pm1'] = obs_pm_all[:,0]
    tbl['pm2'] = obs_pm_all[:,1]
    tbl['pm_cov'] = obs_pm_cov_all
    tbl['ln_bg_prob'] = ln_bg_prob_all
    tbl['post_member_prob'] = post_member_prob3_all
    tbl['post_member_prob_pm'] = post_member_prob3_pm_all
    tbl['post_member_prob_phi2'] = post_member_prob3_phi2_all

    tbl.write('../data/member_prob_sample_all.fits', overwrite=True)

    after3 = GaiaData('../data/member_prob_sample_all.fits')

    high_memb_prob3_pm = after3[after3.post_member_prob_pm > 0.3]
    high_memb_prob3_phi2 = after3[after3.post_member_prob_phi2 > 0.3]
    high_memb_prob3 = after3[(after3.post_member_prob > 0.3)]

    plt.figure(figsize=(18,3))
    plt.scatter(high_memb_prob3_pm.phi1, high_memb_prob3_pm.phi2, c = high_memb_prob3_pm.post_member_prob_pm,
                s = 5, cmap='plasma_r', vmax = 1)
    plt.colorbar()
    plt.xlim(-100, 20); plt.ylim(-10, 5); 
    plt.xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
    plt.title(r'Proper Motion Memb Prob')

    plt.figure(figsize=(18,3))
    plt.scatter(high_memb_prob3_phi2.phi1, high_memb_prob3_phi2.phi2, c=high_memb_prob3_phi2.post_member_prob_phi2, 
                s = 5, cmap='plasma_r', vmin=0.5, vmax=1)
    plt.colorbar()
    plt.xlim(-100, 20); plt.ylim(-10, 5); 
    plt.xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
    plt.title(r'Phi2 Membership Probability')

    plt.figure(figsize=(18,3))
    plt.scatter(high_memb_prob3.phi1, high_memb_prob3.phi2, c = high_memb_prob3.post_member_prob, 
                s = 5, cmap='plasma_r', vmax=1)
    plt.colorbar()
    plt.xlim(-100, 20); plt.ylim(-6, 3); 
    plt.xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
    plt.title(r'Membership Probabilities Combined')

    plt.figure(figsize=(15,3))
    plt.scatter(high_memb_prob3.phi1, high_memb_prob3.phi2,
                s = 5, c= 'k')
    plt.xlim(-100, 20); plt.ylim(-6, 3); 
    plt.scatter(-12,-0.2)
    plt.xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
    plt.title(r'Membership Probabilities Combined')
    #plt.savefig('../memb_probabilities_stream_with_spur.jpg')





    phi1_stream = phi1_stream_all[bkg_ind]
    plt.figure(figsize = (14,5))
    plt.plot(phi1_stream, res['std_phi2_stream'])
    plt.title(r'Width of Stream')
    plt.xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'Width [deg]')
    plt.xlim(-100, 20); plt.ylim(0,0.5)
    plt.grid()

    plt.figure(figsize = (14,2))
    plt.fill_between(phi1_stream[:,0], res['mean_phi2_stream']+ res['std_phi2_stream'],
                                  res['mean_phi2_stream']- res['std_phi2_stream'], color='r',alpha=0.5)
    plt.title(r'Track of Stream with Width')
    plt.scatter(high_memb_prob3.phi1, high_memb_prob3.phi2, s = 4)
    plt.xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
    plt.xlim(-100, 20); plt.ylim(-6,2)
    plt.grid()

    plt.figure(figsize = (14,2))
    plt.plot(phi1_stream, res['mean_pm_stream'][:,0])
    plt.scatter(high_memb_prob3.phi1, high_memb_prob3.pm1, s = 4)
    plt.title(r'PM1 Along Stream')
    plt.xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\mu_{\phi_1}$ [deg]')
    plt.xlim(-100, 20); plt.ylim(-15,0)
    plt.grid()

    plt.figure(figsize = (14,2))
    plt.plot(phi1_stream, res['mean_pm_stream'][:,1])
    plt.scatter(high_memb_prob3.phi1, high_memb_prob3.pm2, s = 4)
    plt.title(r'PM2 along Stream')
    plt.xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\mu_{\phi_2}$ [deg]')
    plt.xlim(-100, 20); plt.ylim(-5,0)
    plt.grid()
