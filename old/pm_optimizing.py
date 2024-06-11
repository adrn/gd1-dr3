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
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, interp1d
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
import read_mist_models


print(f"Running on PyMC3 v{pm.__version__}")

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
    
plot_pretty(fontsize=20, labelsize=20)

gaia = GaiaData('../data/gd1_ps1_with_basic_masks_thin.fits')

stream_mask = gaia.gi_cmd_mask
g = gaia[(stream_mask)]

dist = g.get_distance(min_parallax=1e-3*u.mas)
c = g.get_skycoord(distance=dist)
stream_coord = c.transform_to(gc.GD1)
phi1 = stream_coord.phi1.degree
phi2 = stream_coord.phi2.degree
pm1 = stream_coord.pm_phi1_cosphi2
pm2 = stream_coord.pm_phi2

after = GaiaData('../data/sorted_pm_member_prob_all_stars_8comp.fits')

g_sorted, obs_pm_all, obs_pm_cov_all, phi1_stream_all, phi2_stream_all, bkg_ind = pmf.pre_model(gaia, g, after)
phi1_stream = phi1_stream_all[bkg_ind]
ln_bg_prob_all = after.pm_ln_bkg_prob.astype('float64')
ln_bg_prob = ln_bg_prob_all[bkg_ind]

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
    
    res, logp = pmx.optimize(start={'b4': 0.45, 'std_phi2_spur': 0.15,'beta': 0.3}, 
                             return_info = True)
    
post_member_prob_all = np.exp(
        res['ll_fg_full_all'] 
        - np.logaddexp(res['ll_fg_full_all'], res['ll_bg_full_all']))

post_member_prob_pm_all = np.exp(
        res['ll_fg_pm_all'] 
        - np.logaddexp(res['ll_fg_pm_all'], res['ll_bg_full_all']))

post_member_prob_phi2_all = np.exp(
        res['ll_fg_phi2_total_all'] 
        - np.logaddexp(res['ll_fg_phi2_total_all'], res['ll_bg_full_all']))
    
    
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

tbl.write('../data/member_prob_all.fits', overwrite=True)

spline_pm1 = UnivariateSpline(phi1_stream, res['mean_pm_stream'][:,0], k=5, s = 5)
spline_pm2 = UnivariateSpline(phi1_stream, res['mean_pm_stream'][:,1], k=5, s = 5)
spline_phi2 = UnivariateSpline(phi2_stream, res['mean_phi2_stream'], k=5, s = 5)

#np.save('../data/phi1_stream_from_pm_model.npy', phi1_stream)
#np.save('../data/gd1_track.npy', spline_phi2(phi1_stream))
#np.save('../data/pm1_from_model.npy', spline_pm1(phi1_stream))
#np.save('../data/pm2_from_model.npy', spline_pm2(phi1_stream))








after = GaiaData('../data/member_prob_all.fits')

high_memb_prob_pm = after[after.post_member_prob_pm > 0.3]
high_memb_prob_phi2 = after[after.post_member_prob_phi2 > 0.3]
high_memb_prob = after[(after.post_member_prob > 0.3)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (18, 11))

ax1.scatter(high_memb_prob_pm.phi1, high_memb_prob_pm.phi2, c = high_memb_prob_pm.post_member_prob_pm,
            s = 5, cmap='plasma_r', vmax = 1)
plt.colorbar(ax = ax1)
ax1.set_xlim(-100, 20); plt.ylim(-10, 5); 
ax1.set_xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
ax1.set_title(r'Proper Motion Memb Prob')

plt.scatter(high_memb_prob_phi2.phi1, high_memb_prob_phi.phi2, c=high_memb_prob_phi2.post_member_prob_phi2, 
            s = 5, cmap='plasma_r', vmin=0.5, vmax=1)
plt.colorbar(ax = ax2)
ax2.set_xlim(-100, 20); plt.ylim(-10, 5); 
ax2.set_xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
ax2.set_title(r'Phi2 Membership Probability')


ax3.scatter(high_memb_prob.phi1, high_memb_prob.phi2, c = high_memb_prob.post_member_prob,
            s = 5, cmap='plasma_r', vmax=1)
plt.colorbar(ax = ax3)
ax3.set_xlim(-100, 20); plt.ylim(-6, 3); 
ax3.set_xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
ax3.set_title(r'Membership Probabilities Combined')
    
    
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (18, 11))
ax1.fill_between(phi1_stream[:,0], res['mean_phi2_stream']+ res['std_phi2_stream'],
                              res['mean_phi2_stream']- res['std_phi2_stream'], color='r',alpha=0.5)
ax1.set_title(r'Track of Stream with Width')
ax1.scatter(high_memb_prob3.phi1, high_memb_prob3.phi2, s = 4)
ax1.set_xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\phi_2$ [deg]')
ax1.set_xlim(-100, 20); plt.ylim(-6,2)

ax2.plot(phi1_stream, res['mean_pm_stream'][:,0])
ax2.scatter(high_memb_prob3.phi1, high_memb_prob3.pm1, s = 4)
ax2.set_title(r'PM1 Along Stream')
ax2.set_xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\mu_{\phi_1}$ [deg]')
ax2.set_xlim(-100, 20); plt.ylim(-15,0)

ax3.plot(phi1_stream, res['mean_pm_stream'][:,1])
ax3.scatter(high_memb_prob3.phi1, high_memb_prob3.pm2, s = 4)
ax3.set_title(r'PM2 along Stream')
ax3.set_xlabel(r'$\phi_1$ [deg]'); plt.ylabel(r'$\mu_{\phi_2}$ [deg]')
ax3.set_xlim(-100, 20); plt.ylim(-5,0)
 

sections = np.arange(-100,15,5)
dm = np.concatenate([[14.7, 14.6, 14.5, 14.45, 14.4, 14.35, 14.3, 14.3], 
                     np.linspace(14.3, 14.6, 9), 
                     [14.71, 14.75, 14.8, 15, 15.2, 15.4]])
spline_dm = UnivariateSpline(sections, dm, k=5) 

#Load in MIST isochrone:
feh_sel, age_select = -1.7, 13
iso_file = '../data/isochrones/MIST_PS_iso_feh_{}_vrot0.iso.cmd'.format(feh_sel)
isocmd = read_mist_models.ISOCMD(iso_file)

age_sel = isocmd.age_index(age_select*10**9)
gmag = isocmd.isocmds[age_sel]['PS_g']
imag = isocmd.isocmds[age_sel]['PS_i']
gi_color = gmag-imag

after = GaiaData('../data/member_prob_all.fits')
high_memb_prob_pm = after[after.post_member_prob_pm > 0.1]
high_memb_prob = after[(after.post_member_prob > 0.1)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
ax1.scatter(gi_color, gmag, c='r', s = 2)
after_phi1 = high_memb_prob.phi1.reshape(high_memb_prob.g_0.shape)
ax1.scatter(high_memb_prob.g_0 - high_memb_prob.i_0, high_memb_prob.g_0 - spline_dm(after_phi1),  
            s = 5, c= 'k', alpha = 0.5)
ax1.set_ylim(7, -1)
ax1.set_xlim(-0.8, 1.2)
ax1.set_title('Combined Fit')

ax2.scatter(gi_color, gmag, c='r', s = 2)
after_phi1 = high_memb_prob_pm.phi1.reshape(high_memb_prob_pm.g_0.shape)
ax2.scatter(high_memb_prob_pm.g_0 - high_memb_prob_pm.i_0, high_memb_prob_pm.g_0 - spline_dm(after_phi1),  
             s = 5, c= 'k', alpha = 0.5)
ax2.set_ylim(7, -1)
ax2.set_xlim(-0.5, 1.2)
ax2.set_title('PM fit only')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    