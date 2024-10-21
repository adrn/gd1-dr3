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
    gaia = gaia[(gaia.phi1 > -45) & (gaia.phi1 < -25)]

    stream_mask = gaia.gi_cmd_mask
    g = gaia[(stream_mask)]# & (gaia.g_0 < 18)

    dist = g.get_distance(min_parallax=1e-3*u.mas)
    c = g.get_skycoord(distance=dist)
    stream_coord = c.transform_to(gc.GD1)
    phi1 = stream_coord.phi1.degree
    phi2 = stream_coord.phi2.degree
    pm1 = stream_coord.pm_phi1_cosphi2
    pm2 = stream_coord.pm_phi2

    after = GaiaData('../data/sorted_pm_member_prob_all_stars_8comp.fits')
    after = after[(after.phi1 > -45) & (after.phi1 < -25)]

    g_sorted, obs_pm_all, obs_pm_cov_all, phi1_stream_all, phi2_stream_all, bkg_ind = pmf.pre_model(gaia, g, after)
    ln_bg_prob_all = after.pm_ln_bkg_prob.astype('float64')
    ln_bg_prob = ln_bg_prob_all[bkg_ind]

    #######################
    ## WRITING THE MODEL ##
    #######################
    with pm.Model() as model:
        # mixture weight
        alpha = pm.Uniform('alpha', lower = 0, upper = 1)
        beta = pm.Uniform('beta', lower=0, upper = 1, testval=0.3)

        loglike_fg_pm, loglike_fg_pm_all = pmf.short_pm_model_spur(model, obs_pm_all, obs_pm_cov_all,
                                                               phi1_stream_all, bkg_ind)
        ll_fg_pm = tt.log(alpha) + loglike_fg_pm

        loglike_fg_phi2, loglike_fg_phi2_all = pmf.short_phi2_model_spur(model, phi1_stream_all,
                                                                                phi2_stream_all, bkg_ind,)
        loglike_fg_phi2 = loglike_fg_phi2.reshape(loglike_fg_pm.shape)
        ll_fg_phi2 = tt.log(beta) + loglike_fg_phi2

        loglike_fg_spur =pmf.short_spur_model(model, phi1_stream_all, phi2_stream_all,
                                                                  obs_pm_all, obs_pm_cov_all, bkg_ind)
        loglike_fg_spur = loglike_fg_spur.reshape(loglike_fg_pm.shape)
        ll_fg_phi2_spur = tt.log(alpha) + tt.log(1-beta) + loglike_fg_spur

        #total track likelihood (including spur)
        loglike_fg_phi2_total = pm.logaddexp(loglike_fg_phi2, loglike_fg_spur)

        #total foreground likelihood
        loglike_fg = loglike_fg_pm + loglike_fg_phi2_total
        ll_fg_full = tt.log(alpha) + loglike_fg

        ll_bg_full = tt.log(1 - alpha) + ln_bg_prob

        loglike = pm.logaddexp(ll_fg_full, ll_bg_full)
        pm.Potential("loglike", loglike)

        #######################
        ## OPTIMIZE THE MODEL##
        #######################
        res, logp = pmx.optimize(start={'b4': 0.45,
                                    'std_phi2_spur': 0.15,
                                    'beta': 0.3},
                                 return_info = True)

        #######################
        ## SAMPLING THE MODEL##
        #######################
        trace = pmx.sample(draws=500, tune=500, chains=2,cores=1)

        data = az.from_pymc3(trace=trace)
        az.to_netcdf(data=data, filename = '../data/sample_outputs/spur_trace0.netcdf')
