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




#from an earlier version which had tighter constraints on CMD
phi1_stream_pm_model = np.load('../data/phi1_stream_from_pm_model.npy')
phi1_stream_pm_model = phi1_stream_pm_model.reshape(len(phi1_stream_pm_model), )
stream_pm10 = np.load('../data/true_pm1_from_model.npy')
spline_pm1 = InterpolatedUnivariateSpline(phi1_stream_pm_model[::10], stream_pm10[::10])
stream_pm20 = np.load('../data/true_pm2_from_model.npy')
spline_pm2 = InterpolatedUnivariateSpline(phi1_stream_pm_model[::10], stream_pm20[::10])


est_track = np.load('../data/gd1_track.npy')
n = len(est_track)
spline_phi2 = UnivariateSpline(phi1_stream_pm_model.reshape(est_track.shape)[::10], 
                                           est_track[::10])


def searchsorted(known_array, test_array): #known array is longer
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]
    known_array_middles = known_array_sorted[1:] - np.diff(known_array_sorted.astype('f'))/2
    idx1 = np.searchsorted(known_array_middles, test_array)
    indices = index_sorted[idx1]
    return indices

def pre_model(g_all, g, after):
    #need to generate two sets for later
    g_sorted_all = g_all[g_all.phi1.argsort()]

    g_sorted_all = g_sorted_all[np.isfinite(g_sorted_all.parallax) & (g_sorted_all.parallax > 0)]
    dist = g_sorted_all.get_distance(min_parallax=1e-3*u.mas)
    c1 = g_sorted_all.get_skycoord(distance=dist)
    stream_coord = c1.transform_to(gc.GD1)
    phi1_sorted_all = stream_coord.phi1.degree
    phi2_sorted_all = stream_coord.phi2.degree
    pm1_sorted_all = stream_coord.pm_phi1_cosphi2
    pm2_sorted_all = stream_coord.pm_phi2

    obs_pm_all = np.stack((pm1_sorted_all.value, pm2_sorted_all.value)).T
    Cov_sorted = g_sorted_all.get_cov()[:, 3:5, 3:5]
    obs_pm_cov_all = gc.transform_pm_cov(c1, Cov_sorted, gc.GD1)
    
    g_sorted = g[g.phi1.argsort()]

    g_sorted = g_sorted[np.isfinite(g_sorted.parallax) & (g_sorted.parallax > 0)]
    dist = g_sorted.get_distance(min_parallax=1e-3*u.mas)
    c1 = g_sorted.get_skycoord(distance=dist)
    stream_coord = c1.transform_to(gc.GD1)
    phi1_sorted = stream_coord.phi1.degree #only need this so we can get the indices for later

    
    bkg_ind = searchsorted(after.phi1, phi1_sorted)
    
    phi1_stream_all = phi1_sorted_all.reshape(len(phi1_sorted_all),1)
    phi2_stream_all = phi2_sorted_all.reshape(len(phi2_sorted_all),1)
    
    return g_sorted_all, obs_pm_all, obs_pm_cov_all, phi1_stream_all, phi2_stream_all, bkg_ind


def pm_model_spline(model, obs_pm_all, obs_pm_cov_all, phi1_stream_all, bkg_ind, n_pm_nodes):
    with model:
        obs_pm = obs_pm_all[bkg_ind]
        obs_pm_cov = obs_pm_cov_all[bkg_ind]
        phi1_stream = phi1_stream_all[bkg_ind]
        
        pm_knots = np.linspace(-101, 21, n_pm_nodes)
        B_pm_all = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream_all, "knots": pm_knots[1:-1]},)
        B_pm_all = np.asarray(B_pm_all)
        
        B_pm = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream, "knots": pm_knots[1:-1]},)
        B_pm = np.asarray(B_pm)
        
        est_pm1_nodes = spline_pm1(np.linspace(-101, 21, n_pm_nodes+2))
        est_pm2_nodes = spline_pm2(np.linspace(-101, 21, n_pm_nodes+2))
        lower_pm1_bounds = est_pm1_nodes - 3
        lower_pm2_bounds = est_pm2_nodes - 1.5
        upper_pm1_bounds = est_pm1_nodes + 3
        upper_pm2_bounds = est_pm2_nodes + 1.5
        lower_pm = np.vstack([lower_pm1_bounds, lower_pm2_bounds]).T
        upper_pm = np.vstack([upper_pm1_bounds, upper_pm2_bounds]).T
        est_pm_nodes = np.vstack([est_pm1_nodes, est_pm2_nodes]).T
        
        pm_nodes = pm.Uniform('pm_nodes', lower=lower_pm, upper=upper_pm, shape = (B_pm.shape[1], 2))
        
        mean_pm_stream_all = pm.Deterministic('mean_pm_stream_all', tt.dot(B_pm_all, pm_nodes))
        mean_pm_stream = pm.Deterministic('mean_pm_stream', tt.dot(B_pm, pm_nodes))
        
        
        ln_std_pm_stream = pm.Uniform('ln_std_pm_stream', lower=[-4, -4], upper = [0,0], shape=2)
        std_pm_stream = tt.exp(ln_std_pm_stream)
        cov_pm_stream = tt.diag(std_pm_stream**2)
        full_cov_all = obs_pm_cov_all + cov_pm_stream
        full_cov = obs_pm_cov + cov_pm_stream
        
        #Determinant calculation
        a, a_all = full_cov[:, 0, 0], full_cov_all[:, 0, 0]
        b = c = full_cov[:, 0, 1]
        b_all = c_all = full_cov_all[:, 0, 1]
        d, d_all = full_cov[:, 1, 1], full_cov_all[:, 1, 1]
        det, det_all = a * d - b * c, a_all*d_all - b_all*c_all
        
        diff = obs_pm - mean_pm_stream
        diff_all = obs_pm_all - mean_pm_stream_all
        numer = (
            d * diff[:, 0] ** 2 
            + a * diff[:, 1] ** 2
            - (b + c) * diff[:, 0] * diff[:, 1]
        )
        numer_all = (
            d_all * diff_all[:, 0] ** 2 
            + a_all * diff_all[:, 1] ** 2
            - (b_all + c_all) * diff_all[:, 0] * diff_all[:, 1]
        )
        quad, quad_all = numer / det, numer_all / det_all
        loglike_fg = -0.5 * (quad + tt.log(det) + 2 * tt.log(2*np.pi))
        loglike_fg_all = -0.5 * (quad_all + tt.log(det_all) + 2 * tt.log(2*np.pi)) # same
        
    return loglike_fg, loglike_fg_all


def phi2_model_spline(model, phi1_stream_all, phi2_stream_all, bkg_ind, n_track_nodes, n_width_nodes):
    with model:
        phi1_stream = phi1_stream_all[bkg_ind]
        phi2_stream = phi2_stream_all[bkg_ind]
        
        track_knots = np.linspace(-101, 21, n_track_nodes)
        B_track_all = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream_all, "knots": track_knots[1:-1]},)
        B_track_all = np.asarray(B_track_all)
        
        B_track = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream, "knots": track_knots[1:-1]},)
        B_track = np.asarray(B_track)
        
        est_phi2_nodes = spline_phi2(np.linspace(-101, 21, n_track_nodes+2))
        lower_phi2 = est_phi2_nodes - 3
        upper_phi2 = est_phi2_nodes + 3
        
        track_nodes = pm.Uniform('track_nodes', lower=lower_phi2, upper=upper_phi2, 
                                   shape = B_track.shape[1])
        
        mean_phi2_stream_all = pm.Deterministic('mean_phi2_stream_all', tt.dot(B_track_all, track_nodes))
        mean_phi2_stream_all = mean_phi2_stream_all.reshape(phi2_stream_all.shape)
        
        mean_phi2_stream = pm.Deterministic('mean_phi2_stream', tt.dot(B_track, track_nodes))
        mean_phi2_stream = mean_phi2_stream.reshape(phi2_stream.shape)

        # add a component that gets the width of the stream as a function of phi1
        width_knots = np.linspace(-101, 21, n_width_nodes)
        B_width_all = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream_all, "knots": width_knots[1:-1]},)
        B_width_all = np.asarray(B_width_all)
        
        B_width = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream, "knots": width_knots[1:-1]},)
        B_width = np.asarray(B_width)
        
        width_nodes_init = 0.25+np.zeros(n_width_nodes+2)
        
        width_nodes = pm.Uniform('width_nodes', lower=0.1, upper=1, 
                                   shape = B_width.shape[1], testval=width_nodes_init)
        
        std_phi2_stream_all = pm.Deterministic('std_phi2_stream_all', tt.dot(B_width_all, width_nodes))
        var_phi2_stream_all = std_phi2_stream_all**2
        var_phi2_stream_all = var_phi2_stream_all.reshape(mean_phi2_stream_all.shape)
        
        std_phi2_stream = pm.Deterministic('std_phi2_stream', tt.dot(B_width, width_nodes))
        var_phi2_stream = std_phi2_stream**2
        var_phi2_stream = var_phi2_stream.reshape(mean_phi2_stream.shape)

        #NEW
        diff_phi2_all = phi2_stream_all - mean_phi2_stream_all
        loglike_fg_phi2_all = -0.5 * (tt.log(var_phi2_stream_all) + ((diff_phi2_all**2)/var_phi2_stream_all) + tt.log(2*np.pi))
        
        diff_phi2 = phi2_stream - mean_phi2_stream
        loglike_fg_phi2 = -0.5 * (tt.log(var_phi2_stream) + ((diff_phi2**2)/var_phi2_stream) + tt.log(2*np.pi))
        
    return loglike_fg_phi2, loglike_fg_phi2_all

def spur_model(model, phi1_stream_all, phi2_stream_all, bkg_ind):
    phi1_stream = phi1_stream_all[bkg_ind]
    phi2_stream = phi2_stream_all[bkg_ind]
    
    # track for the spur as well:
    spur_sel_all = np.where((phi1_stream_all > -43) & (phi1_stream_all < -25))[0]
    phi1_spur_all, phi2_spur_all = phi1_stream_all[spur_sel_all], phi2_stream_all[spur_sel_all]
    left_all = phi1_stream_all[np.where((phi1_stream_all < -43) & (phi1_stream_all > -101))[0]]
    right_all = phi1_stream_all[np.where((phi1_stream_all > -25) & (phi1_stream_all < 21))[0]]
    
    spur_sel = np.where((phi1_stream > -43) & (phi1_stream < -25))[0]
    phi1_spur, phi2_spur = phi1_stream[spur_sel], phi2_stream[spur_sel]
    left = phi1_stream[np.where((phi1_stream < -43) & (phi1_stream > -101))[0]]
    right = phi1_stream[np.where((phi1_stream > -25) & (phi1_stream < 21))[0]]
    
    left_all = -np.inf*tt.exp(np.ones(left_all.shape))
    right_all = -np.inf*tt.exp(np.ones(right_all.shape))
    
    left = -np.inf*tt.exp(np.ones(left.shape))
    right = -np.inf*tt.exp(np.ones(right.shape))
    
    b4 = pm.Uniform('spur_track_scale', lower=0.2, upper=1)
    
    mean_spur_track_all = pm.Deterministic('mean_spur_track_all', b4*tt.sqrt(phi1_spur_all + 43))
    mean_spur_track = pm.Deterministic('mean_spur_track', b4*tt.sqrt(phi1_spur + 43))
    
    std_phi2_spur = pm.Uniform('std_phi2_spur', lower=0, upper=1, testval = 0.15)
    var_phi2_spur = std_phi2_spur**2
    
    diff_spur_all = phi2_spur_all - mean_spur_track_all
    loglike_fg_spur_i_all = -0.5 * (tt.log(var_phi2_spur) + ((diff_spur_all**2)/var_phi2_spur) + tt.log(2*np.pi))
    loglike_fg_spur_all = tt.concatenate([left_all, loglike_fg_spur_i_all, right_all])
    
    diff_spur = phi2_spur - mean_spur_track
    loglike_fg_spur_i = -0.5 * (tt.log(var_phi2_spur) + ((diff_spur**2)/var_phi2_spur) + tt.log(2*np.pi))
    loglike_fg_spur = tt.concatenate([left, loglike_fg_spur_i, right])
    
    return loglike_fg_spur, loglike_fg_spur_all


def pm_model_spline_sample(model, obs_pm_all, obs_pm_cov_all, phi1_stream_all, bkg_ind, n_pm_nodes):
    with model:
        obs_pm = obs_pm_all[bkg_ind]
        obs_pm_cov = obs_pm_cov_all[bkg_ind]
        phi1_stream = phi1_stream_all[bkg_ind]
        
        pm_knots = np.linspace(-101, 21, n_pm_nodes)
        
        B_pm = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream, "knots": pm_knots[1:-1]},)
        B_pm = np.asarray(B_pm)
        
        est_pm1_nodes = spline_pm1(np.linspace(-101, 21, n_pm_nodes+2))
        est_pm2_nodes = spline_pm2(np.linspace(-101, 21, n_pm_nodes+2))
        lower_pm1_bounds = est_pm1_nodes - 3
        lower_pm2_bounds = est_pm2_nodes - 1.5
        upper_pm1_bounds = est_pm1_nodes + 3
        upper_pm2_bounds = est_pm2_nodes + 1.5
        lower_pm = np.vstack([lower_pm1_bounds, lower_pm2_bounds]).T
        upper_pm = np.vstack([upper_pm1_bounds, upper_pm2_bounds]).T
        est_pm_nodes = np.vstack([est_pm1_nodes, est_pm2_nodes]).T
        
        pm_nodes = pm.Uniform('pm_nodes', lower=lower_pm, upper=upper_pm, shape = (B_pm.shape[1], 2))
        
        mean_pm_stream = pm.Deterministic('mean_pm_stream', tt.dot(B_pm, pm_nodes))
        
        
        ln_std_pm_stream = pm.Uniform('ln_std_pm_stream', lower=[-4, -4], upper = [0,0], shape=2)
        std_pm_stream = tt.exp(ln_std_pm_stream)
        cov_pm_stream = tt.diag(std_pm_stream**2)
        full_cov = obs_pm_cov + cov_pm_stream
        
        #Determinant calculation
        a = full_cov[:, 0, 0]
        b = c = full_cov[:, 0, 1]
        d = full_cov[:, 1, 1]
        det = a * d - b * c
        
        diff = obs_pm - mean_pm_stream
        numer = (
            d * diff[:, 0] ** 2 
            + a * diff[:, 1] ** 2
            - (b + c) * diff[:, 0] * diff[:, 1]
        )
        
        quad = numer / det
        loglike_fg = -0.5 * (quad + tt.log(det) + 2 * tt.log(2*np.pi))
        
    return loglike_fg

def phi2_model_spline_sample(model, phi1_stream_all, phi2_stream_all, bkg_ind, n_track_nodes, n_width_nodes):
    with model:
        phi1_stream = phi1_stream_all[bkg_ind]
        phi2_stream = phi2_stream_all[bkg_ind]
        
        track_knots = np.linspace(-101, 21, n_track_nodes)
        
        B_track = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream, "knots": track_knots[1:-1]},)
        B_track = np.asarray(B_track)
        
        est_phi2_nodes = spline_phi2(np.linspace(-101, 21, n_track_nodes+2))
        lower_phi2 = est_phi2_nodes - 3
        upper_phi2 = est_phi2_nodes + 3
        
        track_nodes = pm.Uniform('track_nodes', lower=lower_phi2, upper=upper_phi2, 
                                   shape = B_track.shape[1])
        
        mean_phi2_stream = pm.Deterministic('mean_phi2_stream', tt.dot(B_track, track_nodes))
        mean_phi2_stream = mean_phi2_stream.reshape(phi2_stream.shape)

        # add a component that gets the width of the stream as a function of phi1
        width_knots = np.linspace(-101, 21, n_width_nodes)
        
        B_width = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phi1_stream, "knots": width_knots[1:-1]},)
        B_width = np.asarray(B_width)
        
        width_nodes_init = 0.25+np.zeros(n_width_nodes+2)
        
        width_nodes = pm.Uniform('width_nodes', lower=0.1, upper=1, 
                                   shape = B_width.shape[1], testval=width_nodes_init)
        
        std_phi2_stream = pm.Deterministic('std_phi2_stream', tt.dot(B_width, width_nodes))
        var_phi2_stream = std_phi2_stream**2
        var_phi2_stream = var_phi2_stream.reshape(mean_phi2_stream.shape)

        diff_phi2 = phi2_stream - mean_phi2_stream
        loglike_fg_phi2 = -0.5 * (tt.log(var_phi2_stream) + ((diff_phi2**2)/var_phi2_stream) + tt.log(2*np.pi))
        
    return loglike_fg_phi2

def spur_model_sample(model, phi1_stream_all, phi2_stream_all, bkg_ind):
    phi1_stream = phi1_stream_all[bkg_ind]
    phi2_stream = phi2_stream_all[bkg_ind]
    
    # track for the spur as well:
    
    spur_sel = np.where((phi1_stream > -43) & (phi1_stream < -25))[0]
    phi1_spur, phi2_spur = phi1_stream[spur_sel], phi2_stream[spur_sel]
    left = phi1_stream[np.where((phi1_stream < -43) & (phi1_stream > -101))[0]]
    right = phi1_stream[np.where((phi1_stream > -25) & (phi1_stream < 21))[0]]
    
    left = -np.inf*tt.exp(np.ones(left.shape))
    right = -np.inf*tt.exp(np.ones(right.shape))
    
    b4 = pm.Uniform('spur_track_scale', lower=0.2, upper=1)
    
    mean_spur_track = pm.Deterministic('mean_spur_track', b4*tt.sqrt(phi1_spur + 43))
    
    std_phi2_spur = pm.Uniform('std_phi2_spur', lower=0, upper=1, testval = 0.15)
    var_phi2_spur = std_phi2_spur**2
    
    
    diff_spur = phi2_spur - mean_spur_track
    loglike_fg_spur_i = -0.5 * (tt.log(var_phi2_spur) + ((diff_spur**2)/var_phi2_spur) + tt.log(2*np.pi))
    loglike_fg_spur = tt.concatenate([left, loglike_fg_spur_i, right])
    
    return loglike_fg_spur

def short_pm_model_spur(model, obs_pm_all, obs_pm_cov_all, phi1_stream_all, bkg_ind):
    with model:
        obs_pm = obs_pm_all[bkg_ind]
        obs_pm_cov = obs_pm_cov_all[bkg_ind]
        phi1_stream = phi1_stream_all[bkg_ind]
        
        mean_pm_stream = pm.Uniform('mean_pm_stream', lower = [-15, -4], upper = [-11, -1], shape = 2)
        
        ln_std_pm_stream = pm.Uniform('ln_std_pm_stream', lower=[-4, -4], upper = [0,0], shape=2, testval = [-3, -3])
        std_pm_stream = tt.exp(ln_std_pm_stream)
        cov_pm_stream = tt.diag(std_pm_stream**2)
        full_cov_all = obs_pm_cov_all + cov_pm_stream
        full_cov = obs_pm_cov + cov_pm_stream
        
        #Determinant calculation
        a, a_all = full_cov[:, 0, 0], full_cov_all[:, 0, 0]
        b = c = full_cov[:, 0, 1]
        b_all = c_all = full_cov_all[:, 0, 1]
        d, d_all = full_cov[:, 1, 1], full_cov_all[:, 1, 1]
        det, det_all = a * d - b * c, a_all*d_all - b_all*c_all
        
        diff = obs_pm - mean_pm_stream
        diff_all = obs_pm_all - mean_pm_stream
        numer = (
            d * diff[:, 0] ** 2 
            + a * diff[:, 1] ** 2
            - (b + c) * diff[:, 0] * diff[:, 1]
        )
        numer_all = (
            d_all * diff_all[:, 0] ** 2 
            + a_all * diff_all[:, 1] ** 2
            - (b_all + c_all) * diff_all[:, 0] * diff_all[:, 1]
        )
        quad, quad_all = numer / det, numer_all / det_all
        
        loglike_fg = -0.5 * (quad + tt.log(det) + 2 * tt.log(2*np.pi))
        loglike_fg_all = -0.5 * (quad_all + tt.log(det_all) + 2 * tt.log(2*np.pi))
        
    return loglike_fg, loglike_fg_all

def short_phi2_model_spur(model, phi1_stream_all, phi2_stream_all, bkg_ind):
    with model:
        phi1_stream = phi1_stream_all[bkg_ind]
        phi2_stream = phi2_stream_all[bkg_ind]
        
        mean_phi2_stream = pm.Uniform('mean_phi2_stream', lower = -0.5, upper = 0.5)
        
        std_phi2_stream = pm.Uniform('std_phi2_stream', lower = 0, upper = 0.5, testval = 0.2)
        var_phi2_stream = std_phi2_stream**2

        diff_phi2_all = phi2_stream_all - mean_phi2_stream
        loglike_fg_phi2_all = -0.5 * (tt.log(var_phi2_stream) + ((diff_phi2_all**2)/var_phi2_stream) + tt.log(2*np.pi))
        
        diff_phi2 = phi2_stream - mean_phi2_stream
        loglike_fg_phi2 = -0.5 * (tt.log(var_phi2_stream) + ((diff_phi2**2)/var_phi2_stream) + tt.log(2*np.pi))
        
    return loglike_fg_phi2, loglike_fg_phi2_all

def short_spur_model(model, phi1_stream_all, phi2_stream_all, obs_pm_all, obs_pm_cov_all, bkg_ind):
    phi1_stream = phi1_stream_all[bkg_ind]
    phi2_stream = phi2_stream_all[bkg_ind]
    obs_pm = obs_pm_all[bkg_ind]
    obs_pm_cov = obs_pm_cov_all[bkg_ind]
    
    # track for the spur as well:
    
    spur_sel = np.where((phi1_stream > -40) & (phi1_stream < -27))[0]
    phi1_spur, phi2_spur = phi1_stream[spur_sel], phi2_stream[spur_sel]
    left_phi2 = phi1_stream[np.where((phi1_stream < -40) & (phi1_stream > -51))[0]]
    right_phi2 = phi1_stream[np.where((phi1_stream > -27) & (phi1_stream < -19))[0]]
    
    left1_phi2 = -np.inf*tt.exp(np.ones(left_phi2.shape))
    right1_phi2 = -np.inf*tt.exp(np.ones(right_phi2.shape))
     
    b4 = pm.Uniform('spur_track_scale', lower=0.2, upper=0.7, testval=0.45)
    
    mean_spur_track = pm.Deterministic('mean_spur_track', b4*tt.sqrt(phi1_spur + 40))
    
    std_phi2_spur = pm.Uniform('std_phi2_spur', lower=0, upper=0.4, testval = 0.1)
    var_phi2_spur = std_phi2_spur**2
    
    diff_spur = phi2_spur - mean_spur_track
    loglike_fg_spur_i_phi2 = -0.5 * (tt.log(var_phi2_spur) + ((diff_spur**2)/var_phi2_spur) + tt.log(2*np.pi))
    loglike_fg_spur_phi2 = tt.concatenate([left1_phi2, loglike_fg_spur_i_phi2, right1_phi2])
    
    
    obs_pm_spur = obs_pm[spur_sel]
    obs_pm_cov_spur = obs_pm_cov[spur_sel]

    mean_pm_spur = pm.Uniform('mean_pm_spur', lower = [-15, -5], upper = [-8, 0], shape =2)

    ln_std_pm_spur = pm.Uniform('ln_std_pm_spur', lower=[-4, -4], upper = [0,0], shape=2)
    std_pm_spur = tt.exp(ln_std_pm_spur)
    cov_pm_spur = tt.diag(std_pm_spur**2)
    full_cov_spur_all = obs_pm_cov_all + cov_pm_spur
    full_cov_spur = obs_pm_cov_spur + cov_pm_spur

    #Determinant calculation
    a, a_all = full_cov_spur[:, 0, 0], full_cov_spur_all[:, 0, 0]
    b = c = full_cov_spur[:, 0, 1]
    b_all = c_all = full_cov_spur_all[:, 0, 1]
    d, d_all = full_cov_spur[:, 1, 1], full_cov_spur_all[:, 1, 1]
    det, det_all = a * d - b * c, a_all*d_all - b_all*c_all

    diff = obs_pm_spur - mean_pm_spur
    diff_all = obs_pm_all - mean_pm_spur
    numer = (
        d * diff[:, 0] ** 2 
        + a * diff[:, 1] ** 2
        - (b + c) * diff[:, 0] * diff[:, 1]
    )
    numer_all = (
        d_all * diff_all[:, 0] ** 2 
        + a_all * diff_all[:, 1] ** 2
        - (b_all + c_all) * diff_all[:, 0] * diff_all[:, 1]
    )
    quad, quad_all = numer / det, numer_all / det_all

    loglike_fg_spur_i_pm = -0.5 * (quad + tt.log(det) + 2 * tt.log(2*np.pi))
    loglike_fg_spur_i_pm = loglike_fg_spur_i_pm.reshape(loglike_fg_spur_i_phi2.shape)
    loglike_fg_spur_pm = tt.concatenate([left1_phi2, loglike_fg_spur_i_pm, right1_phi2])
    
    loglike_fg_spur = loglike_fg_spur_pm + loglike_fg_spur_phi2
    
    return loglike_fg_spur
   

def binned_pm_model(model, obs_pm, obs_pm_cov):
    with model:
        mean_pm_stream = pm.Uniform('mean_pm_stream', lower=[-20, -10], upper=[0, 0], shape=2)
        ln_std_pm_stream = pm.Uniform('ln_std_pm_stream', -5, 0)
        std_pm_stream = pm.Deterministic('std_pm_stream', tt.exp(ln_std_pm_stream))
        cov_pm_stream = std_pm_stream**2 * tt.eye(2)
        full_cov = obs_pm_cov + cov_pm_stream

        # Assuming cov symmetric
        a = full_cov[:, 0, 0]
        b = c = full_cov[:, 0, 1]
        d = full_cov[:, 1, 1]
        det = a * d - b * c

        diff = obs_pm - mean_pm_stream[None, :]
        numer = (
            d * diff[:, 0] ** 2 
            + a * diff[:, 1] ** 2
            - (b + c) * diff[:, 0] * diff[:, 1]
        )
        quad = numer / det
        loglike_fg = -0.5 * (quad + tt.log(det) + 2 * tt.log(2*np.pi))
    return loglike_fg
 
def pm_model(model, obs_pm, obs_pm_cov, phi1_stream_all):
    with pm.Model() as model:
        
        #take the data to be used in the model (after color cut) and separate it
        bkg_ind = searchsorted(after.phi1, phi1_stream_all)
        obs_pm = obs_pm_all[bkg_ind]
        obs_pm_cov = obs_pm_cov_all[bkg_ind]
        phi1_stream = phi1_stream_all[bkg_ind]

        a1_log = pm.Uniform('pm_quadratic_log', lower = [-6.5,-9], upper = [-5.8,-8.5], shape=2)
        a1 = pm.Deterministic('pm_quadratic', tt.exp(a1_log))

        b1_log = pm.Uniform('pm_linear_log', lower = [-2,-3.8], upper=[-1,-3], shape=2)
        b1 = pm.Deterministic('pm_linear', tt.exp(b1_log))

        c1 = pm.Uniform('pm_constant', lower=[-9,-2.5], upper=[-8,-2], shape=2)

        mean_pm_stream = pm.Deterministic('mean_pm_stream', a1*phi1_stream**2 + b1*phi1_stream + c1)
        mean_pm_stream_all = pm.Deterministic('mean_pm_stream_all', a1*phi1_stream_all**2 + b1*phi1_stream_all + c1)

        ln_std_pm_stream = pm.Uniform('ln_std_pm_stream', -5, 0)
        std_pm_stream = pm.Deterministic('std_pm_stream', tt.exp(ln_std_pm_stream))
        cov_pm_stream = std_pm_stream**2 * tt.eye(2)
        full_cov = obs_pm_cov + cov_pm_stream
        full_cov_all = obs_pm_cov_all + cov_pm_stream
        
        # Assuming cov symmetric
        # this should all stay the same
        a, a_all = full_cov[:, 0, 0], full_cov_all[:, 0, 0]
        b = c = full_cov[:, 0, 1]
        b_all = c_all = full_cov_all[:, 0, 1]
        d, d_all = full_cov[:, 1, 1], full_cov_all[:, 1, 1]
        det, det_all = a * d - b * c, a_all*d_all - b_all*c_all
        
        # this also has to become a function of phi1 for each phi1 associated with each obs_pm row
        #  in order to 
        diff = obs_pm - mean_pm_stream
        diff_all = obs_pm_all - mean_pm_stream_all
        numer = (
            d * diff[:, 0] ** 2 
            + a * diff[:, 1] ** 2
            - (b + c) * diff[:, 0] * diff[:, 1]
        )
        numer_all = (
            d_all * diff_all[:, 0] ** 2 
            + a_all * diff_all[:, 1] ** 2
            - (b_all + c_all) * diff_all[:, 0] * diff_all[:, 1]
        )
        quad, quad_all = numer / det, numer_all / det_all
        loglike_fg = -0.5 * (quad + tt.log(det) + 2 * tt.log(2*np.pi))
        loglike_fg_all = -0.5 * (quad_all + tt.log(det_all) + 2 * tt.log(2*np.pi))
        
    return loglike_fg, loglike_fg_all

def phi2_model(model, phi1_stream, phi2_stream):
    with model:
        #NEW
        # phi2 track as a quadratic function of phi1
        a3_log = pm.Uniform('phi2_quadratic_log', lower=-8, upper=-5, shape=1)
        a3 = pm.Deterministic('phi2_quadratic', -tt.exp(a3_log)) # negative because we know if has to be

        b3_log = pm.Uniform('phi2_linear_log', lower=-3, upper=-1, shape=1)
        b3 = pm.Deterministic('phi2_linear', -tt.exp(b3_log)) # we also know this is negative

        c3 = pm.Uniform('phi2_constant', lower=-2, upper=-1, shape=1)

        mean_phi2_stream = pm.Deterministic('mean_phi2_stream', a3*phi1_stream**2 + b3*phi1_stream + c3)


        # add a component that gets the width of the stream as a function of phi1
        #  maybe a quadratic is notthe best way to model this
        ln_std_pm_stream = pm.Uniform('ln_std_phi2_stream', lower=-5, upper = 0)
        std_phi2_stream = pm.Deterministic('std_phi2_stream', tt.exp(ln_std_pm_stream))
        var_phi2_stream = pm.Deterministic('var_phi2_stream', std_phi2_stream**2)

        #NEW
        diff_phi2 = phi2_stream - mean_phi2_stream
        loglike_fg_phi2 = -0.5 * (tt.log(var_phi2_stream) + ((diff_phi2**2)/var_phi2_stream) + tt.log(2*np.pi))
        loglike_fg_phi2 = loglike_fg_phi2.reshape(loglike_fg_pm.shape)
        
    return loglike_fg_phi2


    
    
def plot_pm_memb_prob(obs_pm, post_member_prob):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 5))
    colorbar_plot = ax1.scatter(obs_pm[:, 0], obs_pm[:, 1], c=post_member_prob, s=0.1, alpha=0.2, cmap='cool')
    plt.colorbar(colorbar_plot,ax=ax1)
    #fig.colorbar(ax=ax1)
    ax1.set_xlim(-16, 2)
    ax1.set_ylim(-10, 3)
    ax1.set_xlabel(r'$\mu_1$ [{:latex_inline}]'.format(u.mas/u.yr))
    ax1.set_ylabel(r'$\mu_2$ [{:latex_inline}]'.format(u.mas/u.yr))
    ax1.set_xlabel(r'$\mu_1$ [{:latex_inline}]'.format(u.mas/u.yr))
    ax1.set_title(r'data')

    ax2.scatter(obs_pm[:, 0], obs_pm[:, 1], s=0.1, alpha=0.2)
    ax2.set_xlim(-16, 2)
    ax2.set_ylim(-10, 3)
    ax2.set_xlabel(r'$\mu_1$ [{:latex_inline}]'.format(u.mas/u.yr))
    ax2.set_ylabel(r'$\mu_2$ [{:latex_inline}]'.format(u.mas/u.yr))
    ax2.set_xlabel(r'$\mu_1$ [{:latex_inline}]'.format(u.mas/u.yr))
    ax2.set_title(r'data')
    plt.show()