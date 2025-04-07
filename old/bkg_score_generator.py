# MUST RUN THIS IN BASIC ENVIRONMENT, NOT PYMC3

import pathlib
import warnings
warnings.filterwarnings('ignore')
import os

import sys
sys.path.append('../')

# Third-party
from astropy.table import Table, vstack
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.lib.recfunctions import stack_arrays
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, interp1d
from scipy.ndimage.filters import gaussian_filter

import gala.coordinates as gc
from pyia import GaiaData
from scipy.stats import binned_statistic

from tqdm import trange
from xdgmm import XDGMM

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

track = np.load('../data/gd1_track.npy')
phi1_ = np.load('../data/phi1_stream_from_pm_model.npy')

phi2_spline = UnivariateSpline(phi1_, track, k=5)
phi1_trace = np.linspace(-100, 20, 30)
phi2_trace = phi2_spline()


stream_top = np.vstack([phi1_trace, phi2_trace + 2]).T
stream_bottom = np.vstack([phi1_trace, phi2_trace - 2]).T
gd1_poly = np.vstack((stream_top, stream_bottom[::-1]))
gd1_phi_path = mpl.path.Path(gd1_poly)

offstream_above_top = np.vstack([phi1_trace, phi2_trace + 3]).T
offstream_above_bottom = np.vstack([phi1_trace, phi2_trace + 2]).T
ctl_poly1 = np.vstack((offstream_above_top, offstream_above_bottom[::-1]))

offstream_below_top =np.vstack([phi1_trace,  phi2_trace-2]).T
offstream_below_bottom = np.vstack([phi1_trace, phi2_trace-3]).T
ctl_poly2 = np.vstack((offstream_below_top, offstream_below_bottom[::-1]))

ctl_phi_path = [mpl.path.Path(ctl_poly1),
                mpl.path.Path(ctl_poly2)]

dist = gaia.get_distance(min_parallax=1e-3*u.mas)
c = gaia.get_skycoord(distance=dist)
stream_coord = c.transform_to(gc.GD1)
phi1 = stream_coord.phi1.degree
phi2 = stream_coord.phi2.degree
pm1 = stream_coord.pm_phi1_cosphi2
pm2 = stream_coord.pm_phi2

g = gaia[np.isfinite(gaia.parallax) & (gaia.parallax > 0) & np.isfinite(pm2)]
dist = g.get_distance(min_parallax=1e-3*u.mas)
c1 = g.get_skycoord(distance=dist)
stream_coord = c1.transform_to(gc.GD1)
phi1 = stream_coord.phi1.degree
phi2 = stream_coord.phi2.degree
pm1 = stream_coord.pm_phi1_cosphi2
pm2 = stream_coord.pm_phi2

X = np.stack((phi1_color, phi2_color)).T
X_pm = np.stack((pm1_color.value, pm2_color.value)).T

c = g.get_skycoord(distance=False)

Cov = g.get_cov()[:, 3:5, 3:5]
Cov_pm = gc.transform_pm_cov(c, Cov, gc.GD1)

sky_gd1_mask = gd1_phi_path.contains_points(X)
sky_ctl_mask = ctl_phi_path[0].contains_points(X) | ctl_phi_path[1].contains_points(X)


phi1_bins = np.arange(-100, 20+1e-3, 10)
pm_bins = np.arange(-20, 10+1e-3, 0.5)

gmms = []
all_tbl = Table()
for i in trange(len(phi1_bins)-1):
    l, r = phi1_bins[i], phi1_bins[i+1]
    phi1_mask = (phi1 >= l) & (phi1 < r)
    g_new = g[phi1_mask]
    control_mask = phi1_mask & sky_ctl_mask

    X_pm_ctl = X_pm[control_mask]
    C_pm_ctl = Cov_pm[control_mask]

    gmm = XDGMM(n_components=8, method='Bovy')
    _ = gmm.fit(X_pm_ctl, C_pm_ctl)
    scores, _ = gmm.score_samples(X_pm[phi1_mask], Cov_pm[phi1_mask])

    gmms.append(gmm)

    # ---
    tbl = Table()
    tbl['phi1'] = g_new['phi1']
    tbl['phi2'] = g_new['phi2']
    tbl['pm1'] = g_new['pm_phi1_cosphi2']
    tbl['pm2'] = g_new['pm_phi2']
    tbl['g_0'] = g_new['g_0']
    tbl['i_0'] = g_new['i_0']
    tbl['pm_mask'] = g_new['pm_mask']
    tbl['gi_cmd_mask'] = g_new['gi_cmd_mask']
    tbl['pm_ln_bkg_prob'] = scores
    all_tbl = vstack([all_tbl, tbl])

sorted_table = all_tbl[all_tbl['phi1'].argsort()]
sorted_table.write('../data/sorted_pm_member_prob_all_stars_8comp.fits', overwrite=True)
