import pathlib
import warnings
import warnings
warnings.filterwarnings('ignore')
import os

import sys
sys.path.append('code/')

# Third-party
import astropy.coordinates as coord
import astropy.table as at
from astropy.io import fits
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from numpy.lib.recfunctions import stack_arrays
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, interp1d
import pandas as pd

import gala.coordinates as gc
import gala.dynamics as gd
from pyia import GaiaData
from astroquery.gaia import Gaia
from astropy.table import Table
import read_mist_models



# need the crossmatched catalog and the isochrones

gaia = GaiaData('data/gd1_gaia_ps1_all.fits')
gaia = gaia[np.isfinite(gaia.parallax) & (gaia.parallax > 0)]
dist = gaia.get_distance(min_parallax=1e-3*u.mas)
c = gaia.get_skycoord(distance=dist)
stream_coord = c.transform_to(gc.GD1)
pm2_0 = stream_coord.pm_phi2
gaia = gaia[np.isfinite(pm2_0)]
dist = gaia.get_distance(min_parallax=1e-3*u.mas)
c = gaia.get_skycoord(distance=dist)
stream_coord = c.transform_to(gc.GD1)
phi1_0 = stream_coord.phi1.degree

sections = np.arange(-100,15,5)
#Load in MIST isochrone:
feh_sel, age_select = -1.7, 13
iso_file = 'data/isochrones/MIST_PS_iso_feh_{}_vrot0.iso.cmd'.format(feh_sel)
isocmd = read_mist_models.ISOCMD(iso_file)

age_sel = isocmd.age_index(age_select*10**9)
gmag = isocmd.isocmds[age_sel]['PS_g']
imag = isocmd.isocmds[age_sel]['PS_i']
gi_color = gmag-imag

dm = np.concatenate([[14.7, 14.6, 14.5, 14.45, 14.4, 14.35, 14.3, 14.3],
                     np.linspace(14.3, 14.6, 9),
                     [14.71, 14.75, 14.8, 15, 15.2, 15.4]])

print('Creating masks...')
all_final = at.Table()
for i in range(len(sections)):
    left = int(sections[i])
    right = left+10
    g = gaia[(phi1_0 < right) & (phi1_0 > left)]

    dist = g.get_distance(min_parallax=1e-3*u.mas)
    c = g.get_skycoord(distance=dist)
    stream_coord = c.transform_to(gc.GD1)
    phi1 = stream_coord.phi1.degree
    phi2 = stream_coord.phi2.degree
    pm1 = stream_coord.pm_phi1_cosphi2
    pm2 = stream_coord.pm_phi2

    pm_polygon = np.array([[-15, -5.4],
                       [-15, -1.5],
                       [-8, -0.5],
                       [-1.85, -0.9],
                       [-1.85, -3.8],
                       [-11, -5.4]])
    pp = mpl.patches.Polygon(pm_polygon,
                         facecolor='none',edgecolor='k', linewidth=2)
    pm_points = np.vstack((pm1, pm2)).T
    pm_mask = pp.get_path().contains_points(pm_points)

    width_changes = np.concatenate([0.6*np.linspace(1, 1/6, 150), 0.1*np.ones(70)])

    iso_contour = np.concatenate((np.vstack([gi_color[:220] + width_changes, gmag[:220]+dm[i] + 0.1]).T,
                                  np.flip(np.vstack([gi_color[:220] - width_changes, gmag[:220]+dm[i] - 0.1]).T, axis = 0)))

    cc_iso = mpl.patches.Polygon(iso_contour, facecolor='none',edgecolor='k', linewidth=1)

    cm_points_iso = np.vstack([g.g_0-g.i_0, g.g_0]).T
    cm_mask_iso = cc_iso.get_path().contains_points(cm_points_iso)

    final_phi1_mask = (phi1 > -100) & (phi1 < 20)

    final_t = g.data[final_phi1_mask]
    final_t['pm_mask'] = pm_mask[final_phi1_mask]
    final_t['gi_cmd_mask'] = cm_mask_iso[final_phi1_mask]

    final_t['phi1'] = phi1[final_phi1_mask]
    final_t['phi2'] = phi2[final_phi1_mask]

    final_t['pm_phi1_cosphi2'] = pm1[final_phi1_mask]
    final_t['pm_phi2'] = pm2[final_phi1_mask]

    all_final = at.vstack([all_final, final_t])

print('Making the file...')
arr, ind = np.unique(all_final['phi1'], return_index=True)
all_final = all_final[ind]
all_final.write('data/gd1_ps1_with_basic_masks_thin.fits', overwrite=True)
