import numpy as np
sys.path.append('../')
from gd1_helpers.membership.gd1_model import (
    Base,
    BackgroundModel,
    StreamDensModel,
    OffTrackModel,
)

def bkg_initialization(bkg_data):

    bkg_init_p = {
        "ln_N": np.log(len(bkg_data['phi1'])),
        #"phi1": {'zs': np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])+1},
        "phi1": {'zs': np.zeros(BackgroundModel.phi1_locs.shape[0]-1)},
        "phi2": {},
        "pm1": {
            "w": np.full_like(BackgroundModel.pm1_knots, 0.5),
            "mean1": np.full_like(BackgroundModel.pm1_knots, 0),
            "ln_std1": np.full_like(BackgroundModel.pm1_knots, 1),
            "mean2": np.full_like(BackgroundModel.pm1_knots, 5),
            "ln_std2": np.full_like(BackgroundModel.pm1_knots, 2)
        },
        "pm2": {
            "w": np.full_like(BackgroundModel.pm2_knots, 0.5),
            "mean1": np.full_like(BackgroundModel.pm2_knots, -2.),
            "ln_std1": np.full_like(BackgroundModel.pm2_knots, 1),
            "mean2": np.full_like(BackgroundModel.pm2_knots, -3),
            "ln_std2": np.full_like(BackgroundModel.pm2_knots, 2)
        },
    }

    return bkg_init_p

def stream_initialization(stream_data, p):

    _phi2_stat = binned_statistic(stream_data["phi1"], stream_data["phi2"], bins=np.linspace(-90, 10, 21))
    _phi2_interp = InterpolatedUnivariateSpline(
        0.5 * (_phi2_stat.bin_edges[:-1] + _phi2_stat.bin_edges[1:]), _phi2_stat.statistic
    )

    _pm1_stat = binned_statistic(stream_data["phi1"], stream_data["pm1"], bins=np.linspace(-80, 0, 32))
    _pm1_interp = InterpolatedUnivariateSpline(
        0.5 * (_pm1_stat.bin_edges[:-1] + _pm1_stat.bin_edges[1:]), _pm1_stat.statistic, ext=3
    )

    _pm2_stat = binned_statistic(stream_data["phi1"], stream_data["pm2"], bins=np.linspace(-80, 0, 32))
    _pm2_interp = InterpolatedUnivariateSpline(
        0.5 * (_pm2_stat.bin_edges[:-1] + _pm2_stat.bin_edges[1:]), _pm2_stat.statistic, ext=3
    )

    _pm1_interp=InterpolatedUnivariateSpline(p.track.track.transform_to(p.track.stream_frame).phi1,
                                             p.track.track.transform_to(p.track.stream_frame).pm_phi1_cosphi2,
                                             ext=3)
    _pm2_interp=InterpolatedUnivariateSpline(p.track.track.transform_to(p.track.stream_frame).phi1,
                                             p.track.track.transform_to(p.track.stream_frame).pm_phi2,
                                             ext=3)

    stream_init_p = {
        "ln_N": np.log(len(stream_data['phi1'])),
        "phi1": {
            "zs": np.zeros(StreamDensModel.phi1_locs.shape[0]-1)
        },
        "phi2": {
            "mean": _phi2_interp(StreamDensModel.phi2_knots),
            "ln_std": np.full_like(StreamDensModel.phi2_knots, -0.5)
        },
        "pm1": {
            "mean": _pm1_interp(StreamDensModel.pm1_knots),
            "ln_std": np.full_like(StreamDensModel.pm1_knots, -0.5)
        },
        "pm2": {
            "mean": _pm2_interp(StreamDensModel.pm2_knots),
            "ln_std": np.full_like(StreamDensModel.pm2_knots, -0.5)
        }
    }

    return stream_init_p


def offtrack_initialization():

    offtrack_init_p = {
        "ln_N": np.log(100),
        ("phi1", "phi2"): {
            "zs": np.zeros(OffTrackModel.phi12_locs.shape[0] - 1)
        },
        "pm1": stream_opt_pars["pm1"].copy(),
        "pm2": stream_opt_pars["pm2"].copy()
    }

    return offtrack_init_p
