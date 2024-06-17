import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from stream_membership import StreamModel
from stream_membership.utils import get_grid
from stream_membership.variables import (
    GridGMMVariable,
    Normal1DSplineMixtureVariable,
    Normal1DSplineVariable,
    UniformVariable,
    Normal1DVariable
)

#phi1_lim = (-100, 20)

class Base:
    phi1_lim = (-20, 20)
    coord_bounds = {"phi1": phi1_lim, "phi2": (-7,7), "pm1": (-20,20), "pm2": (-20,20)}

    @classmethod
    def setup(cls, pawprint, data):
        cls.pawprint, cls.data = pawprint, data

        coord_names = ("phi2", "pm1", "pm2")

        cls.phi1_lim = (np.min(data['phi1']), np.max(data['phi1']))
        phi2_lim = (np.min(data['phi2']), np.max(data['phi2']))
        pm1_lim = (np.min(pawprint.pmprint.vertices[:,0]), np.max(pawprint.pmprint.vertices[:,0]))
        pm2_lim = (np.min(pawprint.pmprint.vertices[:,1]), np.max(pawprint.pmprint.vertices[:,1]))

        cls.coord_bounds = {"phi1": cls.phi1_lim, "phi2": phi2_lim, "pm1": pm1_lim, "pm2": pm2_lim}

        cls.default_grids = {
            "phi1": np.arange(*cls.coord_bounds["phi1"], 0.2),
            "phi2": np.arange(*cls.coord_bounds["phi2"], 0.1),
            "pm1": np.arange(*cls.coord_bounds["pm1"], 0.1),
            "pm2": np.arange(*cls.coord_bounds["pm2"], 0.1),
        }
        return cls.phi1_lim, cls.coord_bounds, cls.default_grids


class BackgroundModel(Base, StreamModel):
    name = "background"

    ln_N_dist = dist.Uniform(-10, 15)

    phi1_locs = get_grid(*Base.phi1_lim, 10.0, pad_num=1).reshape(-1, 1)  # every 10º
    pm1_knots = get_grid(*Base.coord_bounds["phi1"], 10.0) # changed from 10 to 15 because of small scale features in pm2
    pm2_knots = get_grid(*Base.coord_bounds["phi1"], 15.0)

    variables = {
        "phi1": GridGMMVariable(
            param_priors={
                "zs": dist.Uniform(-8.0, 8.0).expand((phi1_locs.shape[0] - 1,)),
            },
            locs=phi1_locs,
            scales=np.full_like(phi1_locs, 10.0),
            coord_bounds=Base.phi1_lim,
        ),
        "phi2": UniformVariable(
            param_priors={}, coord_bounds=Base.coord_bounds["phi2"]
        ),
        "pm1": Normal1DSplineMixtureVariable(
            param_priors={
                "w": dist.Uniform(0, 1).expand((pm1_knots.size,)),
                "mean1": dist.Uniform(-20, 20).expand((pm1_knots.size,)),
                "mean2": dist.Uniform(-20, 20).expand((pm1_knots.size,)),
                "ln_std1": dist.Uniform(-2, 3).expand((pm1_knots.size,)),
                "ln_std2": dist.Uniform(-2, 3).expand((pm1_knots.size,)),
            },
            knots=pm1_knots,
            spline_ks={"w": 1},
            coord_bounds=Base.coord_bounds.get("pm1"),
        ),
        "pm2": Normal1DSplineMixtureVariable(
            param_priors={
                "w": dist.Uniform(0, 1).expand((pm2_knots.size,)), # weight between two components
                "mean1": dist.Uniform(-20, 20).expand((pm2_knots.size,)),
                "mean2": dist.Uniform(-20, 20).expand((pm2_knots.size,)),
                "ln_std1": dist.Uniform(-2, 3).expand((pm2_knots.size,)),
                "ln_std2": dist.Uniform(-2, 3).expand((pm2_knots.size,)),
            },
            knots=pm2_knots,
            spline_ks={"w": 1},
            coord_bounds=Base.coord_bounds.get("pm2"),
        ),
    }

    data_required = {
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }

    @classmethod
    def bkg_update(cls, pawprint, data, knot_sep):
        cls.phi1_lim, cls.coord_bounds, cls.default_grids = Base.setup(pawprint, data)

        cls.phi1_locs = get_grid(*cls.phi1_lim, knot_sep, pad_num=1).reshape(-1, 1)
        cls.pm1_knots = get_grid(*cls.coord_bounds["phi1"], knot_sep)
        cls.pm2_knots = get_grid(*cls.coord_bounds["phi1"], knot_sep)

        cls.variables = {
            "phi1": GridGMMVariable(
                param_priors={
                    "zs": dist.Uniform(-8.0, 8.0).expand((cls.phi1_locs.shape[0] - 1,)),
                },
                locs=cls.phi1_locs,
                scales=np.full_like(cls.phi1_locs, 10.0),
                coord_bounds=cls.phi1_lim,
            ),
            "phi2": UniformVariable(
                param_priors={}, coord_bounds=cls.coord_bounds["phi2"]
            ),
            "pm1": Normal1DSplineMixtureVariable(
                param_priors={
                    "w": dist.Uniform(0, 1).expand((cls.pm1_knots.size,)),
                    "mean1": dist.Uniform(-20, 20).expand((cls.pm1_knots.size,)),
                    "mean2": dist.Uniform(-20, 20).expand((cls.pm1_knots.size,)),
                    "ln_std1": dist.Uniform(-2, 3).expand((cls.pm1_knots.size,)),
                    "ln_std2": dist.Uniform(-2, 3).expand((cls.pm1_knots.size,)),
                },
                knots=cls.pm1_knots,
                spline_ks={"w": 1},
                coord_bounds=cls.coord_bounds.get("pm1"),
            ),
            "pm2": Normal1DSplineMixtureVariable(
                param_priors={
                    "w": dist.Uniform(0, 1).expand((cls.pm2_knots.size,)),
                    "mean1": dist.Uniform(-20, 20).expand((cls.pm2_knots.size,)),
                    "mean2": dist.Uniform(-20, 20).expand((cls.pm2_knots.size,)),
                    "ln_std1": dist.Uniform(-2, 3).expand((cls.pm2_knots.size,)),
                    "ln_std2": dist.Uniform(-2, 3).expand((cls.pm2_knots.size,)),
                },
                knots=cls.pm2_knots,
                spline_ks={"w": 1},
                coord_bounds=cls.coord_bounds.get("pm2"),
            ),
        }

        cls.data_required = {
            "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
            "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
        }

    @classmethod
    def bkg_update_pal5(cls, pawprint, data, knot_sep):
        cls.phi1_lim, cls.coord_bounds, cls.default_grids = Base.setup(pawprint, data)

        cls.phi1_locs = get_grid(*cls.phi1_lim, knot_sep, pad_num=1).reshape(-1, 1)
        cls.pm1_knots = get_grid(*cls.coord_bounds["phi1"], knot_sep)
        cls.pm2_knots = get_grid(*cls.coord_bounds["phi1"], knot_sep)

        cls.phi2_knots = get_grid(*cls.phi1_lim, knot_sep)  # knots every 10º

        cls.variables = {
            "phi1": GridGMMVariable(
                param_priors={
                    "zs": dist.Uniform(-8.0, 8.0).expand((cls.phi1_locs.shape[0] - 1,)),
                },
                locs=cls.phi1_locs,
                scales=np.full_like(cls.phi1_locs, 10.0),
                coord_bounds=cls.phi1_lim,
            ),
            "phi2": Normal1DSplineVariable(
                param_priors={
                    "mean": dist.Uniform(-20,20).expand(cls.phi2_knots.shape),
                    "ln_std": dist.Uniform(0.5,5).expand(cls.phi2_knots.shape),
                },
                knots=cls.phi2_knots,
                # spline_ks = {"w":1},
                coord_bounds=cls.coord_bounds["phi2"],
            ),
            # "phi2": Normal1DSplineMixtureVariable(
            #     param_priors={
            #         "w": dist.Uniform(0, 1).expand((cls.pm1_knots.size,)),
            #         "mean1": dist.Uniform(5,20).expand(cls.phi2_knots.shape),
            #         "mean2": dist.Uniform(-20,-5).expand(cls.phi2_knots.shape),
            #         "ln_std1": dist.Uniform(2,5).expand(cls.phi2_knots.shape),
            #         "ln_std2": dist.Uniform(2,5).expand(cls.phi2_knots.shape),
            #     },
            #     knots=cls.phi2_knots,
            #     spline_ks = {"w":1},
            #     coord_bounds=cls.coord_bounds["phi2"],
            # ),
            "pm1": Normal1DSplineMixtureVariable(
                param_priors={
                    "w": dist.Uniform(0, 1).expand((cls.pm1_knots.size,)),
                    "mean1": dist.Uniform(-20, 20).expand((cls.pm1_knots.size,)),
                    "mean2": dist.Uniform(-20, 20).expand((cls.pm1_knots.size,)),
                    "ln_std1": dist.Uniform(-2, 3).expand((cls.pm1_knots.size,)),
                    "ln_std2": dist.Uniform(-2, 3).expand((cls.pm1_knots.size,)),
                },
                knots=cls.pm1_knots,
                spline_ks={"w": 1},
                coord_bounds=cls.coord_bounds.get("pm1"),
            ),
            "pm2": Normal1DSplineMixtureVariable(
                param_priors={
                    "w": dist.Uniform(0, 1).expand((cls.pm2_knots.size,)),
                    "mean1": dist.Uniform(-20, 20).expand((cls.pm2_knots.size,)),
                    "mean2": dist.Uniform(-20, 20).expand((cls.pm2_knots.size,)),
                    "ln_std1": dist.Uniform(-2, 3).expand((cls.pm2_knots.size,)),
                    "ln_std2": dist.Uniform(-2, 3).expand((cls.pm2_knots.size,)),
                },
                knots=cls.pm2_knots,
                spline_ks={"w": 1},
                coord_bounds=cls.coord_bounds.get("pm2"),
            ),
        }

        cls.data_required = {
            "phi2": {"x": "phi1", "y": "phi2"},
            "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
            "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
        }
        cls._data_required['phi2'] = {"x": "phi1", "y": "phi2"}


class StreamDensModel(Base, StreamModel):
    name = "stream"

    ln_N_dist = dist.Uniform(5, 15)

    phi1_dens_step = 4.0  # knots every 4º
    phi1_locs = get_grid(*Base.phi1_lim, phi1_dens_step, pad_num=1).reshape(-1, 1)

    phi2_knots = get_grid(*Base.phi1_lim, 10.0)  # knots every 10º

    pm1_knots = get_grid(*Base.phi1_lim, 10.0)  # knots every 110º
    pm2_knots = get_grid(*Base.phi1_lim, 10.0)  # knots every 10º

    variables = {
        "phi1": GridGMMVariable(
            param_priors={
                "zs": dist.Uniform(
                    jnp.full(phi1_locs.shape[0] - 1, -8),
                    jnp.full(phi1_locs.shape[0] - 1, 8),
                )
            },
            locs=phi1_locs,
            scales=np.full_like(phi1_locs, phi1_dens_step),
            coord_bounds=Base.phi1_lim,
        ),
        "phi2": Normal1DSplineVariable(
            param_priors={
                "mean": dist.Uniform(
                    jnp.full_like(phi2_knots, -4.0), jnp.full_like(phi2_knots, 1.0)
                ),
                "ln_std": dist.Uniform(
                    jnp.full_like(phi2_knots, -2.0), jnp.full_like(phi2_knots, 0.5)
                ),
            },
            knots=phi2_knots,
            coord_bounds=Base.coord_bounds["phi2"],
        ),
        "pm1": Normal1DSplineVariable(
            param_priors={
                "mean": dist.Uniform(*Base.coord_bounds.get("pm1")).expand(
                    pm1_knots.shape
                ),
                "ln_std": dist.Uniform(-5, 0).expand(pm1_knots.shape),  # ~20 km/s
            },
            knots=pm1_knots,
            coord_bounds=Base.coord_bounds.get("pm1"),
        ),
        "pm2": Normal1DSplineVariable(
            param_priors={
                "mean": dist.Uniform(-10, 10).expand(pm2_knots.shape),
                "ln_std": dist.Uniform(-5, 0).expand(pm2_knots.shape),  # ~20 km/s
            },
            knots=pm2_knots,
            coord_bounds=Base.coord_bounds.get("pm2"),
        ),
    }
    data_required = {
        "phi2": {"x": "phi1", "y": "phi2"},
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }

    def extra_ln_prior(self, params):
        lp = 0.0

        std_map = {"mean": 0.5, "ln_std": 0.25}
        for var_name, var in self.variables.items():
            if hasattr(var, "splines"):
                for par_name, spl_y in params[var_name].items():
                    if par_name in std_map:
                        lp += (
                            dist.Normal(0, std_map[par_name])
                            .log_prob(spl_y[1:] - spl_y[:-1])
                            .sum()
                        )

        return lp

    @classmethod
    def stream_dens_update(cls, pawprint, data, knot_sep):
        cls.phi1_lim, cls.coord_bounds, cls.default_grids = Base.setup(pawprint, data)

        # cls.phi1_lim =(np.min(pawprint.skyprint['stream'].vertices[:,0]), np.max(pawprint.skyprint['stream'].vertices[:,0]))
        cls.phi1_locs = get_grid(*cls.phi1_lim, cls.phi1_dens_step, pad_num=1).reshape(-1, 1)

        cls.phi2_knots = get_grid(*cls.phi1_lim, knot_sep)  # knots every 10º

        cls.pm1_knots = get_grid(*cls.phi1_lim, knot_sep)  # knots every 10º
        cls.pm2_knots = get_grid(*cls.phi1_lim, knot_sep)  # knots every 10º

        cls.variables = {
            "phi1": GridGMMVariable(
                param_priors={
                    "zs": dist.Uniform(
                        jnp.full(cls.phi1_locs.shape[0] - 1, -8),
                        jnp.full(cls.phi1_locs.shape[0] - 1, 8),
                    )
                },
                locs=cls.phi1_locs,
                scales=np.full_like(cls.phi1_locs, cls.phi1_dens_step),
                coord_bounds=cls.phi1_lim,
            ),
            "phi2": Normal1DSplineVariable(
                param_priors={
                    "mean": dist.Uniform(
                        jnp.full_like(cls.phi2_knots, -4.0), jnp.full_like(cls.phi2_knots, 1.0)
                    ),
                    "ln_std": dist.Uniform(
                        jnp.full_like(cls.phi2_knots, -2.0), jnp.full_like(cls.phi2_knots, 0.5)
                    ),
                },
                knots=cls.phi2_knots,
                coord_bounds=cls.coord_bounds["phi2"],
            ),
            "pm1": Normal1DSplineVariable(
                param_priors={
                    "mean": dist.Uniform(*cls.coord_bounds.get("pm1")).expand(
                        cls.pm1_knots.shape
                    ),
                    "ln_std": dist.Uniform(-5, 0).expand(cls.pm1_knots.shape),  # ~20 km/s
                },
                knots=cls.pm1_knots,
                coord_bounds=cls.coord_bounds.get("pm1"),
            ),
            "pm2": Normal1DSplineVariable(
                param_priors={
                    "mean": dist.Uniform(*cls.coord_bounds.get("pm2")).expand(cls.pm2_knots.shape),
                    "ln_std": dist.Uniform(-5, 0).expand(cls.pm2_knots.shape),  # ~20 km/s
                },
                knots=cls.pm2_knots,
                coord_bounds=cls.coord_bounds.get("pm2"),
            ),
        }
        cls.data_required = {
            "phi2": {"x": "phi1", "y": "phi2"},
            "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
            "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
        }


class OffTrackModel(Base, StreamModel):
    name = "offtrack"

    ln_N_dist = dist.Uniform(-5, 10)

    dens_phi1_lim = (-100, 20)
    dens_phi2_lim = (-8, 3.5)

    dens_steps = np.array([3.0, 0.4]) # should find some optimal spacing here
    spar_steps = 5*dens_steps

    dens_locs = np.stack(
        np.meshgrid(
            np.arange(dens_phi1_lim[0], dens_phi1_lim[1] + 1e-3, dens_steps[0]),
            np.arange(dens_phi2_lim[0], dens_phi2_lim[1] + 1e-3, dens_steps[1]),
        )
    ).T.reshape(-1, 2)

    spar_locs = np.stack(
        np.meshgrid(
            get_grid(*Base.coord_bounds["phi1"], spar_steps[0], pad_num=1),
            get_grid(*Base.coord_bounds["phi2"], spar_steps[1], pad_num=1),
        )
    ).T.reshape(-1, 2)
    _mask = (
        (spar_locs[:, 0] >= dens_phi1_lim[0])
        & (spar_locs[:, 0] <= dens_phi1_lim[1])
        & (spar_locs[:, 1] >= dens_phi2_lim[0])
        & (spar_locs[:, 1] <= dens_phi2_lim[1])
    )
    spar_locs = spar_locs[~_mask]

    phi12_locs = np.concatenate((dens_locs, spar_locs))
    phi12_scales = np.concatenate(
        (np.full_like(dens_locs, dens_steps[0]), np.full_like(spar_locs, spar_steps[0]))
    )
    phi12_scales[: dens_locs.shape[0], 1] = dens_steps[1]
    phi12_scales[dens_locs.shape[0] :, 1] = spar_steps[1]

    variables = {
        ("phi1", "phi2"): GridGMMVariable(
            param_priors={
                "zs": dist.Uniform(-8.0, 8.0).expand((phi12_locs.shape[0] - 1,))
                #                 "zs": dist.TruncatedNormal(
                #                     loc=-8, scale=4.0, low=-8.0, high=8.0
                #                 ).expand((phi12_locs.shape[0] - 1,))
            },
            locs=phi12_locs,
            scales=phi12_scales,
            coord_bounds=(
                np.array(
                    [Base.coord_bounds["phi1"][0], Base.coord_bounds["phi2"][0]]
                ),
                np.array(
                    [Base.coord_bounds["phi1"][1], Base.coord_bounds["phi2"][1]]
                ),
            ),
        ),
        "pm1": StreamDensModel.variables["pm1"],
        "pm2": StreamDensModel.variables["pm2"],
    }

    data_required = {
        ("phi1", "phi2"): {"y": ("phi1", "phi2")},
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }

    @classmethod
    def offtrack_update(cls, pawprint, data, dens_steps):
        '''
        dens_steps: array - [phi1_dens_steps, phi2_dens_steps]
        '''

        cls.dens_phi1_lim, cls.coord_bounds, cls.default_grids = Base.setup(pawprint, data)
        cls.dens_phi2_lim = (np.min(data['phi2']), np.max(data['phi2']))

        cls.dens_locs = np.stack(
            np.meshgrid(
                np.arange(cls.dens_phi1_lim[0], cls.dens_phi1_lim[1] + 1e-3, dens_steps[0]),
                np.arange(cls.dens_phi2_lim[0], cls.dens_phi2_lim[1] + 1e-3, dens_steps[1]),
            )
        ).T.reshape(-1, 2)

        spar_steps = 5*dens_steps

        cls.spar_locs = np.stack(
            np.meshgrid(
                get_grid(*cls.coord_bounds["phi1"], spar_steps[0], pad_num=1),
                get_grid(*cls.coord_bounds["phi2"], spar_steps[1], pad_num=1),
            )
        ).T.reshape(-1, 2)
        _mask = (
            (cls.spar_locs[:, 0] >= cls.dens_phi1_lim[0])
            & (cls.spar_locs[:, 0] <= cls.dens_phi1_lim[1])
            & (cls.spar_locs[:, 1] >= cls.dens_phi2_lim[0])
            & (cls.spar_locs[:, 1] <= cls.dens_phi2_lim[1])
        )
        cls.spar_locs = cls.spar_locs[~_mask]

        cls.phi12_locs = np.concatenate((cls.dens_locs, cls.spar_locs))
        cls.phi12_scales = np.concatenate(
            (np.full_like(cls.dens_locs, cls.dens_steps[0]), np.full_like(cls.spar_locs, cls.spar_steps[0]))
        )
        cls.phi12_scales[: cls.dens_locs.shape[0], 1] = cls.dens_steps[1]
        cls.phi12_scales[cls.dens_locs.shape[0] :, 1] = cls.spar_steps[1]

        cls.variables = {
            ("phi1", "phi2"): GridGMMVariable(
                param_priors={
                    "zs": dist.Uniform(-8.0, 8.0).expand((cls.phi12_locs.shape[0] - 1,))
                    #                 "zs": dist.TruncatedNormal(
                    #                     loc=-8, scale=4.0, low=-8.0, high=8.0
                    #                 ).expand((phi12_locs.shape[0] - 1,))
                },
                locs=cls.phi12_locs,
                scales=cls.phi12_scales,
                coord_bounds=(
                    np.array(
                        [cls.coord_bounds["phi1"][0], cls.coord_bounds["phi2"][0]]
                    ),
                    np.array(
                        [cls.coord_bounds["phi1"][1], cls.coord_bounds["phi2"][1]]
                    ),
                ),
            ),
            "pm1": StreamDensModel.variables["pm1"],
            "pm2": StreamDensModel.variables["pm2"],
        }

        cls.data_required = {
            ("phi1", "phi2"): {"y": ("phi1", "phi2")},
            "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
            "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
        }
