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
)

phi1_lim = (-100, 20)


class GD1Base:
    coord_names = ("phi2", "pm1", "pm2")
    coord_bounds = {"phi1": phi1_lim, "phi2": (-7, 5), "pm1": (-15, -1.0)}

    default_grids = {
        "phi1": np.arange(*coord_bounds["phi1"], 0.2),
        "phi2": np.arange(*coord_bounds["phi2"], 0.1),
        "pm1": np.arange(*coord_bounds["pm1"], 0.1),
        "pm2": np.arange(-10, 10 + 1e-3, 0.1),
    }


class GD1BackgroundModel(GD1Base, StreamModel):
    name = "background"

    ln_N_dist = dist.Uniform(-10, 15)

    phi1_locs = get_grid(*phi1_lim, 20.0, pad_num=1).reshape(-1, 1)  # every 20º
    pm1_knots = get_grid(*GD1Base.coord_bounds["phi1"], 10.0)
    pm2_knots = get_grid(*GD1Base.coord_bounds["phi1"], 20.0)

    variables = {
        "phi1": GridGMMVariable(
            param_priors={
                "zs": dist.Uniform(-8.0, 8.0).expand((phi1_locs.shape[0] - 1,)),
            },
            locs=phi1_locs,
            scales=np.full_like(phi1_locs, 20.0),
            coord_bounds=phi1_lim,
        ),
        "phi2": UniformVariable(
            param_priors={}, coord_bounds=GD1Base.coord_bounds["phi2"]
        ),
        "pm1": Normal1DSplineMixtureVariable(
            param_priors={
                "w": dist.Uniform(0, 1).expand((pm1_knots.size,)),
                "mean1": dist.Uniform(-2, 20).expand((pm1_knots.size,)),
                "mean2": dist.Uniform(-2, 20).expand((pm1_knots.size,)),
                "ln_std1": dist.Uniform(-5, 5).expand((pm1_knots.size,)),
                "ln_std2": dist.Uniform(-5, 5).expand((pm1_knots.size,)),
            },
            knots=pm1_knots,
            spline_ks={"w": 1},
            coord_bounds=GD1Base.coord_bounds.get("pm1"),
        ),
        "pm2": Normal1DSplineMixtureVariable(
            param_priors={
                "w": dist.Uniform(0, 1).expand((pm2_knots.size,)),
                "mean1": dist.Uniform(-5, 5).expand((pm2_knots.size,)),
                "mean2": dist.Uniform(-5, 5).expand((pm2_knots.size,)),
                "ln_std1": dist.Uniform(-5, 5).expand((pm2_knots.size,)),
                "ln_std2": dist.Uniform(-5, 5).expand((pm2_knots.size,)),
            },
            knots=pm2_knots,
            spline_ks={"w": 1},
            coord_bounds=GD1Base.coord_bounds.get("pm2"),
        ),
    }

    data_required = {
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }


class GD1StreamModel(GD1Base, StreamModel):
    name = "stream"

    ln_N_dist = dist.Uniform(5, 15)

    phi1_dens_step = 4.0  # knots every 4º
    phi1_locs = get_grid(*phi1_lim, phi1_dens_step, pad_num=1).reshape(-1, 1)

    phi2_knots = get_grid(*phi1_lim, 8.0)  # knots every 8º

    pm1_knots = get_grid(*phi1_lim, 15.0)  # knots every 15º
    pm2_knots = get_grid(*phi1_lim, 25.0)  # knots every 25º

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
            coord_bounds=phi1_lim,
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
            coord_bounds=GD1Base.coord_bounds["phi2"],
        ),
        "pm1": Normal1DSplineVariable(
            param_priors={
                "mean": dist.Uniform(*GD1Base.coord_bounds.get("pm1")).expand(
                    pm1_knots.shape
                ),
                "ln_std": dist.Uniform(-5, -0.75).expand(pm1_knots.shape),  # ~20 km/s
            },
            knots=pm1_knots,
            coord_bounds=GD1Base.coord_bounds.get("pm1"),
        ),
        "pm2": Normal1DSplineVariable(
            param_priors={
                "mean": dist.Uniform(-5, 5).expand(pm2_knots.shape),
                "ln_std": dist.Uniform(-5, -0.75).expand(pm2_knots.shape),  # ~20 km/s
            },
            knots=pm2_knots,
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


class GD1OffTrackModel(GD1Base, StreamModel):
    name = "offtrack"

    ln_N_dist = dist.Uniform(-5, 10)

    dens_phi1_lim = (-60, 0)
    dens_phi2_lim = (-3, 3)

    dens_steps = [2.0, 0.25]
    spar_steps = [8.0, 4.0]

    dens_locs = np.stack(
        np.meshgrid(
            np.arange(dens_phi1_lim[0], dens_phi1_lim[1] + 1e-3, dens_steps[0]),
            np.arange(dens_phi2_lim[0], dens_phi2_lim[1] + 1e-3, dens_steps[1]),
        )
    ).T.reshape(-1, 2)

    spar_locs = np.stack(
        np.meshgrid(
            get_grid(*GD1Base.coord_bounds["phi1"], spar_steps[0], pad_num=1),
            get_grid(*GD1Base.coord_bounds["phi2"], spar_steps[1], pad_num=1),
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
                    [GD1Base.coord_bounds["phi1"][0], GD1Base.coord_bounds["phi2"][0]]
                ),
                np.array(
                    [GD1Base.coord_bounds["phi1"][1], GD1Base.coord_bounds["phi2"][1]]
                ),
            ),
        ),
        "pm1": GD1StreamModel.variables["pm1"],
        "pm2": GD1StreamModel.variables["pm2"],
    }

    data_required = {
        ("phi1", "phi2"): {"y": ("phi1", "phi2")},
        "pm1": {"x": "phi1", "y": "pm1", "y_err": "pm1_err"},
        "pm2": {"x": "phi1", "y": "pm2", "y_err": "pm2_err"},
    }
