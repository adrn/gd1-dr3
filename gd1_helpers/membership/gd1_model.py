import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from stream_membership import CustomTruncatedNormal, SplineDensityModelBase

__all__ = ["GD1BackgroundModel", "GD1StreamModel"]


class GD1ComponentBase:
    coord_names = ("phi2",)  # "pm1")
    coord_bounds = {"phi1": (-100, 20), "phi2": (-7, 5), "pm1": (-15, -1.0)}

    default_grids = {
        "phi1": np.arange(*coord_bounds["phi1"], 0.2),
        "phi2": np.arange(*coord_bounds["phi2"], 0.1),
        "pm1": np.arange(*coord_bounds["pm1"], 0.1),
        # "pm2": np.arange(-10, 10 + 1e-3, 0.1),
    }


class GD1BackgroundModel(GD1ComponentBase, SplineDensityModelBase):
    name = "background"

    knots = {
        "ln_n0": jnp.linspace(-110, 30, 9),
    }
    param_bounds = {"ln_n0": (-5, 8), "phi2": {}, "pm1": {}}

    # Can probably use a lower resolution grid here?
    integration_grid_phi1 = jnp.arange(-100, 20 + 1e-3, 1.0)

    @classmethod
    def setup_numpyro(cls, data=None):
        pars = {}

        # ln_n0 : linear density
        pars["ln_n0"] = numpyro.sample(
            f"ln_n0_{cls.name}",
            dist.Uniform(*cls.param_bounds["ln_n0"]),
            sample_shape=(len(cls.knots["ln_n0"]),),
        )

        return cls(pars=pars, data=data)

    def get_dists(self, data):
        dists = {}
        dists["ln_n0"] = self.splines["ln_n0"]
        dists["phi2"] = dist.Uniform(-7, jnp.full(len(data["phi1"]), 5))
        return dists

    def extra_ln_prior(self):
        lp = 0.0
        lp += (
            dist.Normal(0, 0.5)
            .log_prob(self.splines["ln_n0"]._y[1:] - self.splines["ln_n0"]._y[:-1])
            .sum()
        )
        return lp


class GD1StreamModel(GD1ComponentBase, SplineDensityModelBase):
    name = "stream"

    _step_n0 = 4.0  # deg
    _step_phi2 = 6.0  # deg
    knots = {
        "ln_n0": jnp.arange(-100 - _step_n0, 20 + _step_n0 + 1e-3, _step_n0),
        "phi2": jnp.arange(-100 - _step_phi2, 20 + _step_phi2 + 1e-3, _step_phi2),
    }

    param_bounds = {
        "ln_n0": (-5, 8),
        "phi2": {"mean": GD1ComponentBase.coord_bounds["phi2"], "ln_std": (-2, 0.5)},
    }

    integration_grid_phi1 = jnp.arange(-100, 20 + 1e-3, 0.2)

    @classmethod
    def setup_numpyro(cls, data=None):
        pars = {}

        # ln_n0 : linear density
        pars["ln_n0"] = numpyro.sample(
            f"ln_n0_{cls.name}",
            dist.Uniform(*cls.param_bounds["ln_n0"]),
            sample_shape=(len(cls.knots["ln_n0"]),),
        )

        # Other coordinates:
        for coord_name in cls.coord_names:
            pars[coord_name] = {}
            for par_name in ["mean", "ln_std"]:
                pars[coord_name][par_name] = numpyro.sample(
                    f"{coord_name}_{par_name}_{cls.name}",
                    dist.Uniform(*cls.param_bounds[coord_name][par_name]),
                    sample_shape=(len(cls.knots[coord_name]),),
                )

        return cls(pars=pars, data=data)

    def get_dists(self, data):
        dists = {}
        dists["ln_n0"] = self.splines["ln_n0"]
        dists["phi2"] = CustomTruncatedNormal(
            loc=self.splines["phi2"]["mean"](data["phi1"]),
            scale=jnp.exp(self.splines["phi2"]["ln_std"](data["phi1"])),
            low=self.coord_bounds["phi2"][0],
            high=self.coord_bounds["phi2"][1],
        )
        return dists

    def extra_ln_prior(self):
        lp = 0.0

        lp += (
            dist.Normal(0, 0.25)
            .log_prob(self.splines["ln_n0"]._y[1:] - self.splines["ln_n0"]._y[:-1])
            .sum()
        )

        for par_name in ["mean", "ln_std"]:
            lp += (
                dist.Normal(0, 0.1)
                .log_prob(
                    self.splines["phi2"][par_name]._y[1:]
                    - self.splines["phi2"][par_name]._y[:-1]
                )
                .sum()
            )

        return lp
