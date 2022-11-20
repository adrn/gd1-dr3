import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from stream_membership import CustomTruncatedNormal, SplineDensityModelBase
from stream_membership.helpers import two_normal_mixture, two_truncated_normal_mixture

__all__ = ["GD1BackgroundModel", "GD1StreamModel"]


def z_to_w(z):
    return (jnp.tanh(z) + 1) / 2.0


def w_to_z(w):
    return jnp.arctanh(2 * w - 1)


def _get_knots(low, high, step, pad_num=2, arange=True):
    if arange:
        return jnp.arange(low - pad_num * step, high + pad_num * step + step, step)
    else:
        return jnp.linspace(low - pad_num * step, high + pad_num * step, step)


class GD1ComponentBase:
    coord_names = ("phi2", "pm1", "pm2")
    coord_bounds = {"phi1": (-100, 20), "phi2": (-7, 5), "pm1": (-15, -1.0)}

    default_grids = {
        "phi1": np.arange(*coord_bounds["phi1"], 0.2),
        "phi2": np.arange(*coord_bounds["phi2"], 0.1),
        "pm1": np.arange(*coord_bounds["pm1"], 0.1),
        "pm2": np.arange(-10, 10 + 1e-3, 0.1),
    }


phi1_lim = GD1ComponentBase.coord_bounds["phi1"]


class GD1BackgroundModel(GD1ComponentBase, SplineDensityModelBase):
    name = "background"

    knots = {
        "ln_n0": _get_knots(*phi1_lim, 9, arange=False),
        "pm1": _get_knots(*phi1_lim, 7, arange=False),
        "pm2": _get_knots(*phi1_lim, 9, arange=False),
    }
    param_bounds = {
        "ln_n0": (-5, 8),
        "phi2": {},
        "pm1": {
            "z": (-8, 8),
            "mean1": (-5, 20),
            "ln_std1": (-1, 5),
            "mean2": (-5, 20),
            "ln_std2": (-1, 5),
        },
        "pm2": {
            "z": (-8, 8),
            "mean1": (-8, 8),
            "ln_std1": (-1, 5),
            "mean2": (-8, 8),
            "ln_std2": (-1, 5),
        },
    }

    spline_ks = {"pm1": {"z": 1}, "pm2": {"z": 1}}

    # Can probably use a lower resolution grid here?
    integration_grid_phi1 = jnp.arange(phi1_lim[0], phi1_lim[1] + 1e-3, 0.1)

    @classmethod
    def setup_numpyro(cls, data=None):
        pars = {}

        # ln_n0 : linear density
        pars["ln_n0"] = numpyro.sample(
            f"ln_n0_{cls.name}",
            dist.Uniform(*cls.param_bounds["ln_n0"]),
            sample_shape=(len(cls.knots["ln_n0"]),),
        )

        # proper motions:
        for coord_name in ["pm1", "pm2"]:
            if coord_name not in cls.coord_names:
                continue

            pars[coord_name] = {}

            par_name = "z"
            pars[coord_name][par_name] = numpyro.sample(
                f"{coord_name}_{par_name}_{cls.name}",
                CustomTruncatedNormal(
                    loc=0.0,
                    scale=2,
                    low=cls.param_bounds[coord_name][par_name][0],
                    high=cls.param_bounds[coord_name][par_name][1],
                ),
                sample_shape=(len(cls.knots[coord_name]),),
            )
            for par_name in ["mean1", "ln_std1", "mean2", "ln_std2"]:
                pars[coord_name][par_name] = numpyro.sample(
                    f"{coord_name}_{par_name}_{cls.name}",
                    dist.Uniform(*cls.param_bounds[coord_name][par_name]),
                    sample_shape=(len(cls.knots[coord_name]),),
                )

        return cls(pars=pars, data=data)

    def get_dists(self, data):
        dists = {}

        if "phi2" in self.coord_names:
            dists["phi2"] = dist.Uniform(-7, jnp.full(len(data["phi1"]), 5))

        if "pm1" in self.coord_names:
            dists["pm1"] = two_truncated_normal_mixture(
                w=z_to_w(self.splines["pm1"]["z"](data["phi1"])),
                mean1=self.splines["pm1"]["mean1"](data["phi1"]),
                mean2=self.splines["pm1"]["mean2"](data["phi1"]),
                ln_std1=self.splines["pm1"]["ln_std1"](data["phi1"]),
                ln_std2=self.splines["pm1"]["ln_std2"](data["phi1"]),
                low=self.coord_bounds["pm1"][0],
                high=self.coord_bounds["pm1"][1],
                yerr=data["pm1_err"],
            )

        if "pm2" in self.coord_names:
            dists["pm2"] = two_normal_mixture(
                w=z_to_w(self.splines["pm2"]["z"](data["phi1"])),
                mean1=self.splines["pm2"]["mean1"](data["phi1"]),
                mean2=self.splines["pm2"]["mean2"](data["phi1"]),
                ln_std1=self.splines["pm2"]["ln_std1"](data["phi1"]),
                ln_std2=self.splines["pm2"]["ln_std2"](data["phi1"]),
                yerr=data["pm2_err"],
            )

        return dists

    def extra_ln_prior(self):
        lp = 0.0

        lp += (
            dist.Normal(0, 0.5)
            .log_prob(self.splines["ln_n0"]._y[1:] - self.splines["ln_n0"]._y[:-1])
            .sum()
        )

        std_map = {"mean": 0.5, "ln_std": 0.1, "_z_": 0.5}
        for coord_name in self.coord_names:
            if coord_name not in self.splines:
                continue

            for par_name in self.splines[coord_name]:
                for check in ["mean", "ln_std", "_z_"]:
                    if check in par_name:
                        std = std_map[check]
                        break
                else:
                    std = 1.0

                spl_y = self.splines[coord_name][par_name]._y
                lp += dist.Normal(0, std).log_prob(spl_y[1:] - spl_y[:-1]).sum()

        return lp


class GD1StreamModel(GD1ComponentBase, SplineDensityModelBase):
    name = "stream"

    knots = {
        "ln_n0": _get_knots(*phi1_lim, 4.0),
        "phi2": _get_knots(*phi1_lim, 6.0),
        "pm1": _get_knots(*phi1_lim, 10.0),
        "pm2": _get_knots(*phi1_lim, 10.0),
    }

    param_bounds = {
        "ln_n0": (-5, 8),
        "phi2": {"mean": GD1ComponentBase.coord_bounds["phi2"], "ln_std": (-2, 0.5)},
        "pm1": {
            "mean": GD1ComponentBase.coord_bounds["pm1"],
            "ln_std": (-5, -0.75),
        },  # 20 km/s
        "pm2": {"mean": (-5, 5), "ln_std": (-5, -0.75)},
    }

    integration_grid_phi1 = jnp.arange(phi1_lim[0], phi1_lim[1] + 1e-3, 0.2)

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
        if "phi2" in self.coord_names:
            # dists["phi2"] = CustomTruncatedNormal(
            dists["phi2"] = dist.TruncatedNormal(
                loc=self.splines["phi2"]["mean"](data["phi1"]),
                scale=jnp.exp(self.splines["phi2"]["ln_std"](data["phi1"])),
                low=self.coord_bounds["phi2"][0],
                high=self.coord_bounds["phi2"][1],
            )
        if "pm1" in self.coord_names:
            # dists["pm1"] = CustomTruncatedNormal(
            dists["pm1"] = dist.TruncatedNormal(
                loc=self.splines["pm1"]["mean"](data["phi1"]),
                scale=jnp.exp(self.splines["pm1"]["ln_std"](data["phi1"])),
                low=self.coord_bounds["pm1"][0],
                high=self.coord_bounds["pm1"][1],
            )
        if "pm2" in self.coord_names:
            dists["pm2"] = dist.Normal(
                loc=self.splines["pm2"]["mean"](data["phi1"]),
                scale=jnp.exp(self.splines["pm2"]["ln_std"](data["phi1"])),
            )
        return dists

    def extra_ln_prior(self):
        lp = 0.0

        lp += (
            dist.Normal(0, 0.25)
            .log_prob(self.splines["ln_n0"]._y[1:] - self.splines["ln_n0"]._y[:-1])
            .sum()
        )

        std_map = {"mean": 0.5, "ln_std": 0.1, "_w_": 0.1}
        for coord_name in self.coord_names:
            for par_name in self.splines[coord_name]:
                for check in ["mean", "ln_std", "_w_"]:
                    if check in par_name:
                        std = std_map[check]
                        break
                else:
                    std = 1.0

                spl_y = self.splines[coord_name][par_name]._y
                lp += dist.Normal(0, std).log_prob(spl_y[1:] - spl_y[:-1]).sum()

        return lp
