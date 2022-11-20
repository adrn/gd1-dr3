import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .base import Model
from .helpers import ln_simpson

__all__ = ["SpurModel"]


class SpurModel(Model):
    name = "spur"

    phi1_lim = (-45, 0)
    integ_grid_phi1 = jnp.arange(phi1_lim[0], phi1_lim[1] + 1e-3, 0.1)
    knots = {
        "ln_n0": jnp.arange(-55, 0 + 1e-3, 3.0),  # 3º step
        "phi2": jnp.arange(-55, 0 + 1e-3, 3.0),  # 3º step
    }
    params_to_knots = {
        "ln_n0": "ln_n0",
        "mean_phi2": "phi2",
        "ln_std_phi2": "phi2",
    }
    shapes = {
        "ln_n0": len(knots["ln_n0"]),
        "mean_phi2": len(knots["phi2"]),
        "ln_std_phi2": len(knots["phi2"]),
    }
    bounds = {
        "ln_n0": (-8, 8),
        "mean_phi2": Model.phi2_lim,
        "ln_std_phi2": (-5, 0),
    }

    @classmethod
    def setup_pars(cls, **_):
        pars = {}

        # ln_n0 : linear density
        pars["ln_n0"] = numpyro.sample(
            f"ln_n0_{cls.name}",
            dist.Uniform(*cls.bounds["ln_n0"]),
            sample_shape=(len(cls.knots["ln_n0"]),),
        )

        name = "phi2"
        pars[f"mean_{name}"] = numpyro.sample(
            f"mean_{name}_{cls.name}",
            dist.Uniform(*cls.bounds[f"mean_{name}"]),
            sample_shape=(len(cls.knots[name]),),
        )
        pars[f"ln_std_{name}"] = numpyro.sample(
            f"ln_std_{name}_{cls.name}",
            dist.Uniform(*cls.bounds[f"ln_std_{name}"]),
            sample_shape=(len(cls.knots[name]),),
        )

        # TODO: smoothness priors on derivative of splines

        return pars

    @classmethod
    def setup_splines(cls, pars, **_):
        spls = {}

        spls["ln_n0"] = InterpolatedUnivariateSpline(
            cls.knots["ln_n0"], pars["ln_n0"], k=1
        )

        name = "phi2"
        spls[f"mean_{name}"] = InterpolatedUnivariateSpline(
            cls.knots[name], pars[f"mean_{name}"], k=1
        )
        spls[f"ln_std_{name}"] = InterpolatedUnivariateSpline(
            cls.knots[name], pars[f"ln_std_{name}"], k=1
        )

        return spls

    @classmethod
    def setup_dists(cls, spls, data, stream_spls):
        dists = {}
        if "phi1" in data:
            dists["ln_n0"] = spls["ln_n0"]

            dists["phi2"] = dist.TruncatedNormal(
                loc=spls["mean_phi2"](data["phi1"]),
                scale=jnp.exp(spls["ln_std_phi2"](data["phi1"])),
                low=cls.phi2_lim[0],
                high=cls.phi2_lim[1],
            )

            dists["pm1"] = dist.TruncatedNormal(
                loc=stream_spls["mean_pm1"](data["phi1"]),
                scale=jnp.exp(stream_spls["ln_std_pm1"](data["phi1"])),
                low=cls.pm1_lim[0],
                high=cls.pm1_lim[1],
            )

            dists["pm2"] = dist.Normal(
                loc=stream_spls["mean_pm2"](data["phi1"]),
                scale=jnp.exp(stream_spls["ln_std_pm2"](data["phi1"])),
            )

            return dists

    @classmethod
    def setup_obs(cls, dists, data, **_):
        ln_V = ln_simpson(dists["ln_n0"](cls.integ_grid_phi1), x=cls.integ_grid_phi1)
        numpyro.factor(
            f"obs_ln_n0_{cls.name}",
            -jnp.exp(ln_V) + dists["ln_n0"](data["phi1"]).sum(),
        )
        numpyro.sample(f"obs_phi2_{cls.name}", dists["phi2"], obs=data["phi2"])
        numpyro.sample(f"obs_pm1_{cls.name}", dists["pm1"], obs=data["pm1"])
        numpyro.sample(f"obs_pm2_{cls.name}", dists["pm2"], obs=data["pm2"])

    @classmethod
    def setup_other_priors(cls, spls):
        lp = 0.0

        # Smoothness priors
        smooth = {
            "ln_n0": 1.0 / 30.0,
            "mean_phi2": 1.0 / 10.0,
            "ln_std_phi2": 0.5 / 10.0,
        }
        for param_name, std in smooth.items():
            knots = cls.knots[cls.params_to_knots[param_name]]
            deriv = spls[param_name].derivative(knots)
            lp = lp + dist.Normal(0.0, std).log_prob(deriv).sum()

        numpyro.factor(f"smooth_{cls.name}", lp)

        # TODO: sigmoid shit
