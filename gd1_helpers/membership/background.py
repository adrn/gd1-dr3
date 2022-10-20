import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .base import Model
from .helpers import (
    ln_simpson,
    two_normal_mixture_spl,
    two_truncated_normal_mixture_spl,
)

__all__ = ["BackgroundModel"]


class BackgroundModel(Model):
    r"""
    - $\phi_2$: same as in demo notebook / blog post
    - (ignore at first) parallax: Exponentially-decreasing space density prior with
      $L(\phi_1)$ a function of $\phi_1$
    - pm1: mixture of 2 truncated normal distributions with mixture weight $w(\phi_1)$,
      mean $\mu(\phi_1)$, and two log-variances $\sigma_1^2(\phi_1)$,
      $\sigma_2^2(\phi_1) = \sigma_1^2(\phi_1) + \hat{\sigma}_2^2(\phi_1)$
    - pm2: mixture of 2 normal distributions with mixture weight $w(\phi_1)$, mean
      $\mu(\phi_1)$, and two log-variances $\sigma_1^2(\phi_1)$, $\sigma_2^2(\phi_1) =
      \sigma_1^2(\phi_1) + \hat{\sigma}_2^2(\phi_1)$
    """
    name = "background"

    knots = {
        "ln_n0": jnp.linspace(-110, 30, 9),
        "pm1": jnp.linspace(-110, 30, 7),
        "pm2": jnp.linspace(-110, 30, 7),
    }
    shapes = {
        "ln_n0": len(knots["ln_n0"]),
        "w_pm1": len(knots["pm1"]),
        "mean1_pm1": len(knots["pm1"]),
        "mean2_pm1": len(knots["pm1"]),
        "ln_std1_pm1": len(knots["pm1"]),
        "ln_std2_pm1": len(knots["pm1"]),
        "w_pm2": len(knots["pm2"]),
        "mean1_pm2": len(knots["pm2"]),
        "mean2_pm2": len(knots["pm2"]),
        "ln_std1_pm2": len(knots["pm2"]),
        "ln_std2_pm2": len(knots["pm2"]),
    }
    bounds = {
        "ln_n0": (-2, 8),
        "w_pm1": (0, 1),
        "mean1_pm1": (-5, 20),
        "mean2_pm1": (-5, 20),
        "ln_std1_pm1": (-5, 5),
        "ln_std2_pm1": (-5, 5),
        "w_pm2": (0, 1),
        "mean1_pm2": (-10, 10),
        "mean2_pm2": (-10, 10),
        "ln_std1_pm2": (-5, 5),
        "ln_std2_pm2": (-5, 5),
    }

    @classmethod
    def setup_pars(cls):
        pars = {}

        # ln_n0 : linear density
        pars["ln_n0"] = numpyro.sample(
            f"ln_n0_{cls.name}",
            dist.Uniform(*cls.bounds["ln_n0"]),
            sample_shape=(len(cls.knots["ln_n0"]),),
        )

        # plx : parallax
        # mean_plx_spl = InterpolatedUnivariateSpline(cls.phi2_knots, mean_plx, k=3)
        # return ln_normal(data['parallax'], mean_plx_spl(phi1), data['parallax_error'])

        # pm1 : proper motion mu_1 * cos(phi_2)
        # pm2 : proper motion mu_2
        for pm_name in ["pm1", "pm2"]:
            for comp_name in ["w", "mean1", "mean2", "ln_std1", "ln_std2"]:
                name = f"{comp_name}_{pm_name}"
                pars[name] = numpyro.sample(
                    f"{name}_{cls.name}",
                    dist.Uniform(*cls.bounds[name]),
                    sample_shape=(len(cls.knots[pm_name]),),
                )

        return pars

    @classmethod
    def setup_splines(cls, pars):
        spls = {}

        spls["ln_n0"] = InterpolatedUnivariateSpline(
            cls.knots["ln_n0"], pars["ln_n0"], k=3
        )

        for pm_name in ["pm1", "pm2"]:
            for comp_name in ["w", "mean1", "mean2", "ln_std1", "ln_std2"]:
                name = f"{comp_name}_{pm_name}"
                spls[name] = InterpolatedUnivariateSpline(
                    cls.knots[pm_name], pars[name], k=3
                )

        return spls

    @classmethod
    def setup_dists(cls, spls, data):
        dists = {}
        dists["ln_n0"] = spls["ln_n0"]

        dists["phi2"] = dist.Uniform(-7, jnp.full(len(data["phi1"]), 5))

        dists["pm1"] = two_truncated_normal_mixture_spl(
            w_spl=spls["w_pm1"],
            mean_spls=[spls["mean1_pm1"], spls["mean2_pm1"]],
            ln_std_spls=[spls["ln_std1_pm1"], spls["ln_std2_pm1"]],
            x=data["phi1"],
            yerr=data.get("pm1_err", 0.0),
            low=cls.pm1_lim[0],
            high=cls.pm1_lim[1],
        )

        dists["pm2"] = two_normal_mixture_spl(
            w_spl=spls["w_pm2"],
            mean_spls=[spls["mean1_pm2"], spls["mean2_pm2"]],
            ln_std_spls=[spls["ln_std1_pm2"], spls["ln_std2_pm2"]],
            x=data["phi1"],
            yerr=data.get("pm2_err", 0.0),
        )
        return dists

    @classmethod
    def setup_obs(cls, dists, data):
        ln_V = ln_simpson(dists["ln_n0"](cls.integ_grid_phi1), x=cls.integ_grid_phi1)
        numpyro.factor(
            f"obs_ln_n0_{cls.name}",
            -jnp.exp(ln_V) + dists["ln_n0"](data["phi1"]).sum(),
        )
        numpyro.sample(f"obs_phi2_{cls.name}", dists["phi2"], obs=data["phi2"])
        numpyro.sample(f"obs_pm1_{cls.name}", dists["pm1"], obs=data["pm1"])
        numpyro.sample(f"obs_pm2_{cls.name}", dists["pm2"], obs=data["pm2"])

    # TODO: smoothness priors on derivative of splines
