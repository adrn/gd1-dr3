from functools import partial

import jax
import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .base import Model
from .helpers import (
    ln_normal,
    ln_simpson,
    two_norm_mixture_ln_prob,
    two_truncnorm_mixture_ln_prob,
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

    phi2_cut = None
    pm1_cut = None

    ln_n0_knots = jnp.linspace(-110, 30, 9)
    # plx_knots = jnp.linspace(-110, 30, 9)
    pm1_knots = jnp.linspace(-110, 30, 7)
    pm2_knots = jnp.linspace(-110, 30, 7)

    integ_grid_phi1 = jnp.arange(-100, 20 + 1e-3, 0.1)

    param_names = {
        "ln_n0": len(ln_n0_knots),
        # "mean_plx": len(plx_knots),
        "w_pm1": len(pm1_knots),
        "mean1_pm1": len(pm1_knots),
        "ln_std1_pm1": len(pm1_knots),
        "mean2_pm1": len(pm1_knots),
        "ln_std2_pm1": len(pm1_knots),
        "w_pm2": len(pm2_knots),
        "mean1_pm2": len(pm2_knots),
        "ln_std1_pm2": len(pm2_knots),
        "mean2_pm2": len(pm2_knots),
        "ln_std2_pm2": len(pm2_knots),
    }

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_n0(cls, phi1, pars):
        ln_n0_spl = InterpolatedUnivariateSpline(cls.ln_n0_knots, pars["ln_n0"], k=3)
        return ln_n0_spl(phi1)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def phi2(cls, data, pars):
        return jnp.full(data["phi1"].shape, -jnp.log(cls.phi2_cut[1] - cls.phi2_cut[0]))

    # @classmethod
    # @partial(jax.jit, static_argnums=(0,))
    # def plx(cls, data, mean_plx):
    #     """ln_likelihood for parallax"""
    #     mean_plx_spl = InterpolatedUnivariateSpline(cls.phi2_knots, mean_plx, k=3)
    #     return ln_normal(data['parallax'], mean_plx_spl(phi1), data['parallax_error'])

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pm1(cls, data, pars):
        """ln_likelihood for pm1*cos(phi2)"""
        w_spl = InterpolatedUnivariateSpline(cls.pm1_knots, pars["w_pm1"], k=3)
        mean1_spl = InterpolatedUnivariateSpline(cls.pm1_knots, pars["mean1_pm1"], k=3)
        ln_std1_spl = InterpolatedUnivariateSpline(
            cls.pm1_knots, pars["ln_std1_pm1"], k=3
        )
        mean2_spl = InterpolatedUnivariateSpline(cls.pm1_knots, pars["mean2_pm1"], k=3)
        ln_std2_spl = InterpolatedUnivariateSpline(
            cls.pm1_knots, pars["ln_std2_pm1"], k=3
        )

        p = jnp.stack(
            (
                w_spl(data["phi1"]),
                mean1_spl(data["phi1"]),
                ln_std1_spl(data["phi1"]),
                mean2_spl(data["phi1"]),
                ln_std2_spl(data["phi1"]),
            )
        )
        return two_truncnorm_mixture_ln_prob(
            p, data["pm1"], data["pm1_error"], *cls.pm1_cut
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pm2(cls, data, pars):
        """ln_likelihood for pm2"""
        w_spl = InterpolatedUnivariateSpline(cls.pm2_knots, pars["w_pm2"], k=3)
        mean1_spl = InterpolatedUnivariateSpline(cls.pm2_knots, pars["mean1_pm2"], k=3)
        ln_std1_spl = InterpolatedUnivariateSpline(
            cls.pm2_knots, pars["ln_std1_pm2"], k=3
        )
        mean2_spl = InterpolatedUnivariateSpline(cls.pm2_knots, pars["mean2_pm2"], k=3)
        ln_std2_spl = InterpolatedUnivariateSpline(
            cls.pm2_knots, pars["ln_std2_pm2"], k=3
        )

        p = jnp.stack(
            (
                w_spl(data["phi1"]),
                mean1_spl(data["phi1"]),
                ln_std1_spl(data["phi1"]),
                mean2_spl(data["phi1"]),
                ln_std2_spl(data["phi1"]),
            )
        )
        return two_norm_mixture_ln_prob(p, data["pm2"], data["pm2_error"])

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, pars, data):
        ln_dens = (
            cls.ln_n0(data["phi1"], pars)
            + cls.phi2(data, pars)
            + cls.pm1(data, pars)
            + cls.pm2(data, pars)
            # + cls.plx(data, pars['mean_plx'])
        )

        ln_dens_grid = cls.ln_n0(cls.integ_grid_phi1, pars)
        ln_V = ln_simpson(ln_dens_grid, x=cls.integ_grid_phi1)

        return -jnp.exp(ln_V) + ln_dens.sum()

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, pars):
        lp = 0.0

        prior_stds = {
            # "mean_plx": 1.,
            "mean1_pm1": 3.0,
            "ln_std1_pm1": 0.5,
            "mean2_pm1": 3.0,
            "ln_std2_pm1": 0.5,
            "mean1_pm2": 3.0,
            "ln_std1_pm2": 0.5,
            "mean2_pm2": 3.0,
            "ln_std2_pm2": 0.5,
        }
        for name, size in cls.param_names.items():
            if name not in prior_stds:
                continue

            for i in range(1, size):
                lp += ln_normal(pars[name][i], pars[name][i - 1], prior_stds[name])

        return lp
