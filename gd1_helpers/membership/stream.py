from functools import partial

import jax
import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .base import Model
from .helpers import ln_normal, ln_simpson, ln_truncated_normal

__all__ = ["StreamModel"]


class StreamModel(Model):
    phi2_cut = None
    pm1_cut = None

    ln_n0_knots = jnp.linspace(-110, 30, 35)
    phi2_knots = jnp.linspace(-110, 30, 17)
    # plx_knots = jnp.linspace(-110, 30, 9)
    pm1_knots = jnp.linspace(-110, 30, 9)
    pm2_knots = jnp.linspace(-110, 30, 7)

    integ_grid_phi1 = jnp.arange(-100, 20 + 1e-3, 0.1)

    param_names = {
        "mean_phi2": len(phi2_knots),
        "ln_std_phi2": len(phi2_knots),
        # "mean_plx": len(plx_knots),
        "mean_pm1": len(pm1_knots),
        "ln_std_pm1": len(pm1_knots),
        "mean_pm2": len(pm2_knots),
        "ln_std_pm2": len(pm2_knots),
    }

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_n0(cls, phi1, pars):
        ln_n0_spl = InterpolatedUnivariateSpline(cls.ln_n0_knots, pars["ln_n0"], k=3)
        return ln_n0_spl(phi1)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def phi2(cls, data, pars):
        """ln_likelihood for phi2"""
        mean_phi2_spl = InterpolatedUnivariateSpline(
            cls.phi2_knots, pars["mean_phi2"], k=3
        )
        ln_std_phi2_spl = InterpolatedUnivariateSpline(
            cls.phi2_knots, pars["ln_std_phi2"], k=3
        )
        return ln_normal(
            data["phi2"],
            mean_phi2_spl(data["phi1"]),
            jnp.exp(2 * ln_std_phi2_spl(data["phi1"])),
        )

    # @classmethod
    # @partial(jax.jit, static_argnums=(0,))
    # def plx(cls, data, mean_plx):
    #     """ln_likelihood for parallax"""
    #     mean_plx_spl = InterpolatedUnivariateSpline(cls.phi2_knots, mean_plx, k=3)
    #     return ln_normal(data["parallax"], mean_plx_spl(phi1), data["parallax_error"])

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pm1(cls, data, pars):
        """ln_likelihood for pm1*cos(phi2)"""
        mean_pm1_spl = InterpolatedUnivariateSpline(
            cls.phi2_knots, pars["mean_pm1"], k=3
        )
        ln_std_pm1_spl = InterpolatedUnivariateSpline(
            cls.phi2_knots, pars["ln_std_pm1"], k=3
        )
        ln_var = jnp.logaddexp(2 * ln_std_pm1_spl(data["phi1"]), 2 * data["pm1_error"])
        return ln_truncated_normal(
            data["pm1"],
            mean_pm1_spl(data["phi1"]),
            jnp.exp(ln_var),
            lower=cls.pm1_cut[0],
            upper=cls.pm1_cut[1],
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pm2(cls, data, pars):
        """ln_likelihood for pm2"""
        mean_pm2_spl = InterpolatedUnivariateSpline(
            cls.phi2_knots, pars["mean_pm2"], k=3
        )
        ln_std_pm2_spl = InterpolatedUnivariateSpline(
            cls.phi2_knots, pars["ln_std_pm2"], k=3
        )
        ln_var = jnp.logaddexp(2 * ln_std_pm2_spl(data["phi1"]), 2 * data["pm2_error"])
        return ln_normal(data["pm2"], mean_pm2_spl(data["phi1"]), jnp.exp(ln_var))

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, pars, data):
        ln_dens = (
            cls.ln_n0(data["phi1"], pars)
            + cls.phi2(data, pars)
            + cls.pm1(data, pars)
            + cls.pm2(data, pars)
            # + cls.plx(data, pars["mean_plx"])
        )
        ln_dens_grid = cls.ln_n0(cls.integ_grid_phi1, pars)
        ln_V = ln_simpson(ln_dens_grid, x=cls.integ_grid_phi1)

        return -jnp.exp(ln_V) + ln_dens.sum()

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, pars):
        lp = 0.0

        prior_stds = {
            "mean_phi2": 1.0,
            "ln_std_phi2": 1.0,
            "mean_plx": 1.0,
            "mean_pm1": 3.0,
            "ln_std_pm1": 0.5,
            "mean_pm2": 3.0,
            "ln_std_pm2": 0.5,
        }
        for name, size in cls.param_names.items():
            if name not in prior_stds:
                continue

            for i in range(1, size):
                lp += ln_normal(pars[name][i], pars[name][i - 1], prior_stds[name])

        return lp
