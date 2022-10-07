from functools import partial

import jax
import jax.numpy as jnp

from .helpers import ln_normal
from .stream import StreamModel

__all__ = ["SpurModel"]


class SpurModel(StreamModel):
    name = "spur"

    phi2_cut = None
    pm1_cut = None
    plx_max = None

    integ_grid_phi1 = jnp.arange(-100, 20 + 1e-3, 0.1)

    knot_lim = (-110, 30)
    ln_n0_knots = jnp.arange(knot_lim[0], knot_lim[1] + 1e-3, 4.0)  # note: arange!
    phi2_knots = jnp.arange(knot_lim[0], knot_lim[1] + 1e-3, 4.0)  # note: arange!
    # plx_knots = jnp.linspace(-110, 30, 9)
    # pm1_knots = jnp.linspace(*knot_lim, len(StreamModel.pm1_knots))
    # pm2_knots = jnp.linspace(*knot_lim, len(StreamModel.pm2_knots))

    param_names = {
        "ln_n0": len(ln_n0_knots),
        "mean_phi2": len(phi2_knots),
        "ln_std_phi2": len(phi2_knots),
        # "mean_plx": len(plx_knots),
        # "mean_pm1": len(pm1_knots),
        # "ln_std_pm1": len(pm1_knots),
        # "mean_pm2": len(pm2_knots),
        # "ln_std_pm2": len(pm2_knots),
    }

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, pars):
        lp = 0.0

        prior_stds = {
            "mean_phi2": 0.5,
            "ln_std_phi2": 0.2,
            # "mean_plx": 1.0,
            # "mean_pm1": 3.0,
            # "ln_std_pm1": 0.5,
            # "mean_pm2": 3.0,
            # "ln_std_pm2": 0.5,
        }
        for name, size in cls.param_names.items():
            if name not in prior_stds:
                continue

            for i in range(1, size):
                lp += ln_normal(pars[name][i], pars[name][i - 1], prior_stds[name])

        lp += jnp.sum(
            ln_normal(
                pars["mean_phi2"],
                1.3 * jax.nn.sigmoid((cls.phi2_knots - -39) / 2.5),
                0.05,
            )
        )

        lp += jnp.sum(
            jax.nn.log_sigmoid((pars["ln_n0"] - -45) / 0.5)
            + jax.nn.log_sigmoid((-pars["ln_n0"] - 20) / 1.0)
        )

        return lp
