import jax.numpy as jnp
import numpy as np
import numpyro

from .helpers import ln_simpson

__all__ = ["Model"]


class Model:
    phi1_lim = (-100, 20)
    phi2_lim = (-7, 5.0)
    pm1_lim = (-20, -1.0)

    integ_grid_phi1 = jnp.arange(phi1_lim[0], phi1_lim[1] + 1e-3, 0.2)
    bounds = {}

    @classmethod
    def clip_pars(cls, pars):
        new_pars = {}
        for k in pars:
            if k in cls.bounds:
                # TODO: tolerance MAGIC NUMBER 1e-2
                new_pars[k] = jnp.clip(
                    pars[k], cls.bounds[k][0] + 1e-2, cls.bounds[k][1] - 1e-2
                )
            else:
                new_pars[k] = pars[k]
        return new_pars

    @classmethod
    def strip_class_name(cls, pars):
        return {k[: -(len(cls.name) + 1)]: v for k, v in pars.items()}

    @classmethod
    def compute_factor(cls, ln_n0_spl, phi1):
        ln_V = ln_simpson(ln_n0_spl(cls.integ_grid_phi1), x=cls.integ_grid_phi1)
        return -jnp.exp(ln_V) + ln_n0_spl(phi1).sum()

    @classmethod
    def setup_obs(cls, dists, data):
        numpyro.factor(
            f"obs_ln_n0_{cls.name}", cls.compute_factor(dists["ln_n0"], data["phi1"])
        )
        numpyro.sample(f"obs_phi2_{cls.name}", dists["phi2"], obs=data["phi2"])
        numpyro.sample(f"obs_pm1_{cls.name}", dists["pm1"], obs=data["pm1"])
        numpyro.sample(f"obs_pm2_{cls.name}", dists["pm2"], obs=data["pm2"])

    @classmethod
    def setup_model(cls, data, **kwargs):
        pars = cls.setup_pars(**kwargs)
        spls = cls.setup_splines(pars, **kwargs)
        dists = cls.setup_dists(spls, data, **kwargs)
        cls.setup_obs(dists, data, **kwargs)
        cls.setup_other_priors(spls)

    @classmethod
    def evaluate_on_grids(cls, pars, grids=None, **kwargs):
        from .plot import _default_grids

        if grids is None:
            grids = {}
        for name in _default_grids:
            grids.setdefault(name, _default_grids[name])

        spls = cls.setup_splines(pars, **kwargs)

        all_grids = {}
        terms = {}
        for name in ["phi2", "pm1", "pm2"]:
            grid1, grid2 = np.meshgrid(grids["phi1"], grids[name])
            dists = cls.setup_dists(
                spls, {"phi1": grid1.ravel(), f"{name}_err": 0.0}, **kwargs
            )

            ln_dens = dists[name].log_prob(grid2.ravel()) + spls["ln_n0"](grid1.ravel())
            terms[name] = ln_dens.reshape(grid1.shape)
            all_grids[name] = (grid1, grid2)

        return all_grids, terms

    @classmethod
    def plot_projections(
        cls,
        pars,
        grids=None,
        axes=None,
        label=True,
        **kwargs,
    ):
        from .plot import _default_labels

        grids, ln_denses = cls.evaluate_on_grids(pars=pars, grids=grids, **kwargs)

        if axes is None:
            import matplotlib.pyplot as plt

            _, axes = plt.subplots(
                3,
                1,
                figsize=(10, 8),
                sharex=True,
                sharey="row",
                constrained_layout=True,
            )

        for i, name in enumerate(["phi2", "pm1", "pm2"]):
            grid1, grid2 = grids[name]
            axes[i].pcolormesh(
                grid1, grid2, np.exp(ln_denses[name]), shading="auto", **kwargs
            )
            axes[i].set_ylim(grid2.min(), grid2.max())

            if label:
                axes[i].set_ylabel(_default_labels[name])

        axes[0].set_xlim(grid1.min(), grid1.max())

        return axes.flat[0].figure, axes
