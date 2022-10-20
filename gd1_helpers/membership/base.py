import jax.numpy as jnp
import numpy as np

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
    def setup_model(cls, data, **kwargs):
        pars = cls.setup_pars(**kwargs)
        spls = cls.setup_splines(pars, **kwargs)
        dists = cls.setup_dists(spls, data, **kwargs)
        cls.setup_obs(dists, data, **kwargs)

    @classmethod
    def plot_projections(
        cls,
        pars,
        grids=None,
        axes=None,
        label=True,
        **kwargs,
    ):
        from .plot import _default_grids, _default_labels

        if grids is None:
            grids = {}
        for name in _default_grids:
            grids.setdefault(name, _default_grids[name])

        if axes is None:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                3,
                1,
                figsize=(10, 8),
                sharex=True,
                sharey="row",
                constrained_layout=True,
            )

        spls = cls.setup_splines(pars, **kwargs)

        for i, name in enumerate(["phi2", "pm1", "pm2"]):
            grid1, grid2 = np.meshgrid(grids["phi1"], grids[name])
            dists = cls.setup_dists(
                spls, {"phi1": grid1.ravel(), f"{name}_err": 0.0}, **kwargs
            )

            ln_dens = dists[name].log_prob(grid2.ravel()) + spls["ln_n0"](grid1.ravel())
            ln_dens = ln_dens.reshape(grid1.shape)
            axes[i].pcolormesh(grid1, grid2, np.exp(ln_dens), shading="auto", **kwargs)
            axes[i].set_ylim(grids[name].min(), grids[name].max())

            if label:
                axes[i].set_ylabel(_default_labels[name])

        axes[0].set_xlim(grids["phi1"].min(), grids["phi1"].max())

        return axes.flat[0].figure, axes
