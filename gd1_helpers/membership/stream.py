import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .base import Model

__all__ = ["StreamModel"]


class StreamModel(Model):
    name = "stream"

    knots = {
        "ln_n0": jnp.arange(-100 - 2 * 3.0, 20 + 2 * 3.0 + 1e-3, 3.0),  # 3º step
        "phi2": jnp.arange(-100 - 2 * 5.0, 20 + 2 * 5.0 + 1e-3, 5.0),  # 5º step
        "pm1": jnp.linspace(-110, 30, 9),
        "pm2": jnp.linspace(-110, 30, 7),
    }
    params_to_knots = {
        "ln_n0": "ln_n0",
        "mean_phi2": "phi2",
        "ln_std_phi2": "phi2",
        "mean_pm1": "pm1",
        "ln_std_pm1": "pm1",
        "mean_pm2": "pm2",
        "ln_std_pm2": "pm2",
    }
    shapes = {
        "ln_n0": len(knots["ln_n0"]),
        "mean_phi2": len(knots["phi2"]),
        "ln_std_phi2": len(knots["phi2"]),
        "mean_pm1": len(knots["pm1"]),
        "ln_std_pm1": len(knots["pm1"]),
        "mean_pm2": len(knots["pm2"]),
        "ln_std_pm2": len(knots["pm2"]),
    }
    bounds = {
        "ln_n0": (-8, 8),
        "mean_phi2": Model.phi2_lim,
        "ln_std_phi2": (-2, 0.5),
        "mean_pm1": Model.pm1_lim,
        "ln_std_pm1": (-2.5, 0),
        "mean_pm2": (-5, 5),
        "ln_std_pm2": (-5, -1),
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

        for name in ["phi2", "pm1", "pm2"]:
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

        return pars

    @classmethod
    def setup_splines(cls, pars):
        spls = {}

        spls["ln_n0"] = InterpolatedUnivariateSpline(
            cls.knots["ln_n0"], pars["ln_n0"], k=3
        )

        for name in ["phi2", "pm1", "pm2"]:
            spls[f"mean_{name}"] = InterpolatedUnivariateSpline(
                cls.knots[name], pars[f"mean_{name}"], k=3
            )
            spls[f"ln_std_{name}"] = InterpolatedUnivariateSpline(
                cls.knots[name], pars[f"ln_std_{name}"], k=3
            )

        return spls

    @classmethod
    def setup_dists(cls, spls, data):
        dists = {}

        dists["ln_n0"] = spls["ln_n0"]

        dists["phi2"] = dist.TruncatedNormal(
            loc=spls["mean_phi2"](data["phi1"]),
            scale=jnp.exp(spls["ln_std_phi2"](data["phi1"])),
            low=cls.phi2_lim[0],
            high=cls.phi2_lim[1],
        )

        dists["pm1"] = dist.TruncatedNormal(
            loc=spls["mean_pm1"](data["phi1"]),
            scale=jnp.exp(spls["ln_std_pm1"](data["phi1"])),
            low=cls.pm1_lim[0],
            high=cls.pm1_lim[1],
        )

        dists["pm2"] = dist.Normal(
            loc=spls["mean_pm2"](data["phi1"]),
            scale=jnp.exp(spls["ln_std_pm2"](data["phi1"])),
        )

        return dists

    @classmethod
    def plot_projections(
        cls,
        pars,
        grids=None,
        axes=None,
        label=True,
        plot_stream_knots=True,
        **kwargs,
    ):
        fig, axes = super().plot_projections(
            pars, grids=grids, axes=axes, label=label, **kwargs
        )

        for i, name in enumerate(["phi2", "pm1", "pm2"]):
            if plot_stream_knots:
                axes[i].scatter(
                    cls.knots[name],
                    pars[f"mean_{name}"],
                    color="tab:green",
                )

        return fig, axes

    @classmethod
    def setup_other_priors(cls, spls):
        lp = 0.0

        # Smoothness priors
        smooth = {
            "ln_n0": 1.0 / 30.0,
            "mean_phi2": 1.0 / 10.0,
            "ln_std_phi2": 0.5 / 10.0,
            "mean_pm1": 1.0 / 10.0,
            "ln_std_pm1": 0.5 / 10.0,
            "mean_pm2": 1.0 / 10.0,
            "ln_std_pm2": 0.5 / 10.0,
        }
        for param_name, std in smooth.items():
            knots = cls.knots[cls.params_to_knots[param_name]]
            deriv = spls[param_name].derivative(knots)
            lp = lp + dist.Normal(0.0, std).log_prob(deriv).sum()

        numpyro.factor(f"smooth_{cls.name}", lp)
        return lp
