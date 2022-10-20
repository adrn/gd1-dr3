import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp

from .background import BackgroundModel
from .base import Model
from .helpers import ln_simpson
from .spur import SpurModel
from .stream import StreamModel

__all__ = ["JointModel"]


class JointModel(Model):
    components = {
        ModelComponent.name: ModelComponent
        for ModelComponent in [BackgroundModel, StreamModel, SpurModel]
    }

    @classmethod
    def setup_pars(cls):
        pars = {}
        for comp_name, Component in cls.components.items():
            pars[comp_name] = Component.setup_pars()
        return pars

    @classmethod
    def setup_splines(cls, pars):
        spls = {}
        for comp_name, Component in cls.components.items():
            spls[comp_name] = Component.setup_splines(pars[comp_name])
        return spls

    @classmethod
    def setup_dists(cls, spls, data):
        dists = {}
        for comp_name, Component in cls.components.items():
            if comp_name == "spur":
                dists[comp_name] = Component.setup_dists(
                    spls[comp_name], data, spls["stream"]
                )
            else:
                dists[comp_name] = Component.setup_dists(spls[comp_name], data)

        return dists

    @classmethod
    def setup_obs(cls, dists, data):
        ln_Vs = {}
        ln_n0s = {}
        for name, comp_dists in dists.items():
            grid = cls.components[name].integ_grid_phi1
            ln_Vs[name] = ln_simpson(comp_dists["ln_n0"](grid), x=grid)
            ln_n0s[name] = comp_dists["ln_n0"](data["phi1"])

        ln_V = logsumexp(jnp.array(ln_Vs.values()))
        ln_n0 = logsumexp(jnp.array(ln_n0s.values()), axis=0)
        numpyro.factor(
            "obs_ln_n0",
            -jnp.exp(ln_V) + ln_n0.sum(),
        )

        # Density model
        # TODO: for plotting, need to hack this stuff...
        mix = dist.Categorical(
            probs=jnp.array(
                [
                    jnp.exp(ln_n0s["background"] - ln_n0),
                    jnp.exp(ln_n0s["stream"] - ln_n0),
                    jnp.exp(ln_n0s["spur"] - ln_n0),
                ]
            ).T
        )

        for k in ["phi2", "pm1", "pm2"]:
            numpyro.sample(
                f"obs_{k}",
                dist.MixtureGeneral(
                    mix,
                    [
                        dists["background"][k],
                        dists["stream"][k],
                        dists["spur"][k],
                    ],
                ),
                obs=data[k],
            )
