from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .background import BackgroundModel
from .base import Model
from .spur import SpurModel
from .stream import StreamModel

__all__ = ["JointModel"]


class JointModel(Model):
    components = {
        ModelComponent.name: ModelComponent
        for ModelComponent in [BackgroundModel, StreamModel, SpurModel]
    }

    param_names = {}
    for component_name, ModelComponent in components.items():
        for k, v in ModelComponent.param_names.items():
            param_names[k + f"_{ModelComponent.name}"] = v

    print(f"model has {sum(param_names.values())} parameters")

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_component_pars(cls, flat_pars):
        pars = {}
        for component_name in cls.components:
            tmp = {}
            for k, v in flat_pars.items():
                if k.endswith(component_name):
                    tmp[k[: -(len(component_name) + 1)]] = v
            pars[component_name] = tmp

        # HACK:
        for name in ["mean_pm1", "ln_std_pm1", "mean_pm2", "ln_std_pm2", "mean_plx"]:
            pars["spur"][name] = pars["stream"][name]

        return pars

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pack_component_pars(cls, pars):
        flat_pars = {}
        for component_name in cls.components:
            for k, v in pars[component_name].items():
                flat_pars[k + f"_{component_name}"] = v
            pass
        return flat_pars

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, flat_pars, data):
        component_pars = cls.unpack_component_pars(flat_pars)

        ln_Vs = []
        lls = []
        for name, Component in cls.components.items():
            ln_Vn, lln = Component.ln_likelihood(component_pars[name], data)
            ln_Vs.append(ln_Vn)
            lls.append(lln)
        return logsumexp(jnp.array(ln_Vs)), logsumexp(jnp.array(lls), axis=0)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, flat_pars):
        component_pars = cls.unpack_component_pars(flat_pars)

        lp = 0.0
        for name, Component in cls.components.items():
            lp += Component.ln_prior(component_pars[name])
        return lp
