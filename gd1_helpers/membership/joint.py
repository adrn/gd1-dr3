from functools import partial

import jax

from . import BackgroundModel, Model, StreamModel

__all__ = ["JointModel"]


class JointModel(Model):
    components = {
        ModelComponent.name: ModelComponent
        for ModelComponent in [BackgroundModel, StreamModel]
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
        # TODO: this should save the arrays, and there should be a way of getting just
        # the prob density not the poisson likelihood (for membership probabilities)
        component_pars = cls.unpack_component_pars(flat_pars)
        # lls = {}
        # for name, Component in cls.components.items():
        #     lls[name] = Component.ln_likelihood(component_pars[name], data)

        ll = 0.0
        for name, Component in cls.components.items():
            ll += Component.ln_likelihood(component_pars[name], data)
        return ll

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, flat_pars):
        component_pars = cls.unpack_component_pars(flat_pars)
        # lps = {}
        # for name, Component in cls.components.items():
        #     lps[name] = Component.ln_prior(component_pars[name])

        lp = 0.0
        for name, Component in cls.components.items():
            lp += Component.ln_prior(component_pars[name])
        return lp
