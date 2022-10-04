from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["Model"]


class Model:
    param_names = {}

    # TODO: could implement __init_subclass__() to enforce that subclasses have the
    # necessary ln_likelihood(), ln_prior(), ...

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """
        This function takes a parameter array and unpacks it into a dictionary with the
        parameter names as keys.
        """
        p_dict = {}
        j = 0
        for name, size in cls.param_names.items():
            p_dict[name] = jnp.squeeze(p_arr[j : j + size])
            j += size
        return p_dict

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pack_pars(cls, p_dict):
        """
        This function takes a parameter dictionary and packs it into a JAX array where
        the order is set by the parameter name list defined on the class.
        """
        p_arrs = []
        for name in cls.param_names.keys():
            p_arrs.append(jnp.atleast_1d(p_dict[name]))
        return jnp.concatenate(p_arrs)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_posterior(cls, pars_arr, data, *args):
        pars = cls.unpack_pars(pars_arr)
        return cls.ln_likelihood(pars, data, *args) + cls.ln_prior(pars)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def objective(cls, pars_arr, N, data, *args):
        """
        We normalize the value by the number of data points so that scipy's minimizers
        don't run into overflow issues with the gradients.
        """
        return -cls.ln_posterior(pars_arr, data, *args) / N
