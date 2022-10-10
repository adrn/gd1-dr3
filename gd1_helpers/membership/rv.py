from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from gd1_helpers.membership import Model  # , JointModel
from gd1_helpers.membership.helpers import (
    ln_normal,
    ln_simpson,
    ln_truncated_normal,
    ln_uniform,
    two_norm_mixture_ln_prob,
)


class RVStreamModel(Model):
    name = "stream"

    ln_n0_knots = jnp.arange(-110, 30 + 1e-3, 10)  # arange!
    rv_knots = jnp.linspace(-110, 30, 9)
    param_names = {
        "ln_n0": len(ln_n0_knots),
        "mean_rv": len(rv_knots),
        "ln_std_rv": len(rv_knots),
    }
    param_bounds = {
        "ln_n0": (-10, 5),
        "mean_rv": (-800, 800),
        "ln_std_rv": (-5, 2),
    }

    integ_grid_phi1 = jnp.arange(-100, 20 + 1e-3, 0.2)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_n0(cls, phi1, pars):
        ln_n0_spl = InterpolatedUnivariateSpline(cls.ln_n0_knots, pars["ln_n0"], k=3)
        return ln_n0_spl(phi1)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def rv(cls, data, pars):
        """ln_likelihood for rv"""
        mean_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["mean_rv"], k=3)
        ln_std_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["ln_std_rv"], k=3)
        return ln_normal(
            data["rv"],
            mean_spl(data["phi1"]),
            jnp.exp(2 * ln_std_spl(data["phi1"])) + data["rv_error"] ** 2,
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, pars, data):
        ln_dens = cls.ln_n0(data["phi1"], pars) + cls.rv(data, pars)
        ln_dens_grid = cls.ln_n0(cls.integ_grid_phi1, pars)
        ln_V = ln_simpson(ln_dens_grid, x=cls.integ_grid_phi1)

        return (ln_V, ln_dens)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, pars):
        lp = 0.0

        #         prior_stds = {
        #             "mean_rv": 50.,
        #             "ln_std_rv": 0.5,
        #         }
        #         for name, size in cls.param_names.items():
        #             if name not in prior_stds:
        #                 continue

        #             for i in range(1, size):
        #                 lp += ln_normal(pars[name][i], pars[name][i - 1], prior_stds[name])

        #         for name in ["mean_rv", "ln_std_rv"]:
        #             lp += ln_uniform(pars[name], *cls.param_bounds[name]).sum()

        return lp


class RVBackgroundModel(Model):
    name = "background"

    ln_n0_knots = jnp.arange(-110, 30 + 1e-3, 30)  # arange!
    rv_knots = jnp.linspace(-110, 30, 7)
    param_names = {
        "ln_n0": len(ln_n0_knots),
        "w_rv": len(rv_knots),
        "mean_rv1": len(rv_knots),
        "mean_rv2": len(rv_knots),
        "ln_std_rv1": len(rv_knots),
        "ln_std_rv2": len(rv_knots),
    }
    param_bounds = {
        "ln_n0": (-10, 3),
        "w_rv": (0, 1),
        "mean_rv1": (-500, 500),
        "mean_rv2": (-500, 500),
        "ln_std_rv1": (-5, 5),
        "ln_std_rv2": (-5, 8),
    }
    integ_grid_phi1 = jnp.arange(-100, 20 + 1e-3, 0.2)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_n0(cls, phi1, pars):
        ln_n0_spl = InterpolatedUnivariateSpline(cls.ln_n0_knots, pars["ln_n0"], k=3)
        return ln_n0_spl(phi1)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def rv(cls, data, pars):
        """ln_likelihood for rv"""
        w_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["w_rv"], k=3)
        mean1_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["mean_rv1"], k=3)
        ln_std1_spl = InterpolatedUnivariateSpline(
            cls.rv_knots, pars["ln_std_rv1"], k=3
        )
        mean2_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["mean_rv2"], k=3)
        ln_std2_spl = InterpolatedUnivariateSpline(
            cls.rv_knots, pars["ln_std_rv2"], k=3
        )

        p = jnp.stack(
            (
                w_spl(data["phi1"]),
                mean1_spl(data["phi1"]),
                mean2_spl(data["phi1"]),
                ln_std1_spl(data["phi1"]),
                ln_std2_spl(data["phi1"]),
            )
        )
        return two_norm_mixture_ln_prob(p, data["rv"], data["rv_error"])

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, pars, data):
        ln_dens = cls.ln_n0(data["phi1"], pars) + cls.rv(data, pars)
        ln_dens_grid = cls.ln_n0(cls.integ_grid_phi1, pars)
        ln_V = ln_simpson(ln_dens_grid, x=cls.integ_grid_phi1)

        return (ln_V, ln_dens)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, pars):
        lp = 0.0

        #         prior_stds = {
        #             "mean_rv1": 30.0,
        #             "ln_std_rv1": 0.5,
        #             "mean_rv2": 30.0,
        #             "ln_std_rv2": 0.5,
        #         }
        #         for name, size in cls.param_names.items():
        #             if name not in prior_stds:
        #                 continue

        #             for i in range(1, size):
        #                 lp += ln_normal(pars[name][i], pars[name][i - 1], prior_stds[name])

        #         for name in [
        #             "w_rv",
        #             "mean_rv1",
        #             "ln_std_rv1",
        #             "mean_rv2",
        #             "ln_std_rv2",
        #         ]:
        #             lp += ln_uniform(pars[name], *cls.param_bounds[name]).sum()

        return lp


class RVOffsetModel(Model):
    name = "offset"

    param_names = {"rv0": 6}  # TODO: hard-coded
    param_bounds = {"rv0": (-50, 50)}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, pars, data):
        return -jnp.inf, jnp.full_like(data["phi1"], -jnp.inf)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, pars):
        return ln_normal(pars["rv0"], 0, 15.0).sum()


class RVJointModel(Model):
    components = {
        ModelComponent.name: ModelComponent
        for ModelComponent in [RVBackgroundModel, RVStreamModel, RVOffsetModel]
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
        for name in ["mean_pm1", "ln_std_pm1", "mean_pm2", "ln_std_pm2"]:
            if "stream" in pars and name in pars["stream"]:
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
    def preprocess_data(cls, rv_offsets, data):
        new_data = deepcopy(data)
        new_data["rv"] = jnp.select(
            [data["survey_id"] == i for i in range(len(rv_offsets))],
            [data["rv"] + rv_offset for rv_offset in rv_offsets],
        )
        return new_data

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, flat_pars, data):
        component_pars = cls.unpack_component_pars(flat_pars)

        new_data = cls.preprocess_data(component_pars["offset"]["rv0"], data)

        ln_Vs = []
        lls = []
        for name, Component in cls.components.items():
            if name == "offset":
                continue
            ln_Vn, lln = Component.ln_likelihood(component_pars[name], new_data)
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
