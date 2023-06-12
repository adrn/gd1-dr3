from functools import partial

import jax
import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from gd1_helpers.membership import JointModel, Model
from gd1_helpers.membership.helpers import ln_normal, two_norm_mixture_ln_prob


class RVStreamModel(Model):
    name = "stream"

    rv_knots = jnp.linspace(-110, 30, 7)
    param_names = {
        "mean": len(rv_knots),
        "ln_std": len(rv_knots),
    }
    param_bounds = {
        "mean": (-800, 800),
        "ln_std": (-5, 1),
    }

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def rv(cls, data, pars):
        """ln_likelihood for rv"""
        mean_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["mean"], k=3)
        ln_std_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["ln_std"], k=3)
        var = jnp.exp(2 * ln_std_spl(data["phi1"]))
        return ln_normal(
            data["rv"],
            mean_spl(data["phi1"]),
            var + data["rv_error"] ** 2,
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, pars, data):
        return cls.rv(data, pars)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, pars):
        lp = 0.0

        prior_stds = {
            "mean_rv": 100.0,
            "ln_std_rv": 0.5,
        }
        for name, size in cls.param_names.items():
            if name not in prior_stds:
                continue

            for i in range(1, size):
                lp += ln_normal(pars[name][i], pars[name][i - 1], prior_stds[name])

        return lp


class RVBackgroundModel(Model):
    name = "background"

    rv_knots = jnp.linspace(-110, 30, 7)
    param_names = {
        "arctanh_w": len(rv_knots),
        "mean1": len(rv_knots),
        "mean2": len(rv_knots),
        "ln_std1": len(rv_knots),
        "ln_std2": len(rv_knots),
    }
    param_bounds = {
        "arctanh_w": (-100, 100),
        "mean1": (-250, -130),
        "mean2": (-100, 100),
        "ln_std1": (-5, 5),
        "ln_std2": (-5, 8),
    }

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def rv(cls, data, pars):
        """ln_likelihood for rv"""
        arctanh_w_spl = InterpolatedUnivariateSpline(
            cls.rv_knots, pars["arctanh_w"], k=3
        )
        mean1_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["mean1"], k=3)
        ln_std1_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["ln_std1"], k=3)
        mean2_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["mean2"], k=3)
        ln_std2_spl = InterpolatedUnivariateSpline(cls.rv_knots, pars["ln_std2"], k=3)

        w = 0.5 * (jnp.tanh(arctanh_w_spl(data["phi1"])) + 1)
        p = jnp.stack(
            (
                w,
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
        return cls.rv(data, pars)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, pars):
        lp = 0.0

        prior_stds = {
            "mean1": 50.0,
            "ln_std1": 0.5,
            "mean2": 50.0,
            "ln_std2": 0.5,
        }
        for name, size in cls.param_names.items():
            if name not in prior_stds:
                continue

            for i in range(1, size):
                lp += ln_normal(pars[name][i], pars[name][i - 1], prior_stds[name] ** 2)

        lp += (-jnp.log(jnp.cosh(pars["arctanh_w"]) ** 2) - jnp.log(2)).sum()

        return lp


class RVMixtureModel(JointModel):
    components = {
        ModelComponent.name: ModelComponent
        for ModelComponent in [RVBackgroundModel, RVStreamModel]
    }

    w_mix_knots = jnp.arange(-110, 30 + 1e-3, 30)
    N_survey = None  # TODO: must be set in notebook
    param_names = {
        "arctanh_w_mix": len(w_mix_knots),
        "rv0": None,  # TODO: must be set in notebook - survey offsets
        "ln_extra_err": None,  # TODO: must be set in notebook - error inflation
    }
    param_bounds = {
        "arctanh_w_mix": (-100, 100),
        "rv0": (-100, 100),
        "ln_extra_err": (-5, 4.5),
    }
    for component_name, ModelComponent in components.items():
        for k, v in ModelComponent.param_names.items():
            param_names[k + f"_{ModelComponent.name}"] = v

    # print(f"model has {sum(param_names.values())} parameters")

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def preprocess_data(cls, flat_pars, data):
        new_data = {}
        new_data["phi1"] = data["phi1"]

        condns = [data["survey_id"] == i for i in range(cls.N_survey)]
        new_data["rv"] = jnp.select(
            condns, [data["rv"] - flat_pars["rv0"][i] for i in range(cls.N_survey)]
        )
        new_data["rv_error"] = jnp.select(
            condns,
            [
                jnp.sqrt(
                    data["rv_error"] ** 2 + jnp.exp(2 * flat_pars["ln_extra_err"][i])
                )
                for i in range(cls.N_survey)
            ],
        )
        return new_data

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_likelihood(cls, flat_pars, data):
        component_pars = cls.unpack_component_pars(flat_pars)
        data = cls.preprocess_data(flat_pars, data)

        lls = []
        for name, Component in cls.components.items():
            lln = Component.ln_likelihood(component_pars[name], data)
            lls.append(lln)

        arctanh_w_spl = InterpolatedUnivariateSpline(
            cls.w_mix_knots, flat_pars["arctanh_w_mix"], k=3
        )
        w = 0.5 * (jnp.tanh(arctanh_w_spl(data["phi1"])) + 1)
        return jnp.logaddexp(jnp.log(w) + lls[0], jnp.log(1 - w) + lls[1])

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_prior(cls, flat_pars):
        component_pars = cls.unpack_component_pars(flat_pars)

        lp = 0.0
        for name, Component in cls.components.items():
            lp += Component.ln_prior(component_pars[name])

        lp += (-jnp.log(jnp.cosh(flat_pars["arctanh_w_mix"]) ** 2) - jnp.log(2)).sum()
        lp += ln_normal(flat_pars["rv0"], 0, 25.0).sum()

        return lp

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def ln_posterior(cls, pars_arr, data, *args):
        pars = cls.unpack_pars(pars_arr)
        ln_L = cls.ln_likelihood(pars, data, *args)
        return ln_L.sum() + cls.ln_prior(pars)
