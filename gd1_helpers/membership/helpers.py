import jax
import jax.numpy as jnp
import jax.scipy as jsci

from .truncnorm import logpdf as truncnorm_logpdf

__all__ = ["ln_uniform", "ln_normal", "ln_truncated_normal", "ln_simpson"]


@jax.jit
def ln_uniform(x, lower, upper):
    """Evaluate the log-uniform probability"""
    vals = jnp.full_like(x, -jnp.log(upper - lower))
    vals = jnp.where((x > upper) | (x < lower), -jnp.inf, vals)
    return vals


@jax.jit
def ln_normal(x, mu, var):
    """Evaluate the log-normal probability"""
    return -0.5 * (jnp.log(2 * jnp.pi * var) + (x - mu) ** 2 / var)


@jax.jit
def ln_truncated_normal(x, mu, var, lower=-jnp.inf, upper=jnp.inf):
    """Evaluate the log of a truncated normal probability"""
    std = jnp.sqrt(var)
    return truncnorm_logpdf(
        x, loc=mu, scale=std, a=(lower - mu) / std, b=(upper - mu) / std
    )


@jax.jit
def ln_simpson(ln_y, x):
    """
    Evaluate the log of the definite integral of a function evaluated on a grid using
    Simpson's rule
    """

    dx = jnp.diff(x)[0]
    num_points = len(x)
    if num_points // 2 == num_points / 2:
        raise ValueError("Because of laziness, the input size must be odd")

    weights_first = jnp.asarray([1.0])
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0]), [(num_points - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0])
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)

    return jsci.special.logsumexp(ln_y + jnp.log(weights), axis=-1) + jnp.log(dx / 3)


@jax.jit
def two_norm_mixture_ln_prob(p, data, data_err):
    """
    A mixture of two normal distributions convolved with a normal uncertainty
    distribution.

        p = (w, mu_1, mu_2, ln_s1, ln_s2)

    where

        variance1 = exp(2 * ln_s1)
        variance2 = variance1 + exp(2 * ln_s2)

    so that the 2nd component is always wider. `w` is the mixture weight.

    """
    w, mu1, mu2, ln_s1, ln_s2 = p
    var1 = jnp.exp(2 * ln_s1) + data_err**2
    var2 = var1 + jnp.exp(2 * ln_s2) + data_err**2

    ln_term1 = ln_normal(data, mu1, var1)
    ln_term2 = ln_normal(data, mu2, var2)

    ln_prob = jnp.logaddexp(ln_term1 + jnp.log(w), ln_term2 + jnp.log(1 - w))

    return ln_prob


@jax.jit
def two_truncnorm_mixture_ln_prob(p, data, data_err, lower, upper):
    """
    A mixture of two truncated normal distributions, truncated to between `lower` and
    `upper`, convolved with a normal uncertainty distribution.

        p = (w, mu_1, mu_2, ln_s1, ln_s2)

    where

        variance1 = exp(2 * ln_s1)
        variance2 = variance1 + exp(2 * ln_s2)

    so that the 2nd component is always wider. `w` is the mixture weight.

    """
    w, mu1, mu2, ln_s1, ln_s2 = p
    var1 = jnp.exp(2 * ln_s1) + data_err**2
    var2 = var1 + jnp.exp(2 * ln_s2) + data_err**2

    ln_term1 = ln_truncated_normal(data, mu1, var1, lower=lower, upper=upper)
    ln_term2 = ln_truncated_normal(data, mu2, var2, lower=lower, upper=upper)

    ln_prob = jnp.logaddexp(ln_term1 + jnp.log(w), ln_term2 + jnp.log(1 - w))

    return ln_prob
