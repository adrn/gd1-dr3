import jax
import jax.numpy as jnp
import jax.scipy as jsci

__all__ = ['ln_normal', 'ln_simpson']


@jax.jit
def ln_normal(x, mu, var):
    """Evaluate the log-normal probability"""
    return -0.5 * (jnp.log(2 * jnp.pi * var) + (x - mu) ** 2 / var)


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
    w, mu1, mu2, ln_s1, ln_s2 = p
    var1 = jnp.exp(2 * ln_s1) + data_err**2
    var2 = var1 + jnp.exp(2 * ln_s2) + data_err**2

    ln_term1 = ln_normal(data, mu1, var1)
    ln_term2 = ln_normal(data, mu2, var2)

    ln_prob = jnp.logaddexp(
        ln_term1 + jnp.log(w),
        ln_term2 + jnp.log(1 - w)
    )

    return ln_prob


@jax.jit
def two_truncnorm_mixture_ln_prob(p, data, data_err, lower, upper):
    w, mu1, mu2, ln_s1, ln_s2 = p
    std1 = jnp.sqrt(jnp.exp(2 * ln_s1) + data_err**2)
    std2 = jnp.sqrt(std1**2 + jnp.exp(2 * ln_s2) + data_err**2)

    ln_term1 = truncnorm_logpdf(data, mu1, std1, (lower - mu1) / std1, (upper - mu1) / std1)
    ln_term2 = truncnorm_logpdf(data, mu2, std2, (lower - mu2) / std2, (upper - mu2) / std2)

    ln_prob = jnp.logaddexp(
        ln_term1 + jnp.log(w),
        ln_term2 + jnp.log(1 - w)
    )

    return ln_prob