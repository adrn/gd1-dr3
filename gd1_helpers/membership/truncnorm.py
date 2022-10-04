import scipy.stats as osp_stats
from jax import lax
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy.lax_numpy import _promote_args_inexact
from jax._src.numpy.util import _wraps
from jax.scipy import special, stats


def _log_diff(x, y):
    return special.logsumexp(
        jnp.array([x, y]), b=jnp.array([jnp.ones_like(x), -jnp.ones_like(y)]), axis=0
    )


def _log_gauss_mass(a, b):
    """Log of Gaussian probability mass within an interval"""
    a, b = jnp.array(a), jnp.array(b)
    a, b = jnp.broadcast_arrays(a, b)

    # Note: Docstring carried over from scipy
    # Calculations in right tail are inaccurate, so we'll exploit the
    # symmetry and work only in the left tail
    case_left = b <= 0
    case_right = a > 0
    case_central = ~(case_left | case_right)

    def mass_case_left(a, b):
        return _log_diff(special.log_ndtr(b), special.log_ndtr(a))

    def mass_case_right(a, b):
        return mass_case_left(-b, -a)

    def mass_case_central(a, b):
        # Note: Docstring carried over from scipy
        # Previously, this was implemented as:
        # left_mass = mass_case_left(a, 0)
        # right_mass = mass_case_right(0, b)
        # return _log_sum(left_mass, right_mass)
        # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
        # Correct for this with an alternative formulation.
        # We're not concerned with underflow here: if only one term
        # underflows, it was insignificant; if both terms underflow,
        # the result can't accurately be represented in logspace anyway
        # because sc.log1p(x) ~ x for small x.
        return jnp.log1p(-special.ndtr(a) - special.ndtr(-b))

    out = jnp.select(
        [case_left, case_right, case_central],
        [mass_case_left(a, b), mass_case_right(a, b), mass_case_central(a, b)],
    )
    return out


@_wraps(osp_stats.truncnorm.logpdf, update_doc=False)
def logpdf(x, a, b, loc=0, scale=1):
    x, a, b, loc, scale = _promote_args_inexact("truncnorm.logpdf", x, a, b, loc, scale)
    val = lax.sub(stats.norm.logpdf(x, loc, scale), _log_gauss_mass(a, b))

    x_scaled = lax.div(lax.sub(x, loc), scale)
    val = jnp.where((x_scaled < a) | (x_scaled > b), -jnp.inf, val)
    return val


@_wraps(osp_stats.truncnorm.pdf, update_doc=False)
def pdf(x, a, b, loc=0, scale=1):
    return lax.exp(logpdf(x, a, b, loc, scale))


@_wraps(osp_stats.truncnorm.logsf, update_doc=False)
def logsf(x, a, b, loc=0, scale=1):
    x, a, b, loc, scale = _promote_args_inexact("truncnorm.logsf", x, a, b, loc, scale)
    x, a, b = jnp.broadcast_arrays(x, a, b)
    x = lax.div(lax.sub(x, loc), scale)
    logsf = _log_gauss_mass(x, b) - _log_gauss_mass(a, b)
    logcdf = _log_gauss_mass(a, x) - _log_gauss_mass(a, b)

    logsf = jnp.select(
        # third condition: avoid catastrophic cancellation (from scipy)
        [x >= b, x <= a, logsf > -0.1, x > a],
        [-jnp.inf, 0, jnp.log1p(-jnp.exp(logcdf)), logsf],
    )
    return logsf


@_wraps(osp_stats.truncnorm.sf, update_doc=False)
def sf(x, a, b, loc=0, scale=1):
    return lax.exp(logsf(x, a, b, loc, scale))


@_wraps(osp_stats.truncnorm.logcdf, update_doc=False)
def logcdf(x, a, b, loc=0, scale=1):
    x, a, b, loc, scale = _promote_args_inexact("truncnorm.logcdf", x, a, b, loc, scale)
    x, a, b = jnp.broadcast_arrays(x, a, b)
    x = lax.div(lax.sub(x, loc), scale)
    logcdf = _log_gauss_mass(a, x) - _log_gauss_mass(a, b)
    logsf = _log_gauss_mass(x, b) - _log_gauss_mass(a, b)

    logcdf = jnp.select(
        # third condition: avoid catastrophic cancellation (from scipy)
        [x >= b, x <= a, logcdf > -0.1, x > a],
        [0, -jnp.inf, jnp.log1p(-jnp.exp(logsf)), logcdf],
    )
    return logcdf


@_wraps(osp_stats.truncnorm.cdf, update_doc=False)
def cdf(x, a, b, loc=0, scale=1):
    return lax.exp(logcdf(x, a, b, loc, scale))
