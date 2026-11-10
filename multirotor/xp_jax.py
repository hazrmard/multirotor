"""
The primitives of `multirotor.xp_numba`, for JAX: the same names, built on
``jax.lax`` so that they trace and differentiate.
"""
import jax
import jax.numpy as jnp
from jax import lax

from . import _cache
from jax.numpy import (
    abs, array, concatenate, cos, cross, interp, logical_not, maximum,
    minimum, sin, sqrt, stack, tan
)


_cache.configure_jax(jax)



def _as_arrays(tree):
    """
    numpy arrays can't be indexed by a traced loop index; JAX arrays can. Python
    scalars and functions are left as they are, so that counts stay static.
    """
    keep = (int, float, bool)
    return jax.tree_util.tree_map(
        lambda a: a if callable(a) or isinstance(a, keep) else jnp.asarray(a), tree)



def sqrt_nonneg(x):
    "Square root of ``x``, or 0 where ``x`` is negative, with a finite gradient."
    positive = x > 0.
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.)), 0.)



def select(pred, a, b):
    "``a`` if ``pred`` else ``b``, for scalars."
    return jnp.where(pred, a, b)



def map_(f, n, consts):
    "Array of ``f(i, consts)`` for ``i`` in ``range(n)``, vectorized."
    consts = _as_arrays(consts)
    return jax.vmap(lambda i: f(i, consts))(jnp.arange(n))



def map_rows(f, n, consts):
    "Array, shape (k, n), of the k floats that ``f(i, consts)`` returns, for ``i`` in ``range(n)``."
    consts = _as_arrays(consts)
    return jnp.stack(jax.vmap(lambda i: f(i, consts))(jnp.arange(n)))



def with_row(x, j, row):
    "A copy of the matrix ``x`` whose row ``j`` is ``row``."
    return x.at[j].set(row)



def fold(f, init, n, consts):
    "``acc = f(acc, i, consts)`` for ``i`` in ``range(n)``, starting from ``init``."
    consts = _as_arrays(consts)
    return lax.scan(lambda acc, i: (f(acc, i, consts), None), init, jnp.arange(n))[0]



def scan(f, init, n, consts):
    """
    ``carry, y = f(carry, i, consts)`` for ``i`` in ``range(n)``, starting from
    ``init``. Returns the last carry and the ``y`` of every step, shape (n, k).
    """
    consts = _as_arrays(consts)
    return lax.scan(lambda carry, i: f(carry, i, consts), init, jnp.arange(n))



def cond(pred, f, g, operand):
    "``f(operand)`` if ``pred`` else ``g(operand)``."
    return lax.cond(pred, f, g, operand)



def root(f, low, high, tol, args):
    """
    Root of an increasing ``f(x, args)`` in ``[low, high]`` by bisection, to
    within ``tol``. The derivative with respect to ``args`` comes from the
    implicit function theorem, not from differentiating the iterations.
    """
    def solve(fn, x0):
        def body(bounds):
            lo, hi = bounds
            middle = 0.5 * (lo + hi)
            below = fn(middle) < 0.
            return jnp.where(below, middle, lo), jnp.where(below, hi, middle)
        lo, hi = lax.while_loop(lambda b: b[1] - b[0] > tol, body, (low, high))
        return 0.5 * (lo + hi)
    return lax.custom_root(
        lambda x: f(x, args), 0.5 * (low + high), solve, lambda g, y: y / g(1.))
