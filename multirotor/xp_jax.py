"""
The primitives of `multirotor.xp_numba`, for JAX: the same names, built on
``jax.lax`` so that they trace and differentiate.
"""
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from . import _cache
from jax.numpy import (
    abs, array, concatenate, cos, cross, interp, logical_not, maximum,
    minimum, sin, sqrt, stack, tan, zeros
)


_cache.configure_jax(jax)



def _as_arrays(tree: Any) -> Any:
    """
    Make the arrays of a tree JAX arrays. numpy arrays can't be indexed by a traced loop
    index; JAX arrays can. Python scalars and functions are left as they are, so that counts
    stay static.
    """
    keep = (int, float, bool)
    return jax.tree_util.tree_map(
        lambda a: a if callable(a) or isinstance(a, keep) else jnp.asarray(a), tree)



def sqrt_nonneg(x: jax.Array) -> jax.Array:
    """Square root that is zero for negative numbers, with a finite gradient."""
    positive = x > 0.
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.)), 0.)



def select(pred: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Choose between two scalars."""
    return jnp.where(pred, a, b)



def map_(f: Callable[[int, Any], jax.Array], n: int, consts: Any) -> jax.Array:
    """
    Evaluate a function for each index, vectorized.

    Returns: ``f(i, consts)`` for ``i`` in ``range(n)``, shape (n,).
    """
    consts = _as_arrays(consts)
    return jax.vmap(lambda i: f(i, consts))(jnp.arange(n))



def map_rows(f: Callable[[int, Any], tuple], n: int, consts: Any) -> jax.Array:
    """
    Evaluate a function that returns several floats for each index, vectorized.

    Returns: The k floats that ``f(i, consts)`` returns for ``i`` in ``range(n)``, one
      column per ``i``, shape (k, n).
    """
    consts = _as_arrays(consts)
    return jnp.stack(jax.vmap(lambda i: f(i, consts))(jnp.arange(n)))



def with_row(x: jax.Array, j: int, row: jax.Array) -> jax.Array:
    """Replace a row of a matrix, without changing the matrix."""
    return x.at[j].set(row)



def fold(f: Callable[[Any, int, Any], Any], init: Any, n: int, consts: Any) -> Any:
    """Accumulate over the indices ``0 .. n - 1``."""
    consts = _as_arrays(consts)
    return lax.scan(lambda acc, i: (f(acc, i, consts), None), init, jnp.arange(n))[0]



def scan(
    f: Callable[[Any, int, Any], Tuple[Any, jax.Array]], init: Any, n: int, consts: Any
) -> Tuple[Any, jax.Array]:
    """
    Accumulate over the indices, and keep an output of every step.

    Returns: The ``y`` of every step, shape (n, k).
    """
    consts = _as_arrays(consts)
    return lax.scan(lambda carry, i: f(carry, i, consts), init, jnp.arange(n))



def cond(pred: jax.Array, f: Callable[[Any], Any], g: Callable[[Any], Any], operand: Any) -> Any:
    """Run one of two functions."""
    return lax.cond(pred, f, g, operand)



def root(
    f: Callable[[jax.Array, Any], jax.Array], low: float, high: float, tol: float, args: Any
) -> jax.Array:
    """
    Root of an increasing function, by bisection. The derivative with respect to ``args``
    comes from the implicit function theorem, not from differentiating the iterations.
    """
    def solve(fn: Callable[[jax.Array], jax.Array], x0: jax.Array) -> jax.Array:
        """The bisection, for `lax.custom_root`."""
        def body(bounds: Tuple[jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
            """One bisection step."""
            lo, hi = bounds
            middle = 0.5 * (lo + hi)
            below = fn(middle) < 0.
            return jnp.where(below, middle, lo), jnp.where(below, hi, middle)
        lo, hi = lax.while_loop(lambda b: b[1] - b[0] > tol, body, (low, high))
        return 0.5 * (lo + hi)
    return lax.custom_root(
        lambda x: f(x, args), 0.5 * (low + high), solve, lambda g, y: y / g(1.))
