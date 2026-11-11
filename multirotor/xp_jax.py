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
    minimum, sin, sqrt, stack, tan
)


_cache.configure_jax(jax)



def _as_arrays(tree: Any) -> Any:
    """
    Make the arrays of a tree JAX arrays. numpy arrays can't be indexed by a traced loop
    index; JAX arrays can. Python scalars and functions are left as they are, so that counts
    stay static.

    Parameters
    ----------
    tree : Any
        A pytree, such as a named tuple.

    Returns
    -------
    Any
        The same tree, with its arrays as JAX arrays.
    """
    keep = (int, float, bool)
    return jax.tree_util.tree_map(
        lambda a: a if callable(a) or isinstance(a, keep) else jnp.asarray(a), tree)



def sqrt_nonneg(x: jax.Array) -> jax.Array:
    """
    Square root that is zero for negative numbers, with a finite gradient.

    Parameters
    ----------
    x : jax.Array
        The number.

    Returns
    -------
    jax.Array
        The square root of ``x``, or 0 where ``x`` is negative.
    """
    positive = x > 0.
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.)), 0.)



def select(pred: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Choose between two scalars.

    Parameters
    ----------
    pred : jax.Array
        The condition.
    a, b : jax.Array
        The values.

    Returns
    -------
    jax.Array
        ``a`` if ``pred`` else ``b``.
    """
    return jnp.where(pred, a, b)



def map_(f: Callable[[int, Any], jax.Array], n: int, consts: Any) -> jax.Array:
    """
    Evaluate a function for each index, vectorized.

    Parameters
    ----------
    f : callable
        ``f(i, consts)``, which returns a float.
    n : int
        The number of indices.
    consts : Any
        What ``f`` needs besides the index, usually a named tuple.

    Returns
    -------
    jax.Array
        ``f(i, consts)`` for ``i`` in ``range(n)``, shape (n,).
    """
    consts = _as_arrays(consts)
    return jax.vmap(lambda i: f(i, consts))(jnp.arange(n))



def map_rows(f: Callable[[int, Any], tuple], n: int, consts: Any) -> jax.Array:
    """
    Evaluate a function that returns several floats for each index, vectorized.

    Parameters
    ----------
    f : callable
        ``f(i, consts)``, which returns a tuple of k floats, the same k each time.
    n : int
        The number of indices.
    consts : Any
        What ``f`` needs besides the index, usually a named tuple.

    Returns
    -------
    jax.Array
        The k floats that ``f(i, consts)`` returns for ``i`` in ``range(n)``, one column
        per ``i``, shape (k, n).
    """
    consts = _as_arrays(consts)
    return jnp.stack(jax.vmap(lambda i: f(i, consts))(jnp.arange(n)))



def with_row(x: jax.Array, j: int, row: jax.Array) -> jax.Array:
    """
    Replace a row of a matrix, without changing the matrix.

    Parameters
    ----------
    x : jax.Array
        The matrix.
    j : int
        The row.
    row : jax.Array
        The new row.

    Returns
    -------
    jax.Array
        A copy of ``x`` whose row ``j`` is ``row``.
    """
    return x.at[j].set(row)



def fold(f: Callable[[Any, int, Any], Any], init: Any, n: int, consts: Any) -> Any:
    """
    Accumulate over the indices ``0 .. n - 1``.

    Parameters
    ----------
    f : callable
        ``f(acc, i, consts)``, which returns the new accumulator.
    init : Any
        The accumulator before the first index.
    n : int
        The number of indices.
    consts : Any
        What ``f`` needs besides the accumulator and the index.

    Returns
    -------
    Any
        The accumulator after the last index, with the structure of ``init``.
    """
    consts = _as_arrays(consts)
    return lax.scan(lambda acc, i: (f(acc, i, consts), None), init, jnp.arange(n))[0]



def scan(
    f: Callable[[Any, int, Any], Tuple[Any, jax.Array]], init: Any, n: int, consts: Any
) -> Tuple[Any, jax.Array]:
    """
    Accumulate over the indices, and keep an output of every step.

    Parameters
    ----------
    f : callable
        ``carry, y = f(carry, i, consts)``. Each ``y`` is a 1-D array of the same length.
    init : Any
        The carry before the first index.
    n : int
        The number of indices.
    consts : Any
        What ``f`` needs besides the carry and the index.

    Returns
    -------
    carry : Any
        The carry after the last index.
    ys : jax.Array
        The ``y`` of every step, shape (n, k).
    """
    consts = _as_arrays(consts)
    return lax.scan(lambda carry, i: f(carry, i, consts), init, jnp.arange(n))



def cond(pred: jax.Array, f: Callable[[Any], Any], g: Callable[[Any], Any], operand: Any) -> Any:
    """
    Run one of two functions.

    Parameters
    ----------
    pred : jax.Array
        The condition.
    f, g : callable
        Functions of the operand. They return the same structure.
    operand : Any
        What they take, usually a named tuple.

    Returns
    -------
    Any
        ``f(operand)`` if ``pred`` else ``g(operand)``.
    """
    return lax.cond(pred, f, g, operand)



def root(
    f: Callable[[jax.Array, Any], jax.Array], low: float, high: float, tol: float, args: Any
) -> jax.Array:
    """
    Root of an increasing function, by bisection. The derivative with respect to ``args``
    comes from the implicit function theorem, not from differentiating the iterations.

    Parameters
    ----------
    f : callable
        ``f(x, args)``, increasing in ``x``, with ``f(low) < 0 <= f(high)``.
    low, high : float
        The bracket.
    tol : float
        The width of the bracket at which to stop.
    args : Any
        What ``f`` needs besides ``x``.

    Returns
    -------
    jax.Array
        The root, to within ``tol``.
    """
    def solve(fn: Callable[[jax.Array], jax.Array], x0: jax.Array) -> jax.Array:
        """
        The bisection, for `lax.custom_root`.

        Parameters
        ----------
        fn : callable
            ``f`` with ``args`` bound.
        x0 : jax.Array
            The initial guess, which this ignores.

        Returns
        -------
        jax.Array
            The root.
        """
        def body(bounds: Tuple[jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
            """
            One bisection step.

            Parameters
            ----------
            bounds : tuple of jax.Array
                The bracket.

            Returns
            -------
            tuple of jax.Array
                The half of the bracket that holds the root.
            """
            lo, hi = bounds
            middle = 0.5 * (lo + hi)
            below = fn(middle) < 0.
            return jnp.where(below, middle, lo), jnp.where(below, hi, middle)
        lo, hi = lax.while_loop(lambda b: b[1] - b[0] > tol, body, (low, high))
        return 0.5 * (lo + hi)
    return lax.custom_root(
        lambda x: f(x, args), 0.5 * (low + high), solve, lambda g, y: y / g(1.))
