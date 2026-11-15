"""
The array and control-flow primitives that `multirotor.physics` is written
against, for numba. Math functions are numpy's. Loops and branches are plain
loops and branches that numba compiles, so they cost what hand-written ones do.
`multirotor.xp_jax` has the same names built on ``jax.lax``.

The functions take the body of a loop or branch as an argument. A body must be a
module-level function, and passes what it needs as the ``consts``/``operand``
argument: numba cannot pass a closure as an argument.
"""
import math
from typing import Any, Callable, Tuple

import numpy as np
from numba import njit

from . import _cache
from numpy import (
    abs, array, concatenate, cos, cross, interp, logical_not, maximum,
    minimum, sin, sqrt, stack, tan, zeros
)



CACHE = _cache.configure_numba()



@njit(inline='always', cache=CACHE)
def sqrt_nonneg(x: float) -> float:
    """Square root that is zero for negative numbers."""
    return math.sqrt(max(x, 0.))



@njit(inline='always', cache=CACHE)
def select(pred: bool, a: float, b: float) -> float:
    """Choose between two scalars."""
    return a if pred else b



@njit(inline='always', cache=CACHE)
def map_(f: Callable[[int, Any], float], n: int, consts: Any) -> np.ndarray:
    """
    Evaluate a function for each index.

    Returns: ``f(i, consts)`` for ``i`` in ``range(n)``, shape (n,).
    """
    out = np.empty(n)
    for i in range(n):
        out[i] = f(i, consts)
    return out



@njit(inline='always', cache=CACHE)
def map_rows(f: Callable[[int, Any], tuple], n: int, consts: Any) -> np.ndarray:
    """
    Evaluate a function that returns several floats for each index.

    Returns: The k floats that ``f(i, consts)`` returns for ``i`` in ``range(n)``, one
      column per ``i``, shape (k, n).
    """
    first = f(0, consts)
    out = np.empty((len(first), n))
    for j in range(len(first)):
        out[j, 0] = first[j]
    for i in range(1, n):
        row = f(i, consts)
        for j in range(len(first)):
            out[j, i] = row[j]
    return out



@njit(inline='always', cache=CACHE)
def with_row(x: np.ndarray, j: int, row: np.ndarray) -> np.ndarray:
    """Replace a row of a matrix, without changing the matrix."""
    out = x.copy()
    out[j] = row
    return out



@njit(inline='always', cache=CACHE)
def fold(f: Callable[[Any, int, Any], Any], init: Any, n: int, consts: Any) -> Any:
    """Accumulate over the indices ``0 .. n - 1``."""
    acc = init
    for i in range(n):
        acc = f(acc, i, consts)
    return acc



@njit(inline='always', cache=CACHE)
def scan(
    f: Callable[[Any, int, Any], Tuple[Any, np.ndarray]], init: Any, n: int, consts: Any
) -> Tuple[Any, np.ndarray]:
    """
    Accumulate over the indices, and keep an output of every step.

    Returns: The ``y`` of every step, shape (n, k).
    """
    carry, y = f(init, 0, consts)
    ys = np.empty((n, y.shape[0]))
    ys[0] = y
    for i in range(1, n):
        carry, y = f(carry, i, consts)
        ys[i] = y
    return carry, ys



@njit(inline='always', cache=CACHE)
def cond(pred: bool, f: Callable[[Any], Any], g: Callable[[Any], Any], operand: Any) -> Any:
    """Run one of two functions. Only the chosen one runs."""
    if pred:
        return f(operand)
    return g(operand)



@njit(inline='always', cache=CACHE)
def root(
    f: Callable[[float, Any], float], low: float, high: float, tol: float, args: Any
) -> float:
    """Root of an increasing function, by bisection."""
    while high - low > tol:
        middle = 0.5 * (low + high)
        if f(middle, args) < 0.:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
