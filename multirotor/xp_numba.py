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

import numpy as np
from numba import njit
from numpy import (
    abs, array, concatenate, cos, cross, interp, logical_not, maximum,
    minimum, sin, sqrt, stack, tan
)



@njit(inline='always')
def sqrt_nonneg(x):
    "Square root of ``x``, or 0 where ``x`` is negative."
    return math.sqrt(max(x, 0.))



@njit(inline='always')
def select(pred, a, b):
    "``a`` if ``pred`` else ``b``, for scalars."
    return a if pred else b



@njit(inline='always')
def map_(f, n, consts):
    "Array of ``f(i, consts)`` for ``i`` in ``range(n)``. ``f`` returns a float."
    out = np.empty(n)
    for i in range(n):
        out[i] = f(i, consts)
    return out



@njit(inline='always')
def map_rows(f, n, consts):
    "Array, shape (k, n), of the k floats that ``f(i, consts)`` returns, for ``i`` in ``range(n)``."
    first = f(0, consts)
    out = np.empty((len(first), n))
    for j in range(len(first)):
        out[j, 0] = first[j]
    for i in range(1, n):
        row = f(i, consts)
        for j in range(len(first)):
            out[j, i] = row[j]
    return out



@njit(inline='always')
def with_row(x, j, row):
    "A copy of the matrix ``x`` whose row ``j`` is ``row``."
    out = x.copy()
    out[j] = row
    return out



@njit(inline='always')
def fold(f, init, n, consts):
    "``acc = f(acc, i, consts)`` for ``i`` in ``range(n)``, starting from ``init``."
    acc = init
    for i in range(n):
        acc = f(acc, i, consts)
    return acc



@njit(inline='always')
def scan(f, init, n, consts):
    """
    ``carry, y = f(carry, i, consts)`` for ``i`` in ``range(n)``, starting from
    ``init``. Returns the last carry and the ``y`` of every step, shape (n, k).
    Each ``y`` is a 1-D array of the same length; ``n`` is at least 1.
    """
    carry, y = f(init, 0, consts)
    ys = np.empty((n, y.shape[0]))
    ys[0] = y
    for i in range(1, n):
        carry, y = f(carry, i, consts)
        ys[i] = y
    return carry, ys



@njit(inline='always')
def cond(pred, f, g, operand):
    "``f(operand)`` if ``pred`` else ``g(operand)``. Only the chosen one runs."
    if pred:
        return f(operand)
    return g(operand)



@njit(inline='always')
def root(f, low, high, tol, args):
    """
    Root of an increasing ``f(x, args)`` in ``[low, high]`` by bisection, to
    within ``tol``. ``f(low) < 0 <= f(high)``.
    """
    while high - low > tol:
        middle = 0.5 * (low + high)
        if f(middle, args) < 0.:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
