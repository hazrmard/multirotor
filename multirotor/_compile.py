"""
Compiling a function that Python calls once per simulation step.

`step_function` is ``jax.jit`` with one addition: the compiled program runs the function
inside a loop of one iteration, with the function's first argument (the state it advances)
as the value that the loop carries. This is a workaround found by measurement, not by
design: on the CPU, with JAX 0.12, a vehicle's step took 37 us compiled as it is and 11 us
compiled this way. Why XLA runs it faster was not established, so measure again when the
version of JAX changes (`scripts/benchmarks`). The loop's count is an argument, so that it
cannot be unrolled when the program is compiled.

This matters only for calls from Python that do little work each, as the classes
(`multirotor.simulation.Multirotor`, `multirotor.controller.Controller`) make. A rollout
(`multirotor.rollout.closed_loop`) is a loop already. A program compiled this way cannot be
differentiated in reverse mode; differentiate the plain function, kept as ``.plain``.

Two more things keep such a call cheap, and are the caller's part: pass parameters that do
not change as JAX arrays made once (a numpy array is converted at every call), and return one
array, converted to numpy once.
"""
import functools
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax import lax



def step_function(f: Callable, static_argnames: Sequence[str] = ()) -> Callable:
    """
    Compile ``f`` for calls from Python, one per simulation step.

    Parameters
    ----------
    f : Callable
        A function of arrays (and tuples of arrays) that returns an array or a tuple of
        arrays. Its first argument must be an array, or a tuple of arrays, of numbers: the
        state that ``f`` advances.
    static_argnames : Sequence[str], optional
        Arguments that are not arrays, such as a count of substeps. Pass them by keyword.

    Returns
    -------
    Callable
        The compiled function, with the same arguments and results. ``.plain`` is ``f``.
    """
    def looped(count: jax.Array, state: Any, *args: Any, **kwargs: Any) -> Any:
        """``f``, run ``count`` (one) times in a loop that carries its first argument."""
        state = jax.tree_util.tree_map(jnp.asarray, state)
        like = jax.eval_shape(lambda: f(state, *args, **kwargs))
        start = jax.tree_util.tree_map(lambda a: jnp.zeros(a.shape, a.dtype), like)

        def body(_: jax.Array, carry: tuple) -> tuple:
            state, _ = carry
            # The carried state is given a new value, so that it is a loop variable and not a
            # constant of the loop. The value is never used: the loop runs once.
            changed = jax.tree_util.tree_map(lambda a: a + 1, state)
            return changed, f(state, *args, **kwargs)

        return lax.fori_loop(0, count, body, (state, start))[1]

    compiled = jax.jit(looped, static_argnames=tuple(static_argnames))

    @functools.wraps(f)
    def call(*args: Any, **kwargs: Any) -> Any:
        return compiled(1, *args, **kwargs)

    call.plain = f
    return call
