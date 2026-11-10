"""
Compile a module of equations for a backend. The equations are plain Python
functions that use the global name ``xp`` for arrays and control flow. A
backend is a copy of those functions whose ``xp`` is `multirotor.xp_numba` or
`multirotor.xp_jax`. The source is not changed.
"""
import functools
import types
from importlib import import_module
from typing import Dict, Sequence

from numba import njit



def build(
    source: Dict[str, types.FunctionType], names: Sequence[str], backend: str,
    namespace: dict
) -> types.SimpleNamespace:
    """
    Copies of the functions ``names`` of ``source`` for a backend.

    Parameters
    ----------
    source : Dict[str, FunctionType]
        The plain functions by name.
    names : Sequence[str]
        Which of them to build. Functions that call each other must all be
        built, and find each other in the same namespace.
    backend : str
        ``'numba'``: compiled by `numba.njit`, with `multirotor.xp_numba`.
        ``'jax'``: left to be traced by the caller's ``jax.jit``, with
        `multirotor.xp_jax`.
    namespace : dict
        The globals of the module the functions came from (its constants and
        helpers). It is copied, not changed.

    Returns
    -------
    types.SimpleNamespace
        The built functions as attributes, with the source's docstrings and
        signatures.
    """
    xp = import_module(f'.xp_{backend}', __package__)
    jit = (lambda f: njit(inline='always')(f)) if backend == 'numba' else (lambda f: f)
    ns = dict(namespace)
    ns['xp'] = xp
    built = types.SimpleNamespace()
    for name in names:
        py = source[name]
        fn = types.FunctionType(py.__code__, ns, name, py.__defaults__, py.__closure__)
        functools.update_wrapper(fn, py)
        fn.__kwdefaults__ = py.__kwdefaults__
        fn = jit(fn)
        ns[name] = fn
        setattr(built, name, fn)
    return built
