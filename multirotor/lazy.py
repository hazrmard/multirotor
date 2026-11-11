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
    namespace: dict, not_inlined: Sequence[str] = ()
) -> types.SimpleNamespace:
    """Copies of the functions ``names`` of ``source`` for a backend."""
    xp = import_module(f'.xp_{backend}', __package__)
    cache = backend == 'numba' and xp.CACHE
    never = set(not_inlined)
    jit = (
        lambda f: njit(inline='never' if f.__name__ in never else 'always', cache=cache)(f)
    ) if backend == 'numba' else (lambda f: f)
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
