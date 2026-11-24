"""
A simulation testbed for multirotor vehicles, compiled and differentiated with JAX.

Importing the package turns on 64-bit floats in JAX (``jax_enable_x64``), for the whole
process: the integrator needs them. It also turns on JAX's on-disk compilation cache
(see `multirotor._cache`).
"""
import jax

from . import _cache

jax.config.update('jax_enable_x64', True)
_cache.configure_jax(jax)
