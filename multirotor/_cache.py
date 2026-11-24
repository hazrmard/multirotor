"""
The on-disk cache of compiled code. JAX keys its cache on the program and the versions of
its libraries, so an edit to this package compiles again and nothing stale is read.

The location is ``$MULTIROTOR_CACHE_DIR``, else the user cache directory
(``$XDG_CACHE_HOME`` or ``~/.cache``) under ``multirotor``. Set ``MULTIROTOR_CACHE=0`` to
turn the cache off.
"""
import os
from pathlib import Path
from types import ModuleType



def enabled() -> bool:
    """Whether the cache is on."""
    return os.environ.get('MULTIROTOR_CACHE', '1') != '0'



def root() -> Path:
    """The directory of the cache."""
    given = os.environ.get('MULTIROTOR_CACHE_DIR')
    if given:
        return Path(given)
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'multirotor'



def configure_jax(jax: ModuleType) -> None:
    """Turn on JAX's persistent compilation cache, for programs that took over a second."""
    if enabled():
        jax.config.update('jax_compilation_cache_dir', str(root() / 'jax'))
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
