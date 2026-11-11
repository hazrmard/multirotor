"""
On-disk caches of compiled code, for `multirotor.xp_numba` and `multirotor.xp_jax`.

numba keys a cached function on its own file only, so editing a function that it
calls from another file would leave a stale entry. The numba cache directory is
therefore named by a hash of every source file of the package: any edit starts a
fresh cache. JAX keys its cache on the program and library versions itself.

The location is ``$MULTIROTOR_CACHE_DIR``, else the user cache directory
(``$XDG_CACHE_HOME`` or ``~/.cache``) under ``multirotor``. Set
``MULTIROTOR_CACHE=0`` to turn the caches off.
"""
import hashlib
import os
from pathlib import Path
from types import ModuleType



def enabled() -> bool:
    """
    Whether the caches are on.

    Returns
    -------
    bool
        False if ``MULTIROTOR_CACHE=0``.
    """
    return os.environ.get('MULTIROTOR_CACHE', '1') != '0'



def root() -> Path:
    """
    The directory of the caches.

    Returns
    -------
    Path
        ``$MULTIROTOR_CACHE_DIR`` if set, else ``multirotor`` in the user cache directory.
    """
    given = os.environ.get('MULTIROTOR_CACHE_DIR')
    if given:
        return Path(given)
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'multirotor'



def _source_hash() -> str:
    """
    A hash of every source file of the package, so that an edit starts a fresh cache.

    Returns
    -------
    str
        The first 16 hex digits of the SHA-1 of the files' names and contents.
    """
    digest = hashlib.sha1()
    package = Path(__file__).parent
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]



def configure_numba() -> bool:
    """
    Point numba's cache at this source's directory.

    Returns
    -------
    bool
        Whether caching is on.
    """
    if not enabled():
        return False
    from numba import config
    config.CACHE_DIR = str(root() / 'numba' / _source_hash())
    return True



def configure_jax(jax: ModuleType) -> None:
    """
    Turn on JAX's persistent compilation cache, for programs that took over a second.

    Parameters
    ----------
    jax : ModuleType
        The ``jax`` module.
    """
    if enabled():
        jax.config.update('jax_compilation_cache_dir', str(root() / 'jax'))
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
