"""
The integrator: fixed-step RK4, written for JAX. It is explicit and has no
data-dependent steps, so it can be compiled, mapped over vehicles and differentiated.
`multirotor.rollout` and `multirotor.simulation` advance every vehicle with it.
"""
from typing import Callable, Tuple

import jax
from jax import lax

Array = jax.Array

DEFAULT_SUBSTEPS = {'body': 4, 'motors': 16, 'motors_battery': 16}
"""
RK4 steps per simulation step for each kind of vehicle (`multirotor.tuples.kind_of`), at a
time step of 0.01 s. With motors the equations are stiffer, and need more.
"""



def rk4_step(f: Callable[[Array], Array], x: Array, dt: float) -> Tuple[Array, Array]:
    """One RK4 step of ``dx/dt = f(x)``. Returns the new ``x``, and ``f`` at the start."""
    k1 = f(x)
    k2 = f(x + dt / 2. * k1)
    k3 = f(x + dt / 2. * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4), k1



def integrate(
    f: Callable[[Array], Array], x: Array, dt: float, substeps: int
) -> Tuple[Array, Array]:
    """
    Advance ``x`` by ``dt`` in ``substeps`` RK4 steps. Returns the new ``x``, and ``f`` at
    the start.
    """
    h = dt / substeps
    x1, dxdt = rk4_step(f, x, h)
    return lax.fori_loop(1, substeps, lambda _, x: rk4_step(f, x, h)[0], x1), dxdt
