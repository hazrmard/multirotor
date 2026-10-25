"""
Right-hand sides of the vehicle's equations of motion, one per vehicle
configuration. Each function has an integrator's signature ``(t, y, *args)`` and
returns ``dy/dt`` for the whole integrated state ``y``.

This module is the only code that knows the layout of ``y``. It slices ``y``,
calls the subsystem equations in `multirotor.physics`, and stacks their results.
"""
from typing import Tuple

import numpy as np

from .physics import body_derivative



def rhs_body(
    t: float, x: np.ndarray, forces: np.ndarray, torques: np.ndarray,
    body_params: Tuple[float, float, np.ndarray, np.ndarray], dtype: type,
    decimals: int
) -> np.ndarray:
    """
    Rate of change of the vehicle's 12 body states under forces and torques that
    are fixed over the time step.

    Used when the forces don't depend on the state during a step: for
    `Multirotor.step_dynamics`, and for `Multirotor.step_speeds` on vehicles
    without motors whose thrust is ``k_thrust * speed**2``.

    Parameters
    ----------
    t : float
        Time. The equations are time invariant.
    x : np.ndarray
        The body state: position, velocity, orientation, angular rate.
    forces : np.ndarray
        Forces (N) in the body frame.
    torques : np.ndarray
        Torques (N m) in the body frame.
    body_params : Tuple[float, float, np.ndarray, np.ndarray]
        Gravitational acceleration, mass, inertia matrix and its inverse.
    dtype : type
        The data type in which to evaluate the equations.
    decimals : int
        The number of decimals to which the derivative is rounded.

    Returns
    -------
    np.ndarray
        The rate of change of the body state.
    """
    g, mass, inertia_matrix, inertia_matrix_inverse = body_params
    dxdt = body_derivative(
        forces, torques, x.astype(dtype), g, mass,
        inertia_matrix, inertia_matrix_inverse)
    return np.around(dxdt, decimals)
