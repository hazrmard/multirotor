"""
Right-hand sides of the vehicle's equations of motion, one per vehicle
configuration. Each function has an integrator's signature ``(t, y, *args)`` and
returns ``dy/dt`` for the whole integrated state ``y``.

This module is the only code that knows the layout of ``y``. It slices ``y``,
calls the subsystem equations in `multirotor.physics`, and stacks their results.
"""
from typing import Tuple

import numpy as np
from numba import njit

from .coords import rotating_frame_derivative
from .physics import (
    body_derivative, body_torques, forces_torques, motor_acc, thrusts_physics,
    PROP_K_DRAG
)

N_BODY_STATES = 12
"The first 12 elements of the integrated state are the body states."



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



@njit
def rhs_motor_only(
    t: float, speed: np.ndarray, u: np.ndarray, max_voltage: float,
    motor_params: np.ndarray
) -> np.ndarray:
    """
    Rate of change of motor speeds, for motors stepped on their own (outside a
    vehicle). The integrated state is the ``n`` motor speeds.

    Parameters
    ----------
    t : float
        Time. The equations are time invariant.
    speed : np.ndarray
        Motor speeds (rad/s), shape (n,).
    u : np.ndarray
        Speed signals, shape (n,).
    max_voltage : float
        The maximum voltage the power source can supply.
    motor_params : np.ndarray
        The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).

    Returns
    -------
    np.ndarray
        Angular acceleration of each motor, shape (n,).
    """
    return motor_acc(speed, u, max_voltage, motor_params)



@njit
def rhs_motor(
    t: float, y: np.ndarray, u: np.ndarray, max_voltage: float,
    disturbance: np.ndarray, g: float, mass: float, inertia_matrix: np.ndarray,
    inertia_matrix_inverse: np.ndarray, prop_params: np.ndarray,
    motor_params: np.ndarray, clockwise: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """
    Rate of change of the body and its motors, for vehicles with motors whose
    thrust is ``k_thrust * speed**2``. Motors and body evolve together: forces
    at each evaluation come from the motor speeds at that instant.

    The integrated state ``y`` is the 12 body states followed by the ``n``
    motor speeds.

    Parameters
    ----------
    t : float
        Time. The equations are time invariant.
    y : np.ndarray
        The integrated state, shape (12 + n,).
    u : np.ndarray
        Speed signals, shape (n,).
    max_voltage : float
        The maximum voltage the power source can supply.
    disturbance : np.ndarray
        External forces (N) and torques (N m) in the body frame, shape (6,).
    g : float
        Gravitational acceleration.
    mass : float
        Mass of the vehicle.
    inertia_matrix : np.ndarray
        Inertia matrix, shape (3, 3).
    inertia_matrix_inverse : np.ndarray
        Inverse of the inertia matrix, shape (3, 3).
    prop_params : np.ndarray
        The `PropellerData.params` matrix, shape (N_PROP_PARAMS, n).
    motor_params : np.ndarray
        The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    clockwise : np.ndarray
        1 if a propeller spins clockwise, -1 if anti-clockwise, shape (n,).
    positions : np.ndarray
        Position of each propeller relative to the center of mass, shape (3, n).

    Returns
    -------
    np.ndarray
        The rate of change of ``y``.
    """
    x = y[:N_BODY_STATES]
    speeds = y[N_BODY_STATES:]
    forces, torques = forces_torques(speeds, prop_params, clockwise, positions)
    dydt = np.empty_like(y)
    dydt[:N_BODY_STATES] = body_derivative(
        forces + disturbance[:3], torques + disturbance[3:], x, g, mass,
        inertia_matrix, inertia_matrix_inverse)
    dydt[N_BODY_STATES:] = motor_acc(speeds, u, max_voltage, motor_params)
    return dydt



def rhs_motor_blade_element(
    t: float, y: np.ndarray, u: np.ndarray, max_voltage: float,
    disturbance: np.ndarray, g: float, mass: float, inertia_matrix: np.ndarray,
    inertia_matrix_inverse: np.ndarray, prop_params: np.ndarray,
    motor_params: np.ndarray, clockwise: np.ndarray, positions: np.ndarray,
    rho: float
) -> np.ndarray:
    """
    Like `rhs_motor`, for vehicles whose thrust comes from the blade-element
    model. Thrust depends on the airstream at each propeller, so it is solved
    numerically at each evaluation. Plain Python, and much slower than
    `rhs_motor`.

    Parameters
    ----------
    t, y, u, max_voltage, disturbance, g, mass, inertia_matrix, inertia_matrix_inverse, prop_params, motor_params, clockwise, positions
        As in `rhs_motor`.
    rho : float
        Air density (kg/m^3).

    Returns
    -------
    np.ndarray
        The rate of change of ``y``.
    """
    x = y[:N_BODY_STATES]
    speeds = y[N_BODY_STATES:]
    # Same airstream inputs as `Multirotor.get_forces_torques`.
    airstream_velocities = rotating_frame_derivative(positions, x[:3], x[3:6])
    thrusts = thrusts_physics(speeds, airstream_velocities, prop_params, rho)
    drag_torques = prop_params[PROP_K_DRAG] * speeds**2
    torques = body_torques(thrusts, drag_torques, clockwise, positions).sum(axis=1)
    forces = np.asarray([0., 0., thrusts.sum()])
    dydt = np.empty_like(y)
    dydt[:N_BODY_STATES] = body_derivative(
        forces + disturbance[:3], torques + disturbance[3:], x, g, mass,
        inertia_matrix, inertia_matrix_inverse)
    dydt[N_BODY_STATES:] = motor_acc(speeds, u, max_voltage, motor_params)
    return dydt
