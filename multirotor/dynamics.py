"""
Right-hand sides of the vehicle's equations of motion, one per vehicle
configuration. Each function has an integrator's signature ``(t, x, *args)`` and
returns ``dx/dt`` for the whole integrated state ``x``.

This module is the only code that knows the layout of ``x``. It slices ``x``,
calls the subsystem equations in `multirotor.physics`, and stacks their results.
"""
from typing import Tuple

import numpy as np
from numba import njit

from .coords import rotating_frame_derivative
from .physics import (
    battery_ocv, battery_operating_point, body_derivative, body_torques,
    forces_torques, motor_acc, thrusts_physics,
    BATTERY_CAPACITY, BATTERY_INTERNAL_RESISTANCE, PROP_K_DRAG
)

N_BODY_STATES = 12
"The first 12 elements of the integrated state are the body states."



@njit
def rhs_body(
    t: float, x: np.ndarray, forces: np.ndarray, torques: np.ndarray, g: float,
    mass: float, inertia_matrix: np.ndarray, inertia_matrix_inverse: np.ndarray
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
        Forces (N) in the body frame, shape (3,).
    torques : np.ndarray
        Torques (N m) in the body frame, shape (3,).
    g : float
        Gravitational acceleration.
    mass : float
        Mass of the vehicle.
    inertia_matrix : np.ndarray
        Inertia matrix, shape (3, 3).
    inertia_matrix_inverse : np.ndarray
        Inverse of the inertia matrix, shape (3, 3).

    Returns
    -------
    np.ndarray
        The rate of change of the body state.
    """
    return body_derivative(
        forces, torques, x, g, mass, inertia_matrix, inertia_matrix_inverse)



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
    t: float, x: np.ndarray, u: np.ndarray, max_voltage: float,
    disturbance: np.ndarray, g: float, mass: float, inertia_matrix: np.ndarray,
    inertia_matrix_inverse: np.ndarray, prop_params: np.ndarray,
    motor_params: np.ndarray, clockwise: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """
    Rate of change of the body and its motors, for vehicles with motors whose
    thrust is ``k_thrust * speed**2``. Motors and body evolve together: forces
    at each evaluation come from the motor speeds at that instant.

    The integrated state ``x`` is the 12 body states followed by the ``n``
    motor speeds.

    Parameters
    ----------
    t : float
        Time. The equations are time invariant.
    x : np.ndarray
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
        The rate of change of ``x``.
    """
    body_state = x[:N_BODY_STATES]
    speeds = x[N_BODY_STATES:]
    forces, torques = forces_torques(speeds, prop_params, clockwise, positions)
    dxdt = np.empty_like(x)
    dxdt[:N_BODY_STATES] = body_derivative(
        forces + disturbance[:3], torques + disturbance[3:], body_state, g, mass,
        inertia_matrix, inertia_matrix_inverse)
    dxdt[N_BODY_STATES:] = motor_acc(speeds, u, max_voltage, motor_params)
    return dxdt



@njit
def battery_state(
    speeds: np.ndarray, u: np.ndarray, charge: float, cutoff: bool,
    motor_params: np.ndarray, battery_params: np.ndarray, ocv_soc: np.ndarray,
    ocv_voltage: np.ndarray
) -> Tuple[float, float, float]:
    """
    The battery's terminal voltage and current, and the voltage limit it puts
    on the motors.

    Parameters
    ----------
    speeds : np.ndarray
        Motor speeds (rad/s), shape (n,).
    u : np.ndarray
        Speed signals, shape (n,).
    charge : float
        Total charge (C) drawn from the battery since it was full.
    cutoff : bool
        Whether the battery has cut off the motors.
    motor_params : np.ndarray
        The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    battery_params : np.ndarray
        Battery parameters, shape (N_BATTERY_PARAMS,), indexed by the
        ``BATTERY_*`` constants in `multirotor.physics`.
    ocv_soc : np.ndarray
        States of charge of the open-circuit voltage table.
    ocv_voltage : np.ndarray
        Open-circuit voltage (V) at each ``ocv_soc``.

    Returns
    -------
    Tuple[float, float, float]
        The terminal voltage (V), the battery current (A), and the motors'
        voltage limit (V).
    """
    soc = 1. - charge / battery_params[BATTERY_CAPACITY]
    open_circuit_voltage = battery_ocv(soc, ocv_soc, ocv_voltage)
    if cutoff:
        return open_circuit_voltage, 0., 0.
    terminal_voltage, power, _ = battery_operating_point(
        speeds, u, open_circuit_voltage,
        battery_params[BATTERY_INTERNAL_RESISTANCE], motor_params)
    return terminal_voltage, power / terminal_voltage, terminal_voltage



@njit
def rhs_motor_battery(
    t: float, x: np.ndarray, u: np.ndarray, cutoff: bool,
    disturbance: np.ndarray, g: float, mass: float, inertia_matrix: np.ndarray,
    inertia_matrix_inverse: np.ndarray, prop_params: np.ndarray,
    motor_params: np.ndarray, clockwise: np.ndarray, positions: np.ndarray,
    battery_params: np.ndarray, ocv_soc: np.ndarray, ocv_voltage: np.ndarray,
    charge_drawn: float
) -> np.ndarray:
    """
    Like `rhs_motor`, with a battery model. The battery's terminal voltage
    limits the motors' voltage, and sags with the current they draw and as the
    battery discharges.

    The integrated state ``x`` is the 12 body states, the ``n`` motor speeds,
    and the charge (C) drawn from the battery during the current time step.

    Parameters
    ----------
    t, x, u, disturbance, g, mass, inertia_matrix, inertia_matrix_inverse, prop_params, motor_params, clockwise, positions
        As in `rhs_motor`. ``x`` has one more element: the charge drawn during
        the step.
    cutoff : bool
        Whether the battery has cut off the motors (their voltage is zero).
    battery_params, ocv_soc, ocv_voltage
        As in `battery_state`.
    charge_drawn : float
        Charge (C) drawn from the battery before the current time step.

    Returns
    -------
    np.ndarray
        The rate of change of ``x``.
    """
    n = u.shape[0]
    body_state = x[:N_BODY_STATES]
    speeds = x[N_BODY_STATES:N_BODY_STATES + n]
    charge = charge_drawn + x[N_BODY_STATES + n]
    _, battery_current, motor_voltage_limit = battery_state(
        speeds, u, charge, cutoff, motor_params, battery_params, ocv_soc,
        ocv_voltage)
    forces, torques = forces_torques(speeds, prop_params, clockwise, positions)
    dxdt = np.empty_like(x)
    dxdt[:N_BODY_STATES] = body_derivative(
        forces + disturbance[:3], torques + disturbance[3:], body_state, g, mass,
        inertia_matrix, inertia_matrix_inverse)
    dxdt[N_BODY_STATES:N_BODY_STATES + n] = motor_acc(
        speeds, u, motor_voltage_limit, motor_params)
    dxdt[N_BODY_STATES + n] = battery_current
    return dxdt



def rhs_motor_blade_element(
    t: float, x: np.ndarray, u: np.ndarray, max_voltage: float,
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
    t, x, u, max_voltage, disturbance, g, mass, inertia_matrix, inertia_matrix_inverse, prop_params, motor_params, clockwise, positions
        As in `rhs_motor`.
    rho : float
        Air density (kg/m^3).

    Returns
    -------
    np.ndarray
        The rate of change of ``x``.
    """
    body_state = x[:N_BODY_STATES]
    speeds = x[N_BODY_STATES:]
    # Same airstream inputs as `Multirotor.get_forces_torques`.
    airstream_velocities = rotating_frame_derivative(
        positions, body_state[:3], body_state[3:6])
    thrusts = thrusts_physics(speeds, airstream_velocities, prop_params, rho)
    drag_torques = prop_params[PROP_K_DRAG] * speeds**2
    torques = body_torques(thrusts, drag_torques, clockwise, positions)
    forces = np.asarray([0., 0., thrusts.sum()])
    dxdt = np.empty_like(x)
    dxdt[:N_BODY_STATES] = body_derivative(
        forces + disturbance[:3], torques + disturbance[3:], body_state, g, mass,
        inertia_matrix, inertia_matrix_inverse)
    dxdt[N_BODY_STATES:] = motor_acc(speeds, u, max_voltage, motor_params)
    return dxdt
