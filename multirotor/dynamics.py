"""
Right-hand sides of the vehicle's equations of motion, one per vehicle
configuration. Each function has an integrator's signature ``(t, x, *args)`` and
returns ``dx/dt`` for the whole integrated state ``x``.

This module is the only code that knows the layout of ``x``. It slices ``x``,
calls the subsystem equations in `multirotor.physics`, and stacks their results.

The state ``x`` and the functions each right-hand side calls, for following the
order of execution. An arrow goes from a caller to the function it calls. Every
right-hand side is compiled with numba, except
``rhs_body_with_motors_blade_element``, which is plain Python because it calls
``fsolve``::

    rhs_body                              x = body (12)
        -> body_derivative

    rhs_motor_only                        x = motor speeds (n)
        -> motor_acc -> motor_electrical

    rhs_body_with_motors                  x = body (12), motor speeds (n)
        -> forces_torques -> body_torques
        -> body_derivative
        -> motor_acc -> motor_electrical

    rhs_body_with_motors_and_battery      x = body (12), motor speeds (n),
                                              charge drawn in this step (1)
        -> battery_state -> battery_ocv
                         -> battery_operating_point -> motor_power -> motor_electrical
                                                    -> battery_terminal
        -> forces_torques, body_derivative, motor_acc  (as in rhs_body_with_motors)

    rhs_body_with_motors_blade_element    x = body (12), motor speeds (n)
        -> rotating_frame_derivative (in `multirotor.coords`), for the airstream
        -> thrusts_blade_element_model -> thrust_blade_element_model
        -> body_torques
        -> body_derivative
        -> motor_acc -> motor_electrical

Which one `Multirotor` integrates:

* Without motors, and with ``k_thrust * speed**2`` thrust, forces are fixed over
  a step, so `Multirotor.step_speeds` and `Multirotor.step_dynamics` integrate
  ``rhs_body`` with forces and torques calculated once.
* Without motors, and with blade-element thrust, forces change during the step:
  `Multirotor.step_speeds` integrates `Multirotor.dxdt_speeds`, which calls
  ``rhs_body``.
* With motors, `Multirotor.__init__` picks one of the three ``rhs_body_with_*``
  functions from the thrust model and from whether there is a battery model.
* ``rhs_motor_only`` steps a standalone `Motor`, and the motors of a
  `Multirotor` when their speeds are needed without stepping the body.
"""
from typing import NamedTuple, Tuple

import numpy as np

from . import xp_numba as xp
from .lazy import build

from .coords import rotating_frame_derivative
from .physics import (
    battery_ocv, battery_operating_point, body_derivative, body_torques,
    forces_torques, motor_acc, thrusts_blade_element_model,
    BATTERY_CAPACITY, BATTERY_INTERNAL_RESISTANCE, PROP_K_DRAG
)

N_BODY_STATES = 12
"The first 12 elements of the integrated state are the body states."


class BatteryState(NamedTuple):
    "What `battery_state` returns: the terminal voltage, the battery current, and the motors' voltage limit."
    terminal_voltage: float
    current: float
    voltage_limit: float


class Supply(NamedTuple):
    "The operand of `_no_current` and `_supplying`: the battery's source voltage and the motors."
    open_circuit_voltage: float
    speeds: np.ndarray
    u: np.ndarray
    motor_params: np.ndarray
    battery_params: np.ndarray


def rhs_body(
    t: float, x: np.ndarray, forces: np.ndarray, torques: np.ndarray, g: float,
    mass: float, inertia_matrix: np.ndarray, inertia_matrix_inverse: np.ndarray


) -> np.ndarray:
    """
    Rate of change of the vehicle's 12 body states under forces and torques that
    are fixed over the time step.

    Used when the forces stay fixed during a step: for
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



def rhs_body_with_motors(
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
    return xp.concatenate((
        body_derivative(
            forces + disturbance[:3], torques + disturbance[3:], body_state, g, mass,
            inertia_matrix, inertia_matrix_inverse),
        motor_acc(speeds, u, max_voltage, motor_params)))



def _no_current(op: Supply) -> BatteryState:
    """
    A battery that is cut off or empty supplies no current, and the motors get no voltage.

    Parameters
    ----------
    op : Supply
        The battery's source voltage and the motors.

    Returns
    -------
    BatteryState
        The source voltage, and no current or voltage limit.
    """
    return BatteryState(op.open_circuit_voltage, 0., 0.)



def _supplying(op: Supply) -> BatteryState:
    """
    A battery that supplies the motors: its voltage, and the current at the power they draw.

    Parameters
    ----------
    op : Supply
        The battery's source voltage and the motors.

    Returns
    -------
    BatteryState
        The terminal voltage, the current and the motors' voltage limit.
    """
    point = battery_operating_point(
        op.speeds, op.u, op.open_circuit_voltage,
        op.battery_params[BATTERY_INTERNAL_RESISTANCE], op.motor_params)
    return BatteryState(
        point.terminal_voltage, point.power / point.terminal_voltage, point.terminal_voltage)



def battery_state(
    speeds: np.ndarray, u: np.ndarray, charge: float, cutoff: bool,
    motor_params: np.ndarray, battery_params: np.ndarray, ocv_soc: np.ndarray,
    ocv_voltage: np.ndarray


) -> BatteryState:
    """
    The battery's terminal voltage and current, and the voltage limit it puts
    on the motors. A battery that is cut off or empty supplies no current, and
    the motors get no voltage.

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
    BatteryState
        The terminal voltage (V), the battery current (A), and the motors'
        voltage limit (V).
    """
    soc = 1. - charge / battery_params[BATTERY_CAPACITY]
    open_circuit_voltage = battery_ocv(soc, ocv_soc, ocv_voltage)
    return xp.cond(
        cutoff | (soc <= 0.), _no_current, _supplying,
        Supply(open_circuit_voltage, speeds, u, motor_params, battery_params))



def rhs_body_with_motors_and_battery(
    t: float, x: np.ndarray, u: np.ndarray, cutoff: bool,
    disturbance: np.ndarray, g: float, mass: float, inertia_matrix: np.ndarray,
    inertia_matrix_inverse: np.ndarray, prop_params: np.ndarray,
    motor_params: np.ndarray, clockwise: np.ndarray, positions: np.ndarray,
    battery_params: np.ndarray, ocv_soc: np.ndarray, ocv_voltage: np.ndarray,
    charge_drawn: float


) -> np.ndarray:
    """
    Like `rhs_body_with_motors`, with a battery model. The battery's terminal
    voltage limits the motors' voltage, and sags with the current they draw and
    as the battery discharges.

    The integrated state ``x`` is the 12 body states, the ``n`` motor speeds,
    and the charge (C) drawn from the battery during the current time step.

    Parameters
    ----------
    t, x, u, disturbance, g, mass, inertia_matrix, inertia_matrix_inverse, prop_params, motor_params, clockwise, positions
        As in `rhs_body_with_motors`. ``x`` has one more element: the charge
        drawn during the step.
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
    return xp.concatenate((
        body_derivative(
            forces + disturbance[:3], torques + disturbance[3:], body_state, g, mass,
            inertia_matrix, inertia_matrix_inverse),
        motor_acc(speeds, u, motor_voltage_limit, motor_params),
        xp.array((battery_current,))))



# The equations above are plain Python against ``xp``; compile them for numba
# under their own names. `multirotor.xp_jax` builds the same source for JAX.
NAMES = (
    'rhs_body', 'rhs_motor_only', 'rhs_body_with_motors', '_no_current',
    '_supplying', 'battery_state', 'rhs_body_with_motors_and_battery',
)
SOURCE = {name: globals()[name] for name in NAMES}
globals().update(vars(build(SOURCE, NAMES, 'numba', globals())))



def rhs_body_with_motors_blade_element(
    t: float, x: np.ndarray, u: np.ndarray, max_voltage: float,
    disturbance: np.ndarray, g: float, mass: float, inertia_matrix: np.ndarray,
    inertia_matrix_inverse: np.ndarray, prop_params: np.ndarray,
    motor_params: np.ndarray, clockwise: np.ndarray, positions: np.ndarray,
    rho: float


) -> np.ndarray:
    """
    Like `rhs_body_with_motors`, for vehicles whose thrust comes from the
    blade-element model. Thrust depends on the airstream at each propeller, so
    it is solved numerically at each evaluation. Plain Python, and much slower
    than `rhs_body_with_motors`.

    Parameters
    ----------
    t, x, u, max_voltage, disturbance, g, mass, inertia_matrix, inertia_matrix_inverse, prop_params, motor_params, clockwise, positions
        As in `rhs_body_with_motors`.
    rho : float
        Air density (kg/m^3).

    Returns
    -------
    np.ndarray
        The rate of change of ``x``.
    """
    body_state = x[:N_BODY_STATES]
    speeds = x[N_BODY_STATES:]
    # Same airstream inputs as `Multirotor.get_forces_torques`: the body-frame
    # velocity and angular rate.
    airstream_velocities = rotating_frame_derivative(
        positions, body_state[3:6], body_state[9:12])
    thrusts = thrusts_blade_element_model(
        speeds, airstream_velocities, prop_params, rho)
    drag_torques = prop_params[PROP_K_DRAG] * speeds**2
    torques = body_torques(thrusts, drag_torques, clockwise, positions)
    forces = np.asarray([0., 0., thrusts.sum()])
    return xp.concatenate((
        body_derivative(
            forces + disturbance[:3], torques + disturbance[3:], body_state, g, mass,
            inertia_matrix, inertia_matrix_inverse),
        motor_acc(speeds, u, max_voltage, motor_params)))
