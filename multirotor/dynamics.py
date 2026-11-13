"""
Right-hand sides of the vehicle's equations of motion, one per vehicle
configuration. Each function has an integrator's signature ``(t, x, *args)`` and
returns ``dx/dt`` for the whole integrated state ``x``.

This module is the only code that knows the layout of ``x``. It slices ``x``,
calls the subsystem equations in `multirotor.physics`, and stacks their results.

The small functions and local named tuples here (``_no_current``, ``_supplying``,
``MotorSupply``) are the branches of an ``xp.cond`` and their input; the reasons are
given in `multirotor.physics`.

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
    """
    The output of `battery_state`: the terminal voltage, the battery current, and the motors'
    voltage limit.
    """
    terminal_voltage: float
    current: float
    voltage_limit: float


class MotorSupply(NamedTuple):
    """
    The battery's supply to the motors: its source voltage, and the motors' speeds, signals
    and parameters. The input to `_no_current` and `_supplying`.
    """
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

    ``forces``: Forces (N) in the body frame, shape (3,).
    ``torques``: Torques (N m) in the body frame, shape (3,).
    ``inertia_matrix``: Inertia matrix, shape (3, 3).
    ``inertia_matrix_inverse``: Inverse of the inertia matrix, shape (3, 3).
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

    ``speed``: Motor speeds (rad/s), shape (n,).
    ``u``: Speed signals, shape (n,).
    ``motor_params``: The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    Returns: Angular acceleration of each motor, shape (n,).
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

    ``x``: The integrated state, shape (12 + n,).
    ``u``: Speed signals, shape (n,).
    ``disturbance``: External forces (N) and torques (N m) in the body frame, shape (6,).
    ``inertia_matrix``: Inertia matrix, shape (3, 3).
    ``inertia_matrix_inverse``: Inverse of the inertia matrix, shape (3, 3).
    ``prop_params``: The `PropellerData.params` matrix, shape (N_PROP_PARAMS, n).
    ``motor_params``: The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    ``clockwise``: 1 if a propeller spins clockwise, -1 if anti-clockwise, shape (n,).
    ``positions``: Position of each propeller relative to the center of mass, shape (3, n).
    """
    body_state = x[:N_BODY_STATES]
    speeds = x[N_BODY_STATES:]
    forces, torques = forces_torques(speeds, prop_params, clockwise, positions)
    return xp.concatenate((
        body_derivative(
            forces + disturbance[:3], torques + disturbance[3:], body_state, g, mass,
            inertia_matrix, inertia_matrix_inverse),
        motor_acc(speeds, u, max_voltage, motor_params)))



def _no_current(op: MotorSupply) -> BatteryState:
    """A battery that is cut off or empty supplies no current, and the motors get no voltage."""
    return BatteryState(op.open_circuit_voltage, 0., 0.)



def _supplying(op: MotorSupply) -> BatteryState:
    """A battery that supplies the motors: its voltage, and the current at the power they draw."""
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

    ``speeds``: Motor speeds (rad/s), shape (n,).
    ``u``: Speed signals, shape (n,).
    ``charge``: Total charge (C) drawn from the battery since it was full.
    ``motor_params``: The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    ``battery_params``: Battery parameters, shape (N_BATTERY_PARAMS,), indexed by the
      ``BATTERY_*`` constants in `multirotor.physics`.
    ``ocv_voltage``: Open-circuit voltage (V) at each ``ocv_soc``.
    Returns: The terminal voltage (V), the battery current (A), and the motors' voltage
      limit (V).
    """
    soc = 1. - charge / battery_params[BATTERY_CAPACITY]
    open_circuit_voltage = battery_ocv(soc, ocv_soc, ocv_voltage)
    return xp.cond(
        cutoff | (soc <= 0.), _no_current, _supplying,
        MotorSupply(open_circuit_voltage, speeds, u, motor_params, battery_params))



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

    ``charge_drawn``: Charge (C) drawn from the battery before the current time step.
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

    ``rho``: Air density (kg/m^3).
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
