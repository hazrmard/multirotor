"""
Compiled loops that advance a fleet of ``K`` vehicles. Each loop visits the
vehicles one by one and calls the kernel that advances a single vehicle, so a
vehicle's arithmetic is defined once, in `multirotor.physics`,
`multirotor.dynamics` and `multirotor.coords`.

Every array has the vehicle as its first axis, for example the propeller
parameters have shape ``(K, N_PROP_PARAMS, n)``. A vehicle's slice, such as
``prop_params[k]``, is contiguous and has the shape and type that the
single-vehicle kernel takes.

The right-hand sides take the integrated state of the whole fleet as one flat
vector, the ``K`` vehicles' states one after the other. The flatness is not for
speed: an integrator such as `scipy.integrate.odeint` passes and expects a 1-D
state. Inside, ``x.reshape(K, -1)`` is a free view (the rows are contiguous, so
nothing is copied) and the result is flattened the same way. Each vehicle's
state has the layout of `multirotor.dynamics`.

Which loop calls which kernel::

    fleet_rhs_body -------------------------> dynamics.rhs_body
    fleet_rhs_body_with_motors -------------> dynamics.rhs_body_with_motors
    fleet_rhs_body_with_motors_and_battery -> dynamics.rhs_body_with_motors_and_battery
    fleet_forces_torques -------------------> physics.forces_torques
    fleet_motor_electrical -----------------> physics.motor_electrical
    fleet_battery_end_step -----------------> physics.battery_end_step
    fleet_euler_rate -----------------------> coords.angular_to_euler_rate
    fleet_inertial_velocity ----------------> coords.body_to_inertial
"""
import numpy as np
from numba import njit

from .coords import angular_to_euler_rate, body_to_inertial, direction_cosine_matrix
from .dynamics import (
    rhs_body, rhs_body_with_motors, rhs_body_with_motors_and_battery)
from .physics import battery_end_step, forces_torques, motor_electrical



@njit
def fleet_rhs_body(
    t: float, x: np.ndarray, forces: np.ndarray, torques: np.ndarray, g: float,
    mass: np.ndarray, inertia_matrix: np.ndarray, inertia_matrix_inverse: np.ndarray
) -> np.ndarray:
    """
    Rate of change of the body states of every vehicle under forces and torques
    that are fixed over the time step. `multirotor.dynamics.rhs_body` for the fleet.

    ``x``: The body states of all vehicles, flat (see the module docstring), shape (K *
      12,).
    ``forces``, ``torques``: Forces (N) and torques (N m) in each body frame, shape (K, 3).
    ``mass``: Mass of each vehicle, shape (K,).
    ``inertia_matrix``, ``inertia_matrix_inverse``: Inertia matrix of each vehicle and its
      inverse, shape (K, 3, 3).
    Returns: The rate of change of ``x``, flat, shape (K * 12,).
    """
    k_vehicles = forces.shape[0]
    states = x.reshape(k_vehicles, 12)
    dxdt = np.empty((k_vehicles, 12))
    for k in range(k_vehicles):
        dxdt[k] = rhs_body(
            t, states[k], forces[k], torques[k], g, mass[k], inertia_matrix[k],
            inertia_matrix_inverse[k])
    return dxdt.reshape(-1)



@njit
def fleet_rhs_body_with_motors(
    t: float, x: np.ndarray, u: np.ndarray, max_voltage: np.ndarray,
    disturbance: np.ndarray, g: float, mass: np.ndarray, inertia_matrix: np.ndarray,
    inertia_matrix_inverse: np.ndarray, prop_params: np.ndarray,
    motor_params: np.ndarray, clockwise: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """
    Rate of change of the body and the motors of every vehicle.
    `multirotor.dynamics.rhs_body_with_motors` for the fleet.

    ``x``: The integrated state of all vehicles, flat, shape (K * (12 + n),).
    ``u``: Speed signals, shape (K, n).
    ``max_voltage``: The maximum voltage the power source of each vehicle can supply, shape
      (K,).
    ``disturbance``: External forces and torques in each body frame, shape (K, 6).
    ``mass``: Mass of each vehicle, shape (K,).
    ``inertia_matrix``, ``inertia_matrix_inverse``: Inertia matrix of each vehicle and its
      inverse, shape (K, 3, 3).
    ``prop_params``: Propeller parameters, shape (K, N_PROP_PARAMS, n).
    ``motor_params``: Motor parameters, shape (K, N_MOTOR_PARAMS, n).
    ``clockwise``: Direction of rotation of each propeller, shape (K, n).
    ``positions``: Position of each propeller relative to the center of mass, shape (K, 3,
      n).
    """
    k_vehicles, n = u.shape
    states = x.reshape(k_vehicles, 12 + n)
    dxdt = np.empty((k_vehicles, 12 + n))
    for k in range(k_vehicles):
        dxdt[k] = rhs_body_with_motors(
            t, states[k], u[k], max_voltage[k], disturbance[k], g, mass[k],
            inertia_matrix[k], inertia_matrix_inverse[k], prop_params[k],
            motor_params[k], clockwise[k], positions[k])
    return dxdt.reshape(-1)



@njit
def fleet_rhs_body_with_motors_and_battery(
    t: float, x: np.ndarray, u: np.ndarray, cutoff: np.ndarray,
    disturbance: np.ndarray, g: float, mass: np.ndarray, inertia_matrix: np.ndarray,
    inertia_matrix_inverse: np.ndarray, prop_params: np.ndarray,
    motor_params: np.ndarray, clockwise: np.ndarray, positions: np.ndarray,
    battery_params: np.ndarray, ocv_soc: np.ndarray, ocv_voltage: np.ndarray,
    charge_drawn: np.ndarray
) -> np.ndarray:
    """
    Rate of change of the body, the motors and the battery of every vehicle.
    `multirotor.dynamics.rhs_body_with_motors_and_battery` for the fleet.

    ``t``, ``x``, ``u``, ``disturbance``, ``g``, ``mass``, ``inertia_matrix``,
      ``inertia_matrix_inverse``, ``prop_params``, ``motor_params``, ``clockwise``,
      ``positions``: As in `fleet_rhs_body_with_motors`. Each vehicle's integrated state has
      one more element: the charge drawn during the step. ``x`` has shape (K * (12 + n +
      1),).
    ``cutoff``: Whether each battery has cut off the motors, shape (K,).
    ``battery_params``: Battery parameters, shape (K, N_BATTERY_PARAMS).
    ``ocv_soc``, ``ocv_voltage``: The open-circuit voltage tables, shape (K, m).
    ``charge_drawn``: Charge (C) drawn from each battery before the step, shape (K,).
    """
    k_vehicles, n = u.shape
    states = x.reshape(k_vehicles, 12 + n + 1)
    dxdt = np.empty((k_vehicles, 12 + n + 1))
    for k in range(k_vehicles):
        dxdt[k] = rhs_body_with_motors_and_battery(
            t, states[k], u[k], cutoff[k], disturbance[k], g, mass[k],
            inertia_matrix[k], inertia_matrix_inverse[k], prop_params[k],
            motor_params[k], clockwise[k], positions[k], battery_params[k],
            ocv_soc[k], ocv_voltage[k], charge_drawn[k])
    return dxdt.reshape(-1)



@njit
def fleet_forces_torques(
    speeds: np.ndarray, prop_params: np.ndarray, clockwise: np.ndarray,
    positions: np.ndarray
) -> tuple:
    """
    Total force and torque on the body of every vehicle.
    `multirotor.physics.forces_torques` for the fleet.

    ``speeds``: Propeller speeds (rad/s), shape (K, n).
    ``prop_params``: Propeller parameters, shape (K, N_PROP_PARAMS, n).
    ``clockwise``: Direction of rotation of each propeller, shape (K, n).
    ``positions``: Position of each propeller relative to the center of mass, shape (K, 3,
      n).
    Returns: The forces (N) and torques (N m), each shape (K, 3).
    """
    k_vehicles = speeds.shape[0]
    forces = np.empty((k_vehicles, 3))
    torques = np.empty((k_vehicles, 3))
    for k in range(k_vehicles):
        f, tau = forces_torques(speeds[k], prop_params[k], clockwise[k], positions[k])
        forces[k] = f
        torques[k] = tau
    return forces, torques



@njit
def fleet_motor_electrical(
    speed: np.ndarray, u: np.ndarray, max_voltage: np.ndarray,
    motor_params: np.ndarray, voltage: np.ndarray, current: np.ndarray
) -> None:
    """
    Voltage and current of every motor, written to ``voltage`` and ``current``.
    `multirotor.physics.motor_electrical` for the fleet.

    ``speed``, ``u``: Motor speeds (rad/s) and speed signals, shape (K, n).
    ``max_voltage``: The maximum voltage of each vehicle's power source, shape (K,).
    ``motor_params``: Motor parameters, shape (K, N_MOTOR_PARAMS, n).
    ``voltage``, ``current``: Outputs, shape (K, n).
    """
    for k in range(speed.shape[0]):
        v, i = motor_electrical(speed[k], u[k], max_voltage[k], motor_params[k])
        voltage[k] = v
        current[k] = i



@njit
def fleet_battery_end_step(
    charge: np.ndarray, charge_drawn: np.ndarray, cutoff: np.ndarray,
    overload: np.ndarray, voltage: np.ndarray, current: np.ndarray,
    motor_speeds: np.ndarray, u: np.ndarray, motor_params: np.ndarray,
    battery_params: np.ndarray, ocv_soc: np.ndarray, ocv_voltage: np.ndarray,
    dt: float, motor_voltage_limit: np.ndarray
) -> None:
    """
    Update every battery after an integrated time step.
    `multirotor.physics.battery_end_step` for the fleet.

    ``charge``: Charge (C) drawn during the step, as integrated, shape (K,).
    ``charge_drawn``, ``cutoff``, ``overload``, ``voltage``, ``current``: The batteries'
      state, shape (K,). Updated in place.
    ``motor_speeds``, ``u``: Motor speeds at the end of the step and the speed signals,
      shape (K, n).
    ``motor_params``: Motor parameters, shape (K, N_MOTOR_PARAMS, n).
    ``dt``: The time step (s).
    ``motor_voltage_limit``: Output: the motors' voltage limit for the next step, shape
      (K,).
    """
    for k in range(charge.shape[0]):
        drawn, i, v, cut, over, limit = battery_end_step(
            charge[k], charge_drawn[k], cutoff[k], overload[k], motor_speeds[k], u[k],
            motor_params[k], battery_params[k], ocv_soc[k], ocv_voltage[k], dt)
        charge_drawn[k] = drawn
        current[k] = i
        voltage[k] = v
        cutoff[k] = cut
        overload[k] = over
        motor_voltage_limit[k] = limit



@njit
def fleet_euler_rate(state: np.ndarray) -> np.ndarray:
    """
    Rate of change of the roll, pitch and yaw of every vehicle.

    ``state``: The body states, shape (K, 12).
    Returns: The Euler rates, shape (K, 3).
    """
    rate = np.empty((state.shape[0], 3))
    for k in range(state.shape[0]):
        rate[k] = angular_to_euler_rate(state[k, 9:12], state[k, 6:9])
    return rate



@njit
def fleet_inertial_velocity(state: np.ndarray) -> np.ndarray:
    """
    Velocity of every vehicle in the inertial frame.

    ``state``: The body states, shape (K, 12).
    Returns: The inertial velocities, shape (K, 3).
    """
    velocity = np.empty((state.shape[0], 3))
    for k in range(state.shape[0]):
        dcm = direction_cosine_matrix(state[k, 6], state[k, 7], state[k, 8])
        velocity[k] = body_to_inertial(np.ascontiguousarray(state[k, 3:6]), dcm)
    return velocity
