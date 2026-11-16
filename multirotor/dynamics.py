"""
Right-hand sides of the vehicle's equations of motion, one per vehicle
configuration. Each returns ``dx/dt`` for the whole integrated state ``x``, and takes
the vehicle's parameters as one of the tuples of `multirotor.tuples`.

This module is the only code that knows the layout of ``x``. It slices ``x``,
calls the subsystem equations in `multirotor.physics`, and stacks their results.
The equations do not depend on time.

The state ``x`` and the functions each right-hand side calls. An arrow goes from a
caller to the function it calls::

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

`multirotor.rollout` integrates them with RK4. Vehicles with blade-element thrust are
not covered here: their thrust needs a numerical solve at every evaluation, which
`multirotor.simulation.Multirotor` does in numpy.
"""
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
from jax import lax

from .physics import (
    BATTERY_CAPACITY, BATTERY_INTERNAL_RESISTANCE, battery_ocv, battery_operating_point,
    body_derivative, forces_torques, motor_acc)
from .tuples import BatteryVehicle, BodyVehicle, MotorsVehicle

Array = jax.Array

N_BODY_STATES = 12
"The first 12 elements of the integrated state are the body states."


class BatteryState(NamedTuple):
    """
    The output of `battery_state`: the terminal voltage, the battery current, and the motors'
    voltage limit.
    """
    terminal_voltage: Array
    current: Array
    voltage_limit: Array



def rhs_body(
    x: Array, forces: Array, torques: Array,
    vehicle: Union[BodyVehicle, MotorsVehicle, BatteryVehicle]
) -> Array:
    """
    Rate of change of the 12 body states under forces (N) and torques (N m) in the body
    frame that are fixed over the time step, each shape (3,).
    """
    return body_derivative(
        forces, torques, x, vehicle.g, vehicle.mass, vehicle.inertia, vehicle.inertia_inverse)



def rhs_motor_only(speed: Array, u: Array, max_voltage: Array, motor_params: Array) -> Array:
    """
    Rate of change of the motor speeds (rad/s), for motors stepped on their own. ``u`` are
    the speed signals, shape (n,).
    """
    return motor_acc(speed, u, max_voltage, motor_params)



def rhs_body_with_motors(
    x: Array, u: Array, max_voltage: Array, vehicle: MotorsVehicle
) -> Array:
    """
    Rate of change of the body and its motors. They evolve together: the forces at each
    evaluation come from the motor speeds at that instant.

    ``x``: The 12 body states, then the ``n`` motor speeds, shape (12 + n,).
    ``u``: Speed signals, shape (n,).
    """
    body_state = x[:N_BODY_STATES]
    speeds = x[N_BODY_STATES:]
    forces, torques = forces_torques(
        speeds, vehicle.prop_params, vehicle.clockwise, vehicle.positions)
    return jnp.concatenate((
        rhs_body(
            body_state, forces + vehicle.disturbance[:3], torques + vehicle.disturbance[3:],
            vehicle),
        motor_acc(speeds, u, max_voltage, vehicle.motor_params)))



def battery_state(
    speeds: Array, u: Array, charge: Array, cutoff: Array, vehicle: BatteryVehicle
) -> BatteryState:
    """
    The battery's terminal voltage (V) and current (A), and the voltage limit it puts on
    the motors. A battery that is cut off or empty supplies no current, and the motors get
    no voltage.

    ``charge``: Total charge (C) drawn from the battery since it was full.
    """
    battery_params = vehicle.battery_params
    soc = 1. - charge / battery_params[BATTERY_CAPACITY]
    open_circuit_voltage = battery_ocv(soc, vehicle.ocv_soc, vehicle.ocv_voltage)

    def no_current() -> BatteryState:
        zero = jnp.zeros_like(open_circuit_voltage)
        return BatteryState(open_circuit_voltage, zero, zero)

    def supplying() -> BatteryState:
        point = battery_operating_point(
            speeds, u, open_circuit_voltage, battery_params[BATTERY_INTERNAL_RESISTANCE],
            vehicle.motor_params)
        return BatteryState(
            point.terminal_voltage, point.power / point.terminal_voltage,
            point.terminal_voltage)

    return lax.cond(cutoff | (soc <= 0.), no_current, supplying)



def rhs_body_with_motors_and_battery(
    x: Array, u: Array, cutoff: Array, charge_drawn: Array, vehicle: BatteryVehicle
) -> Array:
    """
    Like `rhs_body_with_motors`, with a battery model. The battery's terminal voltage
    limits the motors' voltage, and sags with the current they draw and as the battery
    discharges.

    ``x``: The 12 body states, the ``n`` motor speeds, and the charge (C) drawn from the
      battery during the current time step, shape (13 + n,).
    ``charge_drawn``: Charge (C) drawn from the battery before the current time step.
    """
    n = u.shape[0]
    body_state = x[:N_BODY_STATES]
    speeds = x[N_BODY_STATES:N_BODY_STATES + n]
    battery = battery_state(speeds, u, charge_drawn + x[N_BODY_STATES + n], cutoff, vehicle)
    forces, torques = forces_torques(
        speeds, vehicle.prop_params, vehicle.clockwise, vehicle.positions)
    return jnp.concatenate((
        rhs_body(
            body_state, forces + vehicle.disturbance[:3], torques + vehicle.disturbance[3:],
            vehicle),
        motor_acc(speeds, u, battery.voltage_limit, vehicle.motor_params),
        battery.current[None]))
