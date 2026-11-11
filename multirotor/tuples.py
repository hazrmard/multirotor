"""
The named tuples that `multirotor.rollout` takes and returns.

Both backends read them as they are: numba types a named tuple as a value on the
stack, and JAX treats it as a pytree that it can differentiate, and returns the
gradient in the same type, so ``grad(loss)(vehicle).mass`` is the gradient with
respect to the mass. A kernel reads ``vehicle.mass`` where it once read ``vehicle[1]``.

The imperative classes convert to and from these tuples, and nowhere else: see
`multirotor.simulation.Multirotor.rollout_params` and
`multirotor.controller.Controller.rollout_params`. The matrices inside them keep the
layout of the imperative classes. A row of ``prop_params`` is named by the ``PROP_*``
constants of `multirotor.physics`, and a row of ``ControllerParams.params`` by the ``PID_*``
constants of `multirotor.controller.data`.

The values can be numpy or JAX arrays. A named tuple is immutable. To change one value of
a tuple ``v``, make a new one with ``v._replace(mass=2.)``. That works with numpy and
JAX, but not inside numba-compiled code.

``inertia_inverse`` is a field of its own, so that a step does not invert the matrix. To
differentiate with respect to the inertia, replace both fields:
``v._replace(inertia=m, inertia_inverse=jnp.linalg.inv(m))``. Replacing only ``inertia``
leaves the dynamics with the old inverse, and the gradient is silently wrong.
"""
from typing import NamedTuple, Union

import numpy as np


class BodyVehicle(NamedTuple):
    """
    The parameters of a vehicle without motors. ``prop_params`` is the propeller matrix,
    ``clockwise`` the sign of each propeller's spin and ``positions`` the propellers' positions
    relative to the center of mass, shape (3, n).
    """
    g: float
    mass: float
    inertia: np.ndarray
    inertia_inverse: np.ndarray
    prop_params: np.ndarray
    clockwise: np.ndarray
    positions: np.ndarray


class MotorsVehicle(NamedTuple):
    """
    The parameters of a vehicle with motors: those of `BodyVehicle` and the motor matrix,
    then the motors' voltage limit and the disturbance (force, torque) on the body.
    """
    g: float
    mass: float
    inertia: np.ndarray
    inertia_inverse: np.ndarray
    prop_params: np.ndarray
    motor_params: np.ndarray
    clockwise: np.ndarray
    positions: np.ndarray
    max_voltage: float
    disturbance: np.ndarray


class BatteryVehicle(NamedTuple):
    "The parameters of a vehicle with motors and a battery model."
    g: float
    mass: float
    inertia: np.ndarray
    inertia_inverse: np.ndarray
    prop_params: np.ndarray
    motor_params: np.ndarray
    clockwise: np.ndarray
    positions: np.ndarray
    battery_params: np.ndarray
    ocv_soc: np.ndarray
    ocv_voltage: np.ndarray
    max_voltage: float
    disturbance: np.ndarray


# The fields before ``max_voltage`` are the parameters, in the order that the right-hand
# sides in `multirotor.dynamics` take them after their per-step arguments, so that
# ``vehicle[:-2]`` can be splatted into them. ``max_voltage`` and ``disturbance`` are the
# last two fields of the types with motors, and a right-hand side takes them before the
# parameters. See `rhs_arguments`.

KINDS = {BodyVehicle: 'body', MotorsVehicle: 'motors', BatteryVehicle: 'motors_battery'}
"The ``closed_loop_<kind>`` function that takes each type of vehicle."


class VehicleState(NamedTuple):
    """
    The state of a vehicle:

    * ``x``: the 12 body states, then the motor speeds if the vehicle has motors.
    * ``aux``: ``(charge drawn, cut off, overloaded)`` of the battery, as floats; a
      placeholder otherwise.
    * ``acceleration``: the body-frame acceleration at the start of the last step, which
      the controller's leash reads.
    * ``t``: the time.
    """
    x: np.ndarray
    aux: np.ndarray
    acceleration: np.ndarray
    t: float


class Steps(NamedTuple):
    """
    The period of each multi-rate stage of the cascade, in simulation steps: the position stage,
    the attitude group (velocity, attitude and rate stages) and the altitude group (altitude and
    climb stages). Each is a positive integer.
    """
    position: int
    attitude: int
    altitude: int


class Memo(NamedTuple):
    """
    The outputs of three stages, which the cascade keeps between steps: the velocity reference
    that the position stage gives the velocity stage, shape (2,); the climb stage's thrust,
    shape (1,); and the rate stage's torques, shape (3,). A stage that does not run at a step
    repeats its output.
    """
    ref_vel: np.ndarray
    thrust: np.ndarray
    torques: np.ndarray


class ControllerParams(NamedTuple):
    """
    The parameters of a `multirotor.controller.Controller`: the matrix of the stages'
    parameters, the inverse of the vehicle's control allocation matrix, the periods of the
    stages (`Steps`), and the options.
    """
    params: np.ndarray
    alloc_inverse: np.ndarray
    steps: Steps
    feedforward_weight: float
    leashing: bool
    square_root_position: bool
    square_root_attitude: bool


class ControllerState(NamedTuple):
    """
    The state of a controller: the matrix of the stages' state ``x``, the `Memo` of the
    outputs that the stages running at their own period hold between their runs, and the number
    of steps ``n`` taken. ``n`` sets the phase of those stages.
    """
    x: np.ndarray
    memo: Memo
    n: int


def rhs_arguments(
    vehicle: Union[MotorsVehicle, BatteryVehicle], u: np.ndarray, limit: Union[float, bool],
    disturbance: np.ndarray


) -> tuple:
    """
    The arguments, after ``t`` and ``x``, of the right-hand side in `multirotor.dynamics` for a
    vehicle with motors.

    Parameters
    ----------
    vehicle : MotorsVehicle or BatteryVehicle
        The vehicle's parameters.
    u : np.ndarray
        The speed signals, shape (n,).
    limit : float or bool
        The motors' voltage limit, or with a battery model the battery's cutoff flag.
    disturbance : np.ndarray
        The force and torque disturbing the body, shape (6,).

    Returns
    -------
    tuple
        ``u``, ``limit``, ``disturbance``, then the vehicle's parameters.
    """
    return (u, limit, disturbance) + tuple(vehicle[:-2])


def kind_of(vehicle: Union[BodyVehicle, MotorsVehicle, BatteryVehicle]) -> str:
    """
    The type of a vehicle's parameters, as the name of its rollout function.

    Parameters
    ----------
    vehicle : BodyVehicle, MotorsVehicle or BatteryVehicle
        The vehicle's parameters.

    Returns
    -------
    str
        ``'body'``, ``'motors'`` or ``'motors_battery'``: the ``closed_loop_<kind>`` function
        that takes it.
    """
    return KINDS[type(vehicle)]
