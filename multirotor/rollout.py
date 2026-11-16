"""
The closed loop of the cascaded controller, control allocation and the vehicle, as pure
functions written for JAX: one step (`step_body`, `step_motors`,
`step_motors_battery`, `control`), or ``n`` steps in one compiled loop (`closed_loop`
and ``closed_loop_<kind>``). They can be compiled with ``jax.jit``, mapped over the
vehicles of a fleet with ``jax.vmap``, and differentiated with respect to any
parameter. `multirotor.simulation.Multirotor` and `multirotor.controller.Controller`
run on the same functions, one step per call.

* The integrator is RK4 with ``substeps`` equal steps per simulation step. The motors'
  dynamics are fast: `multirotor.integrator.DEFAULT_SUBSTEPS` gives a count for each kind
  of vehicle.
* State goes in and out as values. Nothing is changed in place.
* Vehicles with blade-element thrust are not supported (their thrust needs a
  root solve at every evaluation).

Everything goes in and out as the named tuples of `multirotor.tuples`: the vehicle's
parameters (`BodyVehicle`, `MotorsVehicle` or `BatteryVehicle`) and state (`VehicleState`),
and the controller's parameters (`ControllerParams`) and state (`ControllerState`). The
classes make and take them: `Multirotor.rollout_params`, `Multirotor.rollout_state`,
`Controller.rollout_params`, `Controller.rollout_state` and ``set_rollout_state`` of both.
A rollout starts from rest, or from the state of a vehicle and a controller that have been
running, and the state at its end can be written back.

The three step functions share the signature ``step(state, u, vehicle, dt, substeps)``.
"""
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

from .controller.cascade import CascadeStepInputs, cascaded_step
from .controller.data import N_CASCADE_CHANNELS, N_PID_STATES
from .dynamics import (
    N_BODY_STATES, rhs_body, rhs_body_with_motors, rhs_body_with_motors_and_battery)
from .integrator import DEFAULT_SUBSTEPS, integrate  # noqa: F401, DEFAULT_SUBSTEPS is re-exported
from .physics import battery_end_step, forces_torques, sqrt_nonneg
from .tuples import (
    BatteryVehicle, BodyVehicle, ControllerParams, ControllerState, Memo, MotorsVehicle,
    VehicleState)

Array = jax.Array
Vehicle = Union[BodyVehicle, MotorsVehicle, BatteryVehicle]
Step = Callable[[VehicleState, Array, Vehicle, float, int], VehicleState]
"A step function of a vehicle: `step_body`, `step_motors` or `step_motors_battery`."

def step_forces(
    state: VehicleState, forces: Array, torques: Array, vehicle: Vehicle, dt: float,
    substeps: int
) -> VehicleState:
    """
    Advance the body by one step under forces (N) and torques (N m) in the body frame that
    are fixed over the step, each shape (3,). Only the 12 body states of ``state.x`` change.
    """
    body, dxdt = integrate(
        lambda x: rhs_body(x, forces, torques, vehicle), state.x[:N_BODY_STATES], dt, substeps)
    x = jnp.concatenate((body, state.x[N_BODY_STATES:]))
    return VehicleState(x, state.aux, dxdt[3:6], state.t + dt)



def step_body(
    state: VehicleState, u: Array, vehicle: BodyVehicle, dt: float, substeps: int
) -> VehicleState:
    """
    Advance a vehicle without motors by one step: the speed signals ``u``, shape (n,), are
    the propeller speeds (rad/s).
    """
    forces, torques = forces_torques(
        u, vehicle.prop_params, vehicle.clockwise, vehicle.positions)
    return step_forces(
        state, forces + vehicle.disturbance[:3], torques + vehicle.disturbance[3:], vehicle,
        dt, substeps)



def step_motors(
    state: VehicleState, u: Array, vehicle: MotorsVehicle, dt: float, substeps: int
) -> VehicleState:
    """
    Advance a vehicle with motors and no battery model by one step; the motors see
    ``vehicle.max_voltage``. ``state.x`` holds the body states, then the motor speeds.
    """
    x, dxdt = integrate(
        lambda x: rhs_body_with_motors(x, u, vehicle.max_voltage, vehicle), state.x, dt,
        substeps)
    return VehicleState(x, state.aux, dxdt[3:6], state.t + dt)



def step_motors_battery(
    state: VehicleState, u: Array, vehicle: BatteryVehicle, dt: float, substeps: int
) -> VehicleState:
    """
    Advance a vehicle with motors and a battery model by one step. The battery sags and can
    cut the motors off. ``state.aux`` holds the charge drawn (C), and whether the battery is
    cut off and overloaded, as floats.
    """
    x, aux = state.x, state.aux
    charge_drawn, cutoff, overload = aux[0], aux[1] > .5, aux[2] > .5
    # One more integrated state: the charge drawn during this step
    x_new, dxdt = integrate(
        lambda x: rhs_body_with_motors_and_battery(x, u, cutoff, charge_drawn, vehicle),
        jnp.concatenate((x, jnp.zeros(1))), dt, substeps)
    n = x.shape[0]
    end = battery_end_step(
        x_new[n], charge_drawn, cutoff, overload, x_new[N_BODY_STATES:n], u,
        vehicle.motor_params, vehicle.battery_params, vehicle.ocv_soc, vehicle.ocv_voltage, dt)
    aux = jnp.stack((end.charge_drawn, end.cutoff * 1., end.overload * 1.))
    return VehicleState(x_new[:n], aux, dxdt[3:6], state.t + dt)



def allocate(alloc_inverse: Array, thrust_torques: Array) -> Array:
    """
    Propeller speeds (rad/s), shape (n,), that produce a thrust and three torques, shape
    (4,) (`Multirotor.allocate_control`). ``alloc_inverse`` is the inverse of the vehicle's
    control allocation matrix, shape (n, 4).
    """
    return sqrt_nonneg(alloc_inverse @ thrust_torques)



def control(
    controller: ControllerState, state: VehicleState, vehicle: Vehicle,
    ctrl: ControllerParams, target: Array, dt: float
) -> Tuple[ControllerState, Array]:
    """
    One step of the controller for a vehicle in ``state``, towards ``target`` (x, y, z and
    yaw). Returns the controller's new state, and its thrust and three torques, shape (4,).
    """
    inputs = CascadeStepInputs(
        ctrl.params, target, state.x[:N_BODY_STATES], state.acceleration, vehicle.inertia,
        vehicle.mass, vehicle.mass * vehicle.g, dt, ctrl.steps, jnp.zeros(2), False,
        ctrl.feedforward_weight, ctrl.leashing, ctrl.square_root_position,
        ctrl.square_root_attitude)
    x, memo, action = cascaded_step(controller.x, controller.memo, inputs, controller.n)
    return ControllerState(x, memo, controller.n + 1), action



def controller_at_rest() -> ControllerState:
    """The state of a controller that has not run."""
    memo = Memo(jnp.zeros(2), jnp.zeros(1), jnp.zeros(3))
    return ControllerState(jnp.zeros((N_PID_STATES, N_CASCADE_CHANNELS)), memo, 0)



def closed_loop(
    step: Step, state0: VehicleState, vehicle: Vehicle, ctrl: ControllerParams,
    targets: Array, dt: float, substeps: int, start: Optional[ControllerState] = None
) -> Tuple[Tuple[ControllerState, VehicleState], Array]:
    """
    Run the controller and a vehicle for ``len(targets)`` steps. ``closed_loop_body``,
    ``closed_loop_motors`` and ``closed_loop_motors_battery`` call it with the step function
    of their kind of vehicle.

    Parameters
    ----------
    step : Step
        The step function of the vehicle: `step_body`, `step_motors` or
        `step_motors_battery`.
    state0, vehicle, ctrl, targets, dt, substeps, start
        As in `closed_loop_body`.

    Returns
    -------
    carry : tuple
        The final `ControllerState` and `VehicleState`.
    trajectory : jax.Array
        The body states and controller outputs of each step, shape (n, 16).
    """
    def loop_step(
        carry: Tuple[ControllerState, VehicleState], target: Array
    ) -> Tuple[Tuple[ControllerState, VehicleState], Array]:
        """The controller drives, then the vehicle steps. Returns the trajectory's row."""
        controller, state = carry
        controller, action = control(controller, state, vehicle, ctrl, target, dt)
        state = step(state, allocate(ctrl.alloc_inverse, action), vehicle, dt, substeps)
        return (controller, state), jnp.concatenate((state.x[:N_BODY_STATES], action))

    start = controller_at_rest() if start is None else start
    return lax.scan(loop_step, (start, state0), targets)



def closed_loop_body(
    state0: VehicleState, vehicle: BodyVehicle, ctrl: ControllerParams,
    targets: Array, dt: float, substeps: int, start: Optional[ControllerState] = None
) -> Tuple[Tuple[ControllerState, VehicleState], Array]:
    """
    Run the controller and a vehicle without motors for ``len(targets)`` steps, from
    rest or from a given controller state. `closed_loop_motors` and
    `closed_loop_motors_battery` are the same for a vehicle with motors, and with motors
    and a battery.

    Parameters
    ----------
    state0 : VehicleState
        The vehicle's state at the start.
    vehicle : BodyVehicle
        The vehicle's parameters, from `Multirotor.rollout_params`.
    ctrl : ControllerParams
        The controller's parameters, from `Controller.rollout_params`.
    targets : jax.Array
        The target of each step: x, y, z and yaw, shape (n, 4), n at least 1.
    dt : float
        The simulation time step.
    substeps : int
        RK4 steps per simulation step.
    start : ControllerState, optional
        The controller's state to start from, including the number of steps it has taken,
        which sets the phase of the stages that run every few steps. By default the
        controller starts at rest, with ``n=0``.

    Returns
    -------
    carry : tuple
        The final `ControllerState` and `VehicleState`. They are what ``start`` and
        ``state0`` of the next rollout take, and what ``set_rollout_state`` writes back.
    trajectory : jax.Array
        Per step, the vehicle's 12 body states after the step and the 4 controller
        outputs (thrust and torques) that drove it, shape (n, 16).
    """
    return closed_loop(step_body, state0, vehicle, ctrl, targets, dt, substeps, start)



def closed_loop_motors(
    state0: VehicleState, vehicle: MotorsVehicle, ctrl: ControllerParams,
    targets: Array, dt: float, substeps: int, start: Optional[ControllerState] = None
) -> Tuple[Tuple[ControllerState, VehicleState], Array]:
    """
    `closed_loop_body` for a vehicle with motors (`MotorsVehicle`).

    Parameters
    ----------
    state0, vehicle, ctrl, targets, dt, substeps, start
        As in `closed_loop_body`, with ``vehicle`` a `MotorsVehicle`. The state's ``x`` also
        holds the motor speeds.

    Returns
    -------
    carry : tuple
        The final `ControllerState` and `VehicleState`.
    trajectory : jax.Array
        The body states and controller outputs of each step, shape (n, 16).
    """
    return closed_loop(step_motors, state0, vehicle, ctrl, targets, dt, substeps, start)



def closed_loop_motors_battery(
    state0: VehicleState, vehicle: BatteryVehicle, ctrl: ControllerParams,
    targets: Array, dt: float, substeps: int, start: Optional[ControllerState] = None
) -> Tuple[Tuple[ControllerState, VehicleState], Array]:
    """
    `closed_loop_body` for a vehicle with motors and a battery (`BatteryVehicle`).

    Parameters
    ----------
    state0, vehicle, ctrl, targets, dt, substeps, start
        As in `closed_loop_body`, with ``vehicle`` a `BatteryVehicle`. The state's ``aux``
        holds the battery's charge drawn, cutoff and overload.

    Returns
    -------
    carry : tuple
        The final `ControllerState` and `VehicleState`.
    trajectory : jax.Array
        The body states and controller outputs of each step, shape (n, 16).
    """
    return closed_loop(step_motors_battery, state0, vehicle, ctrl, targets, dt, substeps, start)
