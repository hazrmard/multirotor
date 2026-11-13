"""
A closed-loop rollout that numba runs and JAX differentiates: the cascaded
controller, control allocation, and a fixed-step RK4 integration of the vehicle,
for ``n`` steps. It is written once, in plain Python against ``xp``, and built for
a backend like `multirotor.physics` is (see `multirotor.lazy`).

Differences from `Multirotor.step_speeds` and `Controller.step`:

* The integrator is RK4 with ``substeps`` equal steps per simulation step, not
  LSODA. LSODA adapts its steps to the state and is not differentiable; RK4 is.
  Choose ``substeps`` so that the trajectory matches LSODA's closely enough.
* State goes in and out as values. Nothing is changed in place.
* Vehicles with blade-element thrust are not supported (their thrust needs a
  root solve that is not differentiable here).

Everything goes in and out as the named tuples of `multirotor.tuples`: the vehicle's
parameters (`BodyVehicle`, `MotorsVehicle` or `BatteryVehicle`) and state (`VehicleState`),
and the controller's parameters (`ControllerParams`) and state (`ControllerState`). The
imperative classes make and take them: `Multirotor.rollout_params`,
`Multirotor.rollout_state`, `Controller.rollout_params`, `Controller.rollout_state` and
`set_rollout_state` of both. A rollout starts from rest, or from the state of a vehicle and a
controller that have been running, and the state at its end can be written back.

Three step functions share the signature
``step(state, u, vehicle, dt, substeps)``: `step_body` (no motors),
`step_motors` and `step_motors_battery`.
"""
import types
from typing import Callable, NamedTuple, Optional, Tuple, Union

import numpy as np

from . import xp_numba as xp
from .controller.cascade import CascadeStepInputs, cascaded_step
from .controller.data import N_CASCADE_CHANNELS, N_PID_STATES
from .dynamics import (
    N_BODY_STATES, rhs_body, rhs_body_with_motors, rhs_body_with_motors_and_battery)
from .lazy import build
from .physics import battery_end_step, forces_torques
from .tuples import (
    BatteryVehicle, BodyVehicle, ControllerParams, ControllerState, Memo, MotorsVehicle,
    VehicleState)


Rhs = Callable[[float, np.ndarray, tuple], np.ndarray]
"A right-hand side ``f(t, x, args)`` of ``dx/dt``, with the arguments as one tuple."


class BodyRhsArgs(NamedTuple):
    """
    The right-hand side arguments of a body-only vehicle: the inputs of `rhs_body` after ``t``
    and ``x``.
    """
    forces: np.ndarray
    torques: np.ndarray
    g: float
    mass: float
    inertia: np.ndarray
    inertia_inverse: np.ndarray


class MotorsRhsArgs(NamedTuple):
    """
    The right-hand side arguments of a vehicle with motors: the inputs of
    `rhs_body_with_motors` after ``t`` and ``x``.
    """
    u: np.ndarray
    max_voltage: float
    disturbance: np.ndarray
    g: float
    mass: float
    inertia: np.ndarray
    inertia_inverse: np.ndarray
    prop_params: np.ndarray
    motor_params: np.ndarray
    clockwise: np.ndarray
    positions: np.ndarray


class BatteryRhsArgs(NamedTuple):
    """
    The right-hand side arguments of a vehicle with motors and a battery: the inputs of
    `rhs_body_with_motors_and_battery` after ``t`` and ``x``.
    """
    u: np.ndarray
    cutoff: bool
    disturbance: np.ndarray
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
    charge_drawn: float


class Rk4Substep(NamedTuple):
    """
    One RK4 substep: the right-hand side, its start time, the substep length and the right-
    hand side arguments. The input to `_substep`.
    """
    f: Rhs
    t: float
    h: float
    args: tuple


class RolloutSetup(NamedTuple):
    """
    The setup of a rollout, which does not change from step to step: the vehicle's and the
    controller's parameters, the targets, the time step, the RK4 substeps per step, and the
    number of controller steps taken before the rollout. The constant input to the loop.
    """
    vehicle: Union[BodyVehicle, MotorsVehicle, BatteryVehicle]
    ctrl: ControllerParams
    targets: np.ndarray
    dt: float
    substeps: int
    n0: int


class LoopState(NamedTuple):
    """
    The state of the closed loop that a rollout carries from step to step: the controller's
    state matrix ``x`` and its `Memo`, and the vehicle's `VehicleState`.
    """
    x: np.ndarray
    memo: Memo
    state: VehicleState


class ControllerOutput(NamedTuple):
    """
    The output of the controller in one step of a rollout: its new state and memo, its action,
    and the propeller speeds.
    """
    x: np.ndarray
    memo: Memo
    action: np.ndarray
    u: np.ndarray


def rk4_step(
    f: Rhs, t: float, x: np.ndarray, dt: float, args: tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """One RK4 step of ``dx/dt = f(t, x, args)``."""
    k1 = f(t, x, args)
    k2 = f(t + dt / 2., x + dt / 2. * k1, args)
    k3 = f(t + dt / 2., x + dt / 2. * k2, args)
    k4 = f(t + dt, x + dt * k3, args)
    return x + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4), k1



def integrate(
    f: Rhs, t: float, x: np.ndarray, dt: float, substeps: int, args: tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance ``x`` by ``dt`` in ``substeps`` RK4 steps."""
    h = dt / substeps
    x1, dxdt = rk4_step(f, t, x, h, args)
    return xp.fold(_substep, x1, substeps - 1, Rk4Substep(f, t, h, args)), dxdt



def _substep(x: np.ndarray, i: int, c: Rk4Substep) -> np.ndarray:
    """The substep after the first, ``i + 1``."""
    return rk4_step(c.f, c.t + (i + 1) * c.h, x, c.h, c.args)[0]



def _rhs_body(t: float, x: np.ndarray, a: BodyRhsArgs) -> np.ndarray:
    """`rhs_body` with its arguments after ``t`` and ``x`` as one `BodyRhsArgs`."""
    return rhs_body(t, x, a.forces, a.torques, a.g, a.mass, a.inertia, a.inertia_inverse)



def _rhs_motors(t: float, x: np.ndarray, a: MotorsRhsArgs) -> np.ndarray:
    """`rhs_body_with_motors` with its arguments after ``t`` and ``x`` as one `MotorsRhsArgs`."""
    return rhs_body_with_motors(
        t, x, a.u, a.max_voltage, a.disturbance, a.g, a.mass, a.inertia, a.inertia_inverse,
        a.prop_params, a.motor_params, a.clockwise, a.positions)



def _rhs_battery(t: float, x: np.ndarray, a: BatteryRhsArgs) -> np.ndarray:
    """
    `rhs_body_with_motors_and_battery` with its arguments after ``t`` and ``x`` as one
    `BatteryRhsArgs`.
    """
    return rhs_body_with_motors_and_battery(
        t, x, a.u, a.cutoff, a.disturbance, a.g, a.mass, a.inertia, a.inertia_inverse,
        a.prop_params, a.motor_params, a.clockwise, a.positions, a.battery_params, a.ocv_soc,
        a.ocv_voltage, a.charge_drawn)



def step_body(
    state: VehicleState, u: np.ndarray, vehicle: BodyVehicle, dt: float, substeps: int
) -> VehicleState:
    """
    Advance a vehicle without motors by one step: the speed signals are the propeller
    speeds. The three step functions have this signature.

    ``u``: The speed signals, shape (n,).
    """
    forces, torques = forces_torques(
        u, vehicle.prop_params, vehicle.clockwise, vehicle.positions)
    x, dxdt = integrate(
        _rhs_body, state.t, state.x, dt, substeps,
        BodyRhsArgs(
            forces, torques, vehicle.g, vehicle.mass, vehicle.inertia,
            vehicle.inertia_inverse))
    return VehicleState(x, state.aux, dxdt[3:6], state.t + dt)



def step_motors(
    state: VehicleState, u: np.ndarray, vehicle: MotorsVehicle, dt: float, substeps: int
) -> VehicleState:
    """
    Advance a vehicle with motors and no battery model by one step; the motors see
    ``max_voltage``. The arguments and the result are those of `step_body`.
    """
    x, dxdt = integrate(
        _rhs_motors, state.t, state.x, dt, substeps,
        MotorsRhsArgs(
            u, vehicle.max_voltage, vehicle.disturbance, vehicle.g, vehicle.mass,
            vehicle.inertia, vehicle.inertia_inverse, vehicle.prop_params,
            vehicle.motor_params, vehicle.clockwise, vehicle.positions))
    return VehicleState(x, state.aux, dxdt[3:6], state.t + dt)



def step_motors_battery(
    state: VehicleState, u: np.ndarray, vehicle: BatteryVehicle, dt: float, substeps: int
) -> VehicleState:
    """
    Advance a vehicle with motors and a battery model by one step. The battery sags and can
    cut the motors off. The arguments and the result are those of `step_body`.
    """
    x, aux = state.x, state.aux
    charge_drawn, cutoff, overload = aux[0], aux[1] > .5, aux[2] > .5
    x_ext = xp.concatenate((x, xp.array((0.,))))
    x_new, dxdt = integrate(
        _rhs_battery, state.t, x_ext, dt, substeps,
        BatteryRhsArgs(
            u, cutoff, vehicle.disturbance, vehicle.g, vehicle.mass, vehicle.inertia,
            vehicle.inertia_inverse, vehicle.prop_params, vehicle.motor_params,
            vehicle.clockwise, vehicle.positions, vehicle.battery_params, vehicle.ocv_soc,
            vehicle.ocv_voltage, charge_drawn))
    n = x.shape[0]
    end = battery_end_step(
        x_new[n], charge_drawn, cutoff, overload, x_new[N_BODY_STATES:n], u,
        vehicle.motor_params, vehicle.battery_params, vehicle.ocv_soc, vehicle.ocv_voltage, dt)
    return VehicleState(
        x_new[:n], xp.array((end.charge_drawn, end.cutoff * 1., end.overload * 1.)),
        dxdt[3:6], state.t + dt)



def _speed(i: int, c: tuple) -> float:
    """
    The propeller speed that a squared speed asks for.

    Returns: The speed (rad/s), or 0 if the square is negative.
    """
    return xp.sqrt_nonneg(c[0][i])



def allocate(alloc_inverse: np.ndarray, thrust_torques: np.ndarray) -> np.ndarray:
    """
    Propeller speeds that produce a thrust and three torques (`Multirotor.allocate_control`).

    ``alloc_inverse``: The inverse of the vehicle's control allocation matrix, shape (n, 4).
    ``thrust_torques``: The thrust and the three torques, shape (4,).
    Returns: The propeller speeds (rad/s), shape (n,).
    """
    wanted = alloc_inverse @ thrust_torques
    return xp.map_(_speed, wanted.shape[0], (wanted,))



def _drive(carry: LoopState, i: int, c: RolloutSetup) -> ControllerOutput:
    """The controller's action and the propeller speeds it asks for."""
    vehicle, ctrl = c.vehicle, c.ctrl
    inputs = CascadeStepInputs(
        ctrl.params, c.targets[i], carry.state.x[:N_BODY_STATES], carry.state.acceleration,
        vehicle.inertia, vehicle.mass, vehicle.mass * vehicle.g, c.dt, ctrl.steps,
        xp.zeros(2), False, ctrl.feedforward_weight, ctrl.leashing,
        ctrl.square_root_position, ctrl.square_root_attitude)
    x, memo, action = cascaded_step(carry.x, carry.memo, inputs, i + c.n0)
    return ControllerOutput(x, memo, action, allocate(ctrl.alloc_inverse, action))



def _record(state: VehicleState, action: np.ndarray) -> np.ndarray:
    """
    One row of the trajectory.

    ``action``: The controller's thrust and torques, shape (4,).
    Returns: The 12 body states and the action, shape (16,).
    """
    return xp.concatenate((state.x[:N_BODY_STATES], action))



def _loop_step_body(carry: LoopState, i: int, c: RolloutSetup) -> Tuple[LoopState, np.ndarray]:
    """
    One step of a rollout of a vehicle without motors, as `multirotor.xp_numba.scan` takes
    it: the controller drives, then the vehicle steps.

    Returns: The row of the trajectory, shape (16,).
    """
    drive = _drive(carry, i, c)
    state = step_body(carry.state, drive.u, c.vehicle, c.dt, c.substeps)
    return LoopState(drive.x, drive.memo, state), _record(state, drive.action)



def _loop_step_motors(carry: LoopState, i: int, c: RolloutSetup) -> Tuple[LoopState, np.ndarray]:
    "`_loop_step_body` for a vehicle with motors. The arguments and results are the same."
    drive = _drive(carry, i, c)
    state = step_motors(carry.state, drive.u, c.vehicle, c.dt, c.substeps)
    return LoopState(drive.x, drive.memo, state), _record(state, drive.action)



def _loop_step_motors_battery(carry: LoopState, i: int, c: RolloutSetup) -> Tuple[LoopState, np.ndarray]:
    "`_loop_step_body` for a vehicle with motors and a battery. The arguments and results are the same."
    drive = _drive(carry, i, c)
    state = step_motors_battery(carry.state, drive.u, c.vehicle, c.dt, c.substeps)
    return LoopState(drive.x, drive.memo, state), _record(state, drive.action)



def _carry0(state0: VehicleState, start: Optional[ControllerState]) -> Tuple[LoopState, int]:
    """The carry that a rollout starts from."""
    if start is None:
        memo = Memo(xp.zeros(2), xp.zeros(1), xp.zeros(3))
        return LoopState(xp.zeros((N_PID_STATES, N_CASCADE_CHANNELS)), memo, state0), 0
    return LoopState(start.x, start.memo, state0), start.n



def _result(carry: LoopState, n0: int, steps: int) -> Tuple[ControllerState, VehicleState]:
    """The final controller and vehicle states, as the tuples that a rollout starts from."""
    return ControllerState(carry.x, carry.memo, n0 + steps), carry.state



def closed_loop_body(
    state0: VehicleState, vehicle: BodyVehicle, ctrl: ControllerParams,
    targets: np.ndarray, dt: float, substeps: int, start: Optional[ControllerState] = None
) -> Tuple[Tuple[ControllerState, VehicleState], np.ndarray]:
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
    targets : np.ndarray
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
    trajectory : np.ndarray
        Per step, the vehicle's 12 body states after the step and the 4 controller
        outputs (thrust and torques) that drove it, shape (n, 16).
    """
    carry0, n0 = _carry0(state0, start)
    carry, trajectory = xp.scan(
        _loop_step_body, carry0, targets.shape[0],
        RolloutSetup(vehicle, ctrl, targets, dt, substeps, n0))
    return _result(carry, n0, targets.shape[0]), trajectory



def closed_loop_motors(
    state0: VehicleState, vehicle: MotorsVehicle, ctrl: ControllerParams,
    targets: np.ndarray, dt: float, substeps: int, start: Optional[ControllerState] = None
) -> Tuple[Tuple[ControllerState, VehicleState], np.ndarray]:
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
    trajectory : np.ndarray
        The body states and controller outputs of each step, shape (n, 16).
    """
    carry0, n0 = _carry0(state0, start)
    carry, trajectory = xp.scan(
        _loop_step_motors, carry0, targets.shape[0],
        RolloutSetup(vehicle, ctrl, targets, dt, substeps, n0))
    return _result(carry, n0, targets.shape[0]), trajectory



def closed_loop_motors_battery(
    state0: VehicleState, vehicle: BatteryVehicle, ctrl: ControllerParams,
    targets: np.ndarray, dt: float, substeps: int, start: Optional[ControllerState] = None
) -> Tuple[Tuple[ControllerState, VehicleState], np.ndarray]:
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
    trajectory : np.ndarray
        The body states and controller outputs of each step, shape (n, 16).
    """
    carry0, n0 = _carry0(state0, start)
    carry, trajectory = xp.scan(
        _loop_step_motors_battery, carry0, targets.shape[0],
        RolloutSetup(vehicle, ctrl, targets, dt, substeps, n0))
    return _result(carry, n0, targets.shape[0]), trajectory



NAMES = (
    'rk4_step', 'integrate', '_substep', '_rhs_body', '_rhs_motors', '_rhs_battery', 'step_body', 'step_motors',
    'step_motors_battery', '_speed', 'allocate', '_drive', '_record', '_loop_step_body', '_loop_step_motors',
    '_loop_step_motors_battery', '_carry0', '_result', 'closed_loop_body', 'closed_loop_motors',
    'closed_loop_motors_battery')
SOURCE = {name: globals()[name] for name in NAMES}
NOT_INLINED = ('step_body', 'step_motors', 'step_motors_battery')   # large, and called once per step
globals().update(vars(build(SOURCE, NAMES, 'numba', globals(), NOT_INLINED)))



def build_for(backend: str) -> types.SimpleNamespace:
    """
    The rollout functions built for ``backend`` (``'numba'`` or ``'jax'``), with
    the physics, controller and dynamics they call built for it too.

    Returns
    -------
    types.SimpleNamespace
        `closed_loop_body`, `closed_loop_motors`, `closed_loop_motors_battery`, `step_body`, `step_motors` and `step_motors_battery`, and
        the other functions of this module.
    """
    from .controller import cascade
    from . import coords, dynamics, physics
    built_coords = build(coords.SOURCE, coords.NAMES, backend, vars(coords))
    built_physics = build(physics.SOURCE, physics.NAMES, backend, vars(physics))
    built_cascade = build(
        cascade.SOURCE, cascade.NAMES, backend, {**vars(cascade), **vars(built_coords)},
        cascade.NOT_INLINED)
    built_dynamics = build(
        dynamics.SOURCE, dynamics.NAMES, backend, {**vars(dynamics), **vars(built_physics)})
    return build(SOURCE, NAMES, backend, {
        **globals(), **vars(built_physics), **vars(built_cascade), **vars(built_dynamics)},
        NOT_INLINED)
