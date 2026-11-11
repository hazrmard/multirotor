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
import numpy as np

from . import xp_numba as xp
from .controller.cascade import cascaded_step
from .dynamics import (
    N_BODY_STATES, rhs_body, rhs_body_with_motors, rhs_body_with_motors_and_battery)
from .lazy import build
from .physics import battery_end_step, forces_torques
from .tuples import ControllerState, VehicleState



def rk4_step(f, t, x, dt, args):
    "One RK4 step of ``dx/dt = f(t, x, args)``. Returns the new state and ``f(t, x)``."
    k1 = f(t, x, args)
    k2 = f(t + dt / 2., x + dt / 2. * k1, args)
    k3 = f(t + dt / 2., x + dt / 2. * k2, args)
    k4 = f(t + dt, x + dt * k3, args)
    return x + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4), k1



def integrate(f, t, x, dt, substeps, args):
    """
    Advance ``x`` by ``dt`` in ``substeps`` RK4 steps. Returns the new state and
    ``f(t, x)`` at the start. ``substeps`` is a positive integer; with JAX it must
    be static (close over it rather than tracing it).
    """
    h = dt / substeps
    x1, dxdt = rk4_step(f, t, x, h, args)
    return xp.fold(_substep, x1, substeps - 1, (f, t, h, args)), dxdt



def _substep(x, i, c):
    "The substep after the first, ``i + 1``."
    f, t, h, args = c
    return rk4_step(f, t + (i + 1) * h, x, h, args)[0]



def _rhs_body(t, x, a):
    return rhs_body(t, x, a[0], a[1], a[2], a[3], a[4], a[5])



def _rhs_motors(t, x, a):
    return rhs_body_with_motors(
        t, x, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10])



def _rhs_battery(t, x, a):
    return rhs_body_with_motors_and_battery(
        t, x, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
        a[12], a[13], a[14])



def step_body(state, u, vehicle, dt, substeps):
    "A vehicle without motors: the speed signals are the propeller speeds."
    forces, torques = forces_torques(
        u, vehicle.prop_params, vehicle.clockwise, vehicle.positions)
    x, dxdt = integrate(
        _rhs_body, state.t, state.x, dt, substeps,
        (forces, torques, vehicle.g, vehicle.mass, vehicle.inertia, vehicle.inertia_inverse))
    return VehicleState(x, state.aux, dxdt[3:6], state.t + dt)



def step_motors(state, u, vehicle, dt, substeps):
    "A vehicle with motors and no battery model; the motors see ``max_voltage``."
    x, dxdt = integrate(
        _rhs_motors, state.t, state.x, dt, substeps,
        (u, vehicle.max_voltage, vehicle.disturbance, vehicle.g, vehicle.mass,
         vehicle.inertia, vehicle.inertia_inverse, vehicle.prop_params, vehicle.motor_params,
         vehicle.clockwise, vehicle.positions))
    return VehicleState(x, state.aux, dxdt[3:6], state.t + dt)



def step_motors_battery(state, u, vehicle, dt, substeps):
    "A vehicle with motors and a battery model, which sags and can cut the motors off."
    x, aux = state.x, state.aux
    charge_drawn, cutoff, overload = aux[0], aux[1] > .5, aux[2] > .5
    x_ext = xp.concatenate((x, xp.array((0.,))))
    x_new, dxdt = integrate(
        _rhs_battery, state.t, x_ext, dt, substeps,
        (u, cutoff, vehicle.disturbance, vehicle.g, vehicle.mass, vehicle.inertia,
         vehicle.inertia_inverse, vehicle.prop_params, vehicle.motor_params,
         vehicle.clockwise, vehicle.positions, vehicle.battery_params, vehicle.ocv_soc,
         vehicle.ocv_voltage, charge_drawn))
    n = x.shape[0]
    charge_drawn, _, _, cutoff, overload, _ = battery_end_step(
        x_new[n], charge_drawn, cutoff, overload, x_new[N_BODY_STATES:n], u,
        vehicle.motor_params, vehicle.battery_params, vehicle.ocv_soc, vehicle.ocv_voltage, dt)
    return VehicleState(
        x_new[:n], xp.array((charge_drawn, cutoff * 1., overload * 1.)), dxdt[3:6], state.t + dt)



def _speed(i, c):
    return xp.sqrt_nonneg(c[0][i])



def allocate(alloc_inverse, thrust_torques):
    "Propeller speeds that produce a thrust and three torques (`Multirotor.allocate_control`)."
    wanted = alloc_inverse @ thrust_torques
    return xp.map_(_speed, wanted.shape[0], (wanted,))



def _drive(carry, i, c):
    "The controller's action and the propeller speeds it asks for."
    controller_x, memo, state = carry
    vehicle, ctrl, targets, dt, n0 = c[0], c[1], c[2], c[3], c[5]
    controller_x, memo, action = cascaded_step(
        ctrl.params, controller_x, memo, targets[i], state.x[:N_BODY_STATES], state.acceleration,
        vehicle.inertia, vehicle.mass, vehicle.mass * vehicle.g, dt, i + n0, ctrl.steps[0],
        ctrl.steps[1], ctrl.steps[2], xp.array((0., 0.)), False, ctrl.feedforward_weight,
        ctrl.leashing, ctrl.square_root_position, ctrl.square_root_attitude)
    return controller_x, memo, action, allocate(ctrl.alloc_inverse, action)



def _record(state, action):
    return xp.concatenate((state.x[:N_BODY_STATES], action))



def _loop_step_body(carry, i, c):
    controller_x, memo, action, u = _drive(carry, i, c)
    state = step_body(carry[2], u, c[0], c[3], c[4])
    return (controller_x, memo, state), _record(state, action)



def _loop_step_motors(carry, i, c):
    controller_x, memo, action, u = _drive(carry, i, c)
    state = step_motors(carry[2], u, c[0], c[3], c[4])
    return (controller_x, memo, state), _record(state, action)



def _loop_step_motors_battery(carry, i, c):
    controller_x, memo, action, u = _drive(carry, i, c)
    state = step_motors_battery(carry[2], u, c[0], c[3], c[4])
    return (controller_x, memo, state), _record(state, action)



def _carry0(state0, start):
    "The carry of the scan, and the number of steps taken before it."
    if start is None:
        zeros = (xp.array((0., 0.)), xp.array((0.,)), xp.array((0., 0., 0.)))
        return (xp.array(((0.,) * 12,) * 7), zeros, state0), 0
    return (start.x, start.memo, state0), start.n



def _result(carry, n0, steps):
    "The final controller and vehicle states, as the tuples that a rollout starts from."
    return ControllerState(carry[0], carry[1], n0 + steps), carry[2]



def closed_loop_body(state0, vehicle, ctrl, targets, dt, substeps, start=None):
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
        _loop_step_body, carry0, targets.shape[0], (vehicle, ctrl, targets, dt, substeps, n0))
    return _result(carry, n0, targets.shape[0]), trajectory



def closed_loop_motors(state0, vehicle, ctrl, targets, dt, substeps, start=None):
    "`closed_loop_body` for a vehicle with motors (`MotorsVehicle`)."
    carry0, n0 = _carry0(state0, start)
    carry, trajectory = xp.scan(
        _loop_step_motors, carry0, targets.shape[0], (vehicle, ctrl, targets, dt, substeps, n0))
    return _result(carry, n0, targets.shape[0]), trajectory



def closed_loop_motors_battery(state0, vehicle, ctrl, targets, dt, substeps, start=None):
    "`closed_loop_body` for a vehicle with motors and a battery (`BatteryVehicle`)."
    carry0, n0 = _carry0(state0, start)
    carry, trajectory = xp.scan(
        _loop_step_motors_battery, carry0, targets.shape[0],
        (vehicle, ctrl, targets, dt, substeps, n0))
    return _result(carry, n0, targets.shape[0]), trajectory



NAMES = (
    'rk4_step', 'integrate', '_substep', '_rhs_body', '_rhs_motors', '_rhs_battery', 'step_body', 'step_motors',
    'step_motors_battery', '_speed', 'allocate', '_drive', '_record', '_loop_step_body', '_loop_step_motors',
    '_loop_step_motors_battery', '_carry0', '_result', 'closed_loop_body', 'closed_loop_motors',
    'closed_loop_motors_battery')
SOURCE = {name: globals()[name] for name in NAMES}
NOT_INLINED = ('step_body', 'step_motors', 'step_motors_battery')   # large, and called once per step
globals().update(vars(build(SOURCE, NAMES, 'numba', globals(), NOT_INLINED)))



def build_for(backend: str):
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
