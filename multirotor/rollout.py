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

The state of a vehicle is a tuple ``(x, aux, acc)``:

* ``x``: the 12 body states, then the motor speeds if the vehicle has motors.
* ``aux``: ``(charge drawn, cut off, overloaded)`` of the battery, as floats; a
  placeholder otherwise.
* ``acc``: the body-frame acceleration at the start of the last step, which the
  controller's leash reads.

Three step functions share the signature
``step(state, u, plant, t, dt, substeps)``: `step_body` (no motors),
`step_motors` and `step_motors_battery`. The ``plant`` tuple is built by
`plant_arguments`, which reads a `Multirotor`.
"""
import numpy as np

from . import xp_numba as xp
from .controller.cascade import cascaded_step
from .dynamics import (
    N_BODY_STATES, rhs_body, rhs_body_with_motors, rhs_body_with_motors_and_battery)
from .lazy import build
from .physics import battery_end_step, forces_torques



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



def step_body(state, u, plant, t, dt, substeps):
    "A vehicle without motors: the speed signals are the propeller speeds."
    g, mass, inertia, inertia_inverse, prop_params, clockwise, positions = plant[:7]
    forces, torques = forces_torques(u, prop_params, clockwise, positions)
    x, dxdt = integrate(
        _rhs_body, t, state[0], dt, substeps,
        (forces, torques, g, mass, inertia, inertia_inverse))
    return x, state[1], dxdt[3:6]



def step_motors(state, u, plant, t, dt, substeps):
    "A vehicle with motors and no battery model; the motors see ``max_voltage``."
    (g, mass, inertia, inertia_inverse, prop_params, clockwise, positions,
     motor_params, max_voltage, disturbance) = plant[:10]
    x, dxdt = integrate(
        _rhs_motors, t, state[0], dt, substeps,
        (u, max_voltage, disturbance, g, mass, inertia, inertia_inverse,
         prop_params, motor_params, clockwise, positions))
    return x, state[1], dxdt[3:6]



def step_motors_battery(state, u, plant, t, dt, substeps):
    "A vehicle with motors and a battery model, which sags and can cut the motors off."
    (g, mass, inertia, inertia_inverse, prop_params, clockwise, positions,
     motor_params, max_voltage, disturbance, battery_params, ocv_soc,
     ocv_voltage) = plant[:13]
    x, aux = state[0], state[1]
    charge_drawn, cutoff, overload = aux[0], aux[1] > .5, aux[2] > .5
    x_ext = xp.concatenate((x, xp.array((0.,))))
    x_new, dxdt = integrate(
        _rhs_battery, t, x_ext, dt, substeps,
        (u, cutoff, disturbance, g, mass, inertia, inertia_inverse, prop_params,
         motor_params, clockwise, positions, battery_params, ocv_soc, ocv_voltage,
         charge_drawn))
    n = x.shape[0]
    charge_drawn, _, _, cutoff, overload, _ = battery_end_step(
        x_new[n], charge_drawn, cutoff, overload, x_new[N_BODY_STATES:n], u,
        motor_params, battery_params, ocv_soc, ocv_voltage, dt)
    return x_new[:n], xp.array((charge_drawn, cutoff * 1., overload * 1.)), dxdt[3:6]



def _speed(i, c):
    return xp.sqrt_nonneg(c[0][i])



def allocate(alloc_inverse, thrust_torques):
    "Propeller speeds that produce a thrust and three torques (`Multirotor.allocate_control`)."
    wanted = alloc_inverse @ thrust_torques
    return xp.map_(_speed, wanted.shape[0], (wanted,))



def _drive(carry, i, c):
    "The controller's action and the propeller speeds it asks for."
    controller_x, memo, state = carry
    plant, ctrl, targets, dt = c[0], c[1], c[2], c[3]
    (cparams, alloc_inverse, steps, feedforward_weight, leashing, square_root_position,
     square_root_attitude) = ctrl
    g, mass, inertia = plant[0], plant[1], plant[2]
    controller_x, memo, action = cascaded_step(
        cparams, controller_x, memo, targets[i], state[0][:N_BODY_STATES], state[2], inertia,
        mass, mass * g, dt, i, steps[0], steps[1], steps[2], xp.array((0., 0.)), False,
        feedforward_weight, leashing, square_root_position, square_root_attitude)
    return controller_x, memo, action, allocate(alloc_inverse, action)



def _record(state, action):
    return xp.concatenate((state[0][:N_BODY_STATES], action))



def _loop_step_body(carry, i, c):
    controller_x, memo, action, u = _drive(carry, i, c)
    state = step_body(carry[2], u, c[0], i * c[3], c[3], c[4])
    return (controller_x, memo, state), _record(state, action)



def _loop_step_motors(carry, i, c):
    controller_x, memo, action, u = _drive(carry, i, c)
    state = step_motors(carry[2], u, c[0], i * c[3], c[3], c[4])
    return (controller_x, memo, state), _record(state, action)



def _loop_step_motors_battery(carry, i, c):
    controller_x, memo, action, u = _drive(carry, i, c)
    state = step_motors_battery(carry[2], u, c[0], i * c[3], c[3], c[4])
    return (controller_x, memo, state), _record(state, action)



def _carry0(state0):
    return (
        xp.array(((0.,) * 12,) * 7),
        (xp.array((0., 0.)), xp.array((0.,)), xp.array((0., 0., 0.))), state0)



def closed_loop_body(state0, plant, ctrl, targets, dt, substeps):
    """
    Run the controller and a vehicle without motors for ``len(targets)`` steps, from
    rest. `closed_loop_motors` and `closed_loop_motors_battery` are the same for a
    vehicle with motors, and with motors and a battery.

    Parameters
    ----------
    state0 : tuple
        The vehicle's ``(x, aux, acc)`` at the start.
    plant : tuple
        The vehicle's arguments, from `plant_arguments`.
    ctrl : tuple
        The controller's arguments, from `controller_arguments`.
    targets : np.ndarray
        The target of each step: x, y, z and yaw, shape (n, 4), n at least 1.
    dt : float
        The simulation time step.
    substeps : int
        RK4 steps per simulation step.

    Returns
    -------
    carry : tuple
        The final controller state, memo and vehicle state.
    trajectory : np.ndarray
        Per step, the vehicle's 12 body states after the step and the 4 controller
        outputs (thrust and torques) that drove it, shape (n, 16).
    """
    return xp.scan(
        _loop_step_body, _carry0(state0), targets.shape[0], (plant, ctrl, targets, dt, substeps))



def closed_loop_motors(state0, plant, ctrl, targets, dt, substeps):
    "`closed_loop_body` for a vehicle with motors."
    return xp.scan(
        _loop_step_motors, _carry0(state0), targets.shape[0], (plant, ctrl, targets, dt, substeps))



def closed_loop_motors_battery(state0, plant, ctrl, targets, dt, substeps):
    "`closed_loop_body` for a vehicle with motors and a battery."
    return xp.scan(
        _loop_step_motors_battery, _carry0(state0), targets.shape[0],
        (plant, ctrl, targets, dt, substeps))



NAMES = (
    'rk4_step', 'integrate', '_substep', '_rhs_body', '_rhs_motors', '_rhs_battery', 'step_body', 'step_motors',
    'step_motors_battery', '_speed', 'allocate', '_drive', '_record', '_loop_step_body', '_loop_step_motors',
    '_loop_step_motors_battery', '_carry0', 'closed_loop_body', 'closed_loop_motors',
    'closed_loop_motors_battery')
SOURCE = {name: globals()[name] for name in NAMES}
NOT_INLINED = ('step_body', 'step_motors', 'step_motors_battery')   # large, and called once per step
globals().update(vars(build(SOURCE, NAMES, 'numba', globals(), NOT_INLINED)))



def plant_arguments(vehicle):
    """
    The step function's name, the vehicle's state and its ``plant`` tuple, read
    from a `multirotor.simulation.Multirotor`. Arrays are float64 copies.

    Returns
    -------
    kind : str
        ``'body'``, ``'motors'`` or ``'motors_battery'``: the vehicle's
        ``step_<kind>`` function.
    state : tuple
        ``(x, aux, acc)`` from the vehicle's current state.
    plant : tuple
        The arguments of the step function.

    Raises
    ------
    ValueError
        For blade-element thrust.
    """
    if not vehicle._data.use_thrust_constant:
        raise ValueError('Blade-element thrust is not supported.')
    p = vehicle._kernel_params
    plant = (
        p.g, p.mass, np.array(p.inertia_matrix, float), np.array(p.inertia_matrix_inverse, float),
        np.array(p.prop_params, float), np.array(p.clockwise, float), np.array(p.positions, float))
    x = np.array(vehicle.state, float)
    aux = np.zeros(3)
    if vehicle._data.motors is None:
        return 'body', (x, aux, np.zeros(3)), plant
    plant += (
        np.array(p.motor_params, float), float(vehicle.battery.motor_voltage_limit), np.zeros(6))
    x = np.concatenate((x, vehicle._data.motors.speed.astype(float)))
    if not vehicle.battery.has_model:
        return 'motors', (x, aux, np.zeros(3)), plant
    b = vehicle.battery
    plant += (np.array(b._params_vector, float), np.array(b._ocv_soc, float), np.array(b._ocv_voltage, float))
    aux = np.array((b.charge_drawn, float(b.cutoff), float(b.overload)))
    return 'motors_battery', (x, aux, np.zeros(3)), plant



def controller_arguments(controller, vehicle):
    "The ``ctrl`` tuple of the ``closed_loop_<kind>`` functions, read from a `multirotor.controller.Controller`."
    return (
        np.array(controller._data.params, float), np.array(vehicle.alloc_inverse, float),
        (controller.steps_p, controller.steps_a, controller.steps_z),
        float(controller.feedforward_weight), bool(controller.ctrl_p.leashing),
        bool(controller.ctrl_p.square_root_scaling), bool(controller.ctrl_a.square_root_scaling))



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
