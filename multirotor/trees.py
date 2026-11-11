"""
Named trees of the parameters and the state of a vehicle and its controller, for the
functional rollout (`multirotor.rollout`).

The rollout takes plain tuples and matrices. A *tree* is the same values with names:
nested dicts of arrays. JAX differentiates with respect to a tree, so one can write
``grad(loss)(params)`` and read ``params['ctrl_r']['k_p']`` in the result. The functions
come in pairs: one reads a tree out of the rollout's arguments or the imperative
objects, the other builds the arguments, or writes the objects, from a tree.

* Controller parameters: `controller_params` and `with_controller_params` map a controller's parameter matrix to
  ``{stage: {name: array}}`` and back, and `set_controller_params` writes a tree into a
  `multirotor.controller.Controller`. The names are those of the stage classes
  (``k_p``, ``max_velocity``, ...), and each name's row in the matrix comes from the
  class, so the two cannot disagree.
* Vehicle parameters: `vehicle_params` and `vehicle_tuple`.
* State: `get_state`, `rollout_state`, `advance_state` and `set_state` take the state of
  a vehicle and its controller out of the objects, into a rollout, and back, so that a
  simulation can alternate between the imperative classes and the rollout.

Everything accepts and returns numpy arrays or JAX arrays alike; JAX is not imported.
"""
from typing import Dict, Tuple

import numpy as np

from . import physics
from .controller import data as controller_data
from .controller.pid import _Row, AltController, AltRateController, AttController
from .controller.pid import PosController, RateController, VelController
from .rollout import VEHICLE_FIELDS, vehicle_arguments



def _is_jax(value) -> bool:
    return type(value).__module__.split('.')[0] in ('jax', 'jaxlib')



def _any_jax(tree) -> bool:
    if isinstance(tree, dict):
        return any(_any_jax(v) for v in tree.values())
    if isinstance(tree, (tuple, list)):
        return any(_any_jax(v) for v in tree)
    return _is_jax(tree)



def _stack(rows):
    "A matrix of the rows, as a JAX array if any of them is one."
    if any(_is_jax(r) for r in rows):
        import jax.numpy as jnp
        return jnp.stack(rows)
    return np.stack([np.asarray(r, float) for r in rows])



# ---------------------------------------------------------------------------
# Controller parameters

_STAGES = {
    'ctrl_p': (PosController, controller_data.CASCADE_POSITION, controller_data.CASCADE_VELOCITY),
    'ctrl_v': (VelController, controller_data.CASCADE_VELOCITY, controller_data.CASCADE_ATTITUDE),
    'ctrl_a': (AttController, controller_data.CASCADE_ATTITUDE, controller_data.CASCADE_RATE),
    'ctrl_r': (RateController, controller_data.CASCADE_RATE, controller_data.CASCADE_ALTITUDE),
    'ctrl_z': (AltController, controller_data.CASCADE_ALTITUDE, controller_data.CASCADE_CLIMB),
    'ctrl_vz': (
        AltRateController, controller_data.CASCADE_CLIMB, controller_data.N_CASCADE_CHANNELS),
}
"Per stage of a `Controller`: its class, and the columns of the matrices that it owns."



def _rows(stage_class) -> Dict[str, int]:
    "The matrix row of each parameter of a stage class that is stored in the matrix."
    found = {}
    for name in dir(stage_class):
        attribute = getattr(stage_class, name)
        if isinstance(attribute, _Row):
            found[name] = attribute.row
    return found



def controller_params(params: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """
    The named gains and limits (the parameters) of a controller's parameter matrix.

    Parameters
    ----------
    params : np.ndarray
        The matrix of a `ControllerData` with six stages, shape (N_PID_PARAMS, 12),
        for example ``controller._data.params`` or ``rollout.controller_arguments(...)[0]``.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        ``{stage: {name: array}}``, with a value for each channel of the stage, shape
        (n,). The stages are ``ctrl_p``, ``ctrl_v``, ``ctrl_a``, ``ctrl_r``, ``ctrl_z``
        and ``ctrl_vz``, and the names are those of `Controller.get_params`, such as
        ``k_p`` and ``max_velocity``, that are stored in the matrix. The values are
        views of ``params`` for numpy.
    """
    return {
        stage: {name: params[row, start:stop] for name, row in _rows(cls).items()}
        for stage, (cls, start, stop) in _STAGES.items()}



def with_controller_params(params, tree: Dict[str, Dict[str, np.ndarray]]):
    """
    A copy of a parameter matrix with the parameters of ``tree`` in place of its own.
    Differentiable: the copy is made with JAX if any value in the tree or the matrix
    is a JAX array.

    Parameters
    ----------
    params
        The matrix, as in `controller_params`.
    tree : Dict[str, Dict[str, np.ndarray]]
        ``{stage: {name: value}}``, as `controller_params` returns. Only the entries present are
        replaced. A value is one number for all channels of the stage, or one per channel.

    Returns
    -------
    The new matrix. ``params`` is not changed.

    Raises
    ------
    KeyError
        For a stage or a name that does not exist.
    """
    use_jax = _any_jax(tree) or _is_jax(params)
    if use_jax:
        import jax.numpy as jnp
        new = jnp.asarray(params)
    else:
        new = np.array(params, float)
    for stage, values in tree.items():
        cls, start, stop = _STAGES[stage]
        rows = _rows(cls)
        for name, value in values.items():
            row = rows[name]
            if use_jax:
                new = new.at[row, start:stop].set(jnp.broadcast_to(jnp.asarray(value), (stop - start,)))
            else:
                new[row, start:stop] = value
    return new



def set_controller_params(controller, tree: Dict[str, Dict[str, np.ndarray]]) -> None:
    """
    Write controller parameters into a `multirotor.controller.Controller`, as `Controller.set_params`
    does. Use it to apply the result of an optimization to the controller that the
    imperative classes run.
    """
    controller.set_params(**{
        stage: {name: np.array(value, float) for name, value in values.items()}
        for stage, values in tree.items()})



# ---------------------------------------------------------------------------
# Vehicle parameters

def _names(prefix: str) -> Dict[str, int]:
    "Rows of a parameter matrix of `multirotor.physics`, by lower-case name."
    return {
        name[len(prefix):].lower(): value for name, value in vars(physics).items()
        if name.startswith(prefix) and isinstance(value, int)}



_PROPELLER, _MOTOR, _BATTERY = _names('PROP_'), _names('MOTOR_'), _names('BATTERY_')
_FIELDS_OF_KIND = {'body': 7, 'motors': 10, 'motors_battery': 13}



def vehicle_params(vehicle: Tuple, kind: str) -> dict:
    """
    The vehicle's parameters, by name.

    Parameters
    ----------
    vehicle : tuple
        The ``vehicle`` argument of a ``closed_loop_<kind>`` function, from
        `multirotor.rollout.vehicle_arguments`.
    kind : str
        ``'body'``, ``'motors'`` or ``'motors_battery'``.

    Returns
    -------
    dict
        ``g``, ``mass``, ``inertia``, ``clockwise`` and ``positions``,
        and ``propeller``: ``{name: array (n,)}`` for each row of the propeller matrix
        (``k_thrust``, ``k_drag``, ...). With motors: ``motor`` likewise (``resistance``,
        ...), ``max_voltage`` and ``disturbance``. With a battery: ``battery``
        (``capacity``, ``internal_resistance``, ``min_voltage``), ``ocv_soc`` and
        ``ocv_voltage``.

    Notes
    -----
    The inertia matrix's inverse is not in the tree: `vehicle_tuple` computes it, so it
    cannot disagree with ``inertia``. The control allocation matrix is not part of the
    vehicle's parameters here: it is in the ``ctrl`` argument of a rollout, and keeps the
    model that the controller was built with when ``positions`` or ``k_thrust`` change.
    """
    values = dict(zip(VEHICLE_FIELDS, vehicle))
    tree = {key: values[key] for key in ('g', 'mass', 'inertia', 'clockwise', 'positions')}
    tree['propeller'] = {name: values['prop_params'][row] for name, row in _PROPELLER.items()
                         if row < len(values['prop_params'])}
    if kind != 'body':
        tree['motor'] = {name: values['motor_params'][row] for name, row in _MOTOR.items()
                         if row < len(values['motor_params'])}
        tree['max_voltage'], tree['disturbance'] = values['max_voltage'], values['disturbance']
    if kind == 'motors_battery':
        tree['battery'] = {name: values['battery_params'][row] for name, row in _BATTERY.items()}
        tree['ocv_soc'], tree['ocv_voltage'] = values['ocv_soc'], values['ocv_voltage']
    return tree



def _matrix(tree: Dict[str, np.ndarray], names: Dict[str, int]):
    "The parameter matrix whose rows are the entries of ``tree``, in the order of the rows."
    return _stack([tree[name] for name, _ in sorted(names.items(), key=lambda item: item[1])])



def vehicle_tuple(tree: dict, kind: str) -> Tuple:
    """
    The ``vehicle`` argument of a ``closed_loop_<kind>`` function, from a tree as
    `vehicle_params` returns it. Differentiable with respect to the tree's values.
    """
    values = {key: tree[key] for key in ('g', 'mass', 'inertia', 'clockwise', 'positions')}
    if _any_jax(tree['inertia']):
        import jax.numpy as jnp
        values['inertia_inverse'] = jnp.linalg.inv(tree['inertia'])
    else:
        values['inertia_inverse'] = np.linalg.inv(np.asarray(tree['inertia'], float))
    values['prop_params'] = _matrix(tree['propeller'], _PROPELLER)
    if kind != 'body':
        values['motor_params'] = _matrix(tree['motor'], _MOTOR)
        values['max_voltage'], values['disturbance'] = tree['max_voltage'], tree['disturbance']
    if kind == 'motors_battery':
        values['battery_params'] = _stack([tree['battery'][name] for name, _ in sorted(
            _BATTERY.items(), key=lambda item: item[1])])
        values['ocv_soc'], values['ocv_voltage'] = tree['ocv_soc'], tree['ocv_voltage']
    return tuple(values[field] for field in VEHICLE_FIELDS[:_FIELDS_OF_KIND[kind]])



# ---------------------------------------------------------------------------
# State

def get_state(vehicle, controller) -> dict:
    """
    The state of a vehicle and its controller, as a tree of float64 copies.

    Returns
    -------
    dict
        ``vehicle``: ``t`` (time), ``x`` (the 12 body states, then the motor speeds),
        ``aux`` (``charge drawn, cut off, overloaded`` of the battery) and
        ``acceleration`` (body frame, at the start of the last step).
        ``controller``: ``n`` (steps taken), ``x`` (the state matrix) and the memo of the
        stages that run at their own rate: ``ref_vel``, ``thrust`` and ``torques``.
    """
    _, (x, aux, acceleration), _ = vehicle_arguments(vehicle)
    return {
        'vehicle': {'t': float(vehicle.t), 'x': x, 'aux': aux, 'acceleration': acceleration},
        'controller': {
            'n': int(controller.n), 'x': np.array(controller._data.state, float),
            'ref_vel': np.array(controller._ref_vel, float),
            'thrust': np.array(controller._thrust, float),
            'torques': np.array(controller._torques, float)}}



def rollout_state(state: dict) -> Tuple[Tuple, Tuple]:
    """
    The ``state0`` and ``start`` arguments of a ``closed_loop_<kind>`` function, from a
    state tree.
    """
    v, c = state['vehicle'], state['controller']
    return ((v['x'], v['aux'], v['acceleration']),
            (c['x'], (c['ref_vel'], c['thrust'], c['torques']), c['n']))



def advance_state(state: dict, carry: Tuple, steps: int, dt: float) -> dict:
    """
    The state tree after a rollout of ``steps`` steps.

    Parameters
    ----------
    state : dict
        The state that the rollout started from.
    carry : tuple
        The first result of the ``closed_loop_<kind>`` function.
    steps : int
        The number of steps of the rollout, ``len(targets)``.
    dt : float
        The simulation time step.
    """
    controller_x, memo, (x, aux, acceleration) = carry
    return {
        'vehicle': {
            't': state['vehicle']['t'] + steps * dt, 'x': x, 'aux': aux,
            'acceleration': acceleration},
        'controller': {
            'n': state['controller']['n'] + steps, 'x': controller_x, 'ref_vel': memo[0],
            'thrust': memo[1], 'torques': memo[2]}}



def set_state(vehicle, controller, state: dict) -> None:
    """
    Write a state tree into a vehicle and its controller, which then continue from it.

    The motors' voltages and currents and the battery's voltage and current are outputs
    of a step, not state. They are not in the tree, and are updated by the next step.

    Raises
    ------
    ValueError
        If the state is not that of this kind of vehicle.
    """
    v, c = state['vehicle'], state['controller']
    kind, (x0, _, _), _ = vehicle_arguments(vehicle)
    x = np.asarray(v['x'], float)
    if x.shape != x0.shape:
        raise ValueError(f'The vehicle state has shape {x.shape}; expected {x0.shape}.')
    n_body = vehicle.state.shape[0]
    vehicle.state = x[:n_body].astype(vehicle.dtype)
    vehicle._dxdt[3:6] = np.asarray(v['acceleration'], float)
    vehicle.t = float(v['t'])
    if kind != 'body':
        vehicle._data.motors.speed[:] = x[n_body:]
    if kind == 'motors_battery':
        battery = vehicle.battery
        battery.charge_drawn = float(v['aux'][0])
        battery.cutoff, battery.overload = bool(v['aux'][1]), bool(v['aux'][2])
        battery.current = 0.
        battery.voltage = 0. if battery.empty else battery.open_circuit_voltage()
    controller._data.state[:] = np.asarray(c['x'], float)
    controller._ref_vel = np.array(c['ref_vel'], float)
    controller._thrust = np.array(c['thrust'], float)
    controller._torques = np.array(c['torques'], float)
    controller.n = int(c['n'])
    controller.t = vehicle.t
    controller.action = np.concatenate((controller._thrust, controller._torques)).astype(controller.dtype)
