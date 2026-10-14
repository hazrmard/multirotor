"""Benchmark case registry.

A case is a named benchmark: a ``Runner`` class plus a list of parameter sets. A
``Runner`` exposes:

* ``reset()``: restore the initial state (not timed).
* ``run(n)``: advance the simulation by ``n`` steps (timed).
* ``dt``: simulated seconds per step, used for the real-time factor.
* ``scope_points()``: the callables a scope pass times (see ``scopes.ScopePoint``).

``CASES`` maps a case name to ``(Runner class, list of parameter sets)``.
``QUICK_STEPS`` overrides the step counts for ``--quick`` smoke tests.
"""
from copy import deepcopy
from dataclasses import replace
from typing import Dict, List

import numpy as np

from multirotor.env import DynamicsMultirotorEnv
from multirotor.helpers import DataLog, find_nominal_speed, vehicle_params_factory
from multirotor.models.tarott18 import MP, PP, SP
from multirotor.optimize import get_controller, run_sim
from multirotor.simulation import Multirotor
from multirotor.vehicle import MotorParams, PropellerParams

from scopes import Level, ScopePoint

M_PROP = 0.5   # kg per propeller
M_BODY = 4.0   # kg
D_PROP = 0.635  # m, arm length (same as the tarott18 octorotor)
ALTITUDE = 5.   # m, hold point for closed-loop cases


def make_vehicle(
    n: int = 8, thrust_model: str = 'quadratic', motor: bool = False
) -> Multirotor:
    """Build a ``Multirotor`` with ``n`` identical rotors.

    Parameters
    ----------
    n : int
        Number of rotors.
    thrust_model : {'quadratic', 'fsolve'}
        How each propeller computes thrust from its speed.

        * ``'quadratic'``: thrust = ``k_thrust * speed**2``
          (``PropellerParams.use_thrust_constant=True``). Constants come from
          ``models.tarott18``.
        * ``'fsolve'``: blade-element model (``use_thrust_constant=False``). For
          every propeller and every model evaluation, ``physics.thrust`` solves
          for the induced velocity with ``scipy.optimize.fsolve``. Uses an 18 in
          diameter, 6 in pitch propeller.
    motor : bool
        Attach the ``tarott18`` motor model to each propeller.
    """
    if thrust_model == 'quadratic':
        prop = deepcopy(PP)
    elif thrust_model == 'fsolve':
        prop = PropellerParams(
            moment_of_inertia=PP.moment_of_inertia, use_thrust_constant=False,
            diameter=18., pitch=6., k_drag=PP.k_drag, k_thrust=PP.k_thrust)
    else:
        raise ValueError(
            f"unknown thrust_model {thrust_model!r}; expected 'quadratic' or 'fsolve'")
    if motor:
        prop = replace(prop, motor=deepcopy(MP))
    vp = vehicle_params_factory(n, M_PROP, D_PROP, prop, M_BODY)
    return Multirotor(vp, SP)


def nominal_speed(vehicle: Multirotor) -> float:
    """Propeller speed (rad/s) at which one rotor carries its share of the weight.

    Solved with ``helpers.find_nominal_speed`` on the propeller's own thrust
    function, so it is correct for both thrust models.
    """
    p = vehicle.params
    weight_per_rotor = p.mass * vehicle.simulation.g / p.nprops
    thrust = vehicle.propellers[0].thrust
    return float(find_nominal_speed(lambda w: thrust(float(np.ravel(w)[0])),
                                    weight_per_rotor))


def motor_signal(motor: MotorParams, speed: float) -> float:
    """Speed signal that holds a motor at ``speed`` (rad/s) in steady state.

    At steady state the electrical torque balances the load:
    ``k_torque * i = k_drag * speed**2 + k_df * speed``, and the supply voltage is
    ``V = k_emf * speed + resistance * i``. The motor converts the signal ``u`` to
    voltage with ``V = speed_voltage_scaling * u``.
    """
    current = (motor.k_drag * speed**2 + motor.k_df * speed) / motor.k_torque
    voltage = motor.k_emf * speed + motor.resistance * current
    return voltage / motor.speed_voltage_scaling


def hover_speeds(vehicle: Multirotor) -> np.ndarray:
    """Per-rotor speed signals that hold the vehicle at hover.

    Equals ``nominal_speed`` without a motor model. With a motor model it is the
    signal that settles the motor at ``nominal_speed``.
    """
    w = nominal_speed(vehicle)
    motor = vehicle.params.propellers[0].motor
    u = w if motor is None else motor_signal(motor, w)
    return np.full(vehicle.params.nprops, u, vehicle.dtype)


def settle_at_hover(vehicle: Multirotor) -> None:
    """Start the propellers (and motors) at the hover speed instead of at rest."""
    w = nominal_speed(vehicle)
    for p in vehicle.propellers:
        p.speed = w
        if p.motor is not None:
            p.motor.speed = w


N_SCHEDULE = 64


def excitation(base: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Deterministic schedule of inputs, ``base * (1 +- frac)``, cycled per step.

    A vehicle fed exactly its hover inputs sits at equilibrium, where ``odeint``
    takes the fewest model evaluations. Perturbed inputs make it work for its
    answer.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-frac, frac, size=(N_SCHEDULE, *np.shape(base)))
    return (np.asarray(base) * (1 + noise)).astype(np.asarray(base).dtype)


def vehicle_points(v: Multirotor) -> List[ScopePoint]:
    """Scope points for a vehicle: its step methods, integrator, model and propellers.

    ``dxdt_speeds`` and ``dxdt_dynamics`` both become the scope ``model``; only the
    one the case calls records anything.
    """
    pts = [ScopePoint(Level.COARSE, v, 'step_speeds', 'step_speeds'),
           ScopePoint(Level.COARSE, v, 'step_dynamics', 'step_dynamics'),
           ScopePoint(Level.FINE, v, 'integrator', 'integrator'),
           ScopePoint(Level.FINE, v, 'dxdt_speeds', 'model'),
           ScopePoint(Level.FINE, v, 'dxdt_dynamics', 'model'),
           ScopePoint(Level.FINE, v, 'get_forces_torques', 'forces_torques')]
    pts += [ScopePoint(Level.FINE, p, 'step', 'propeller') for p in v.propellers]
    return pts


def log_points() -> List[ScopePoint]:
    # DataLog is created inside reset()/run_sim, so patch the class.
    return [ScopePoint(Level.COARSE, DataLog, 'log', 'log'),
            ScopePoint(Level.COARSE, DataLog, 'done_logging', 'log_done')]


class Runner:
    dt: float = SP.dt

    def reset(self) -> None:
        raise NotImplementedError

    def run(self, n: int) -> None:
        raise NotImplementedError

    def scope_points(self) -> List[ScopePoint]:
        """Callables that a scope pass times as scopes.

        Each ``ScopePoint(level, owner, attr, name)`` names a callable attribute
        ``attr`` of ``owner`` (an instance or a class) and the scope ``name`` it
        records. ``owner.attr`` must exist. A missing attribute is skipped and
        reported under ``skipped`` in the scope tree, so check ``skipped`` after
        adding a point.

        ``level`` is ``Level.COARSE`` or ``Level.FINE``. A pass at ``FINE`` wraps
        both kinds.

        Example::

            def scope_points(self):
                return [ScopePoint(Level.COARSE, self.c, 'step', 'controller')]
        """
        return []


class StepSpeeds(Runner):
    """Raw ODE step. Isolates ``Multirotor.step_speeds``.

    ``excite=False`` holds exact hover speeds (equilibrium: cheapest possible
    integration). ``excite=True`` cycles through +-5% perturbed speeds.
    """
    def __init__(self, n=8, thrust_model='quadratic', motor=False, excite=True):
        self.v = make_vehicle(n, thrust_model, motor)
        hover = hover_speeds(self.v)
        self.us = excitation(hover, 0.05) if excite else hover[None]
        self.dt = self.v.simulation.dt

    def reset(self):
        self.v.reset()
        settle_at_hover(self.v)

    def run(self, n):
        step, us = self.v.step_speeds, self.us
        m = len(us)
        for i in range(n):
            step(us[i % m])

    def scope_points(self):
        return vehicle_points(self.v)


class StepDynamics(Runner):
    """Raw ODE step with force/torque inputs. Isolates ``step_dynamics``.

    ``excite=False`` holds hover (z-force = weight, other inputs zero).
    ``excite=True`` cycles through z-force within +-5% of the weight and torques
    within +-0.05 N m.
    """
    def __init__(self, n=8, excite=True):
        self.v = make_vehicle(n)
        hover = np.asarray([0, 0, self.v.params.mass * self.v.simulation.g, 0, 0, 0],
                           self.v.dtype)
        self.us = np.tile(hover, (N_SCHEDULE, 1)) if excite else hover[None]
        if excite:
            rng = np.random.default_rng(0)
            self.us[:, 2] *= 1 + rng.uniform(-0.05, 0.05, N_SCHEDULE)
            self.us[:, 3:] = rng.uniform(-0.05, 0.05, (N_SCHEDULE, 3))
        self.dt = self.v.simulation.dt

    def reset(self):
        self.v.reset()

    def run(self, n):
        step, us = self.v.step_dynamics, self.us
        m = len(us)
        for i in range(n):
            step(us[i % m])

    def scope_points(self):
        return vehicle_points(self.v)


class ControllerStep(Runner):
    """``Controller.step`` alone; the vehicle state is held fixed."""
    def __init__(self, n=8):
        self.v = make_vehicle(n)
        self.c = get_controller(self.v)
        self.ref = np.asarray([0, 0, ALTITUDE, 0], self.v.dtype)
        self.dt = self.v.simulation.dt

    def reset(self):
        self.v.reset()
        self.c.reset()

    def run(self, n):
        step, ref = self.c.step, self.ref
        for _ in range(n):
            step(reference=ref)

    def scope_points(self):
        return [ScopePoint(Level.COARSE, self.c, 'step', 'controller')]


class ClosedLoop(Runner):
    """controller -> allocation -> ``step_speeds``; optionally logging to ``DataLog``."""
    def __init__(self, n=8, thrust_model='quadratic', motor=False, log=False):
        self.v = make_vehicle(n, thrust_model, motor)
        self.c = get_controller(self.v)
        self.ref = np.asarray([0, 0, ALTITUDE, 0], self.v.dtype)
        self.log = log
        self.dt = self.v.simulation.dt
        self.datalog = None

    def reset(self):
        self.v.reset()
        self.c.reset()
        self.datalog = DataLog(self.v, self.c) if self.log else None

    def run(self, n):
        v, c, ref, dl = self.v, self.c, self.ref, self.datalog
        for _ in range(n):
            a = c.step(reference=ref)
            v.step_speeds(v.allocate_control(a[0], a[1:4]))
            if dl is not None:
                dl.log()
        if dl is not None:
            dl.done_logging()

    def scope_points(self):
        return ([ScopePoint(Level.COARSE, self.c, 'step', 'controller'),
                 ScopePoint(Level.COARSE, self.v, 'allocate_control', 'allocation')]
                + vehicle_points(self.v) + log_points())


class RunSim(Runner):
    """``optimize.run_sim`` through the gym-style env, over a bounded trajectory."""
    def __init__(self, n=8):
        self.v = make_vehicle(n)
        self.env = DynamicsMultirotorEnv(self.v, allocate=True)
        self.c = get_controller(self.v)
        self.dt = self.v.simulation.dt

    def reset(self):
        self.env.reset()
        self.c.reset()

    def run(self, n):
        point = np.asarray([0, 0, ALTITUDE], self.v.dtype)
        traj = ((point, None) for _ in range(n))
        log = run_sim(self.env, traj, self.c)
        # run_sim stops early when the env reports done; us/step would then be
        # divided by steps that never ran.
        if len(log) != n:
            raise RuntimeError(f'run_sim ended after {len(log)} of {n} steps')

    def scope_points(self):
        return ([ScopePoint(Level.COARSE, self.c, 'step', 'controller'),
                 ScopePoint(Level.COARSE, self.env, 'step', 'env.step'),
                 ScopePoint(Level.COARSE, self.env, 'reward', 'reward'),
                 ScopePoint(Level.COARSE, self.v, 'allocate_control', 'allocation')]
                + vehicle_points(self.v) + log_points())


# case name -> (Runner class, list of parameter sets)
CASES: Dict[str, tuple] = {
    'step_speeds': (StepSpeeds, [
        dict(n=n, thrust_model='quadratic', motor=False, excite=True, steps=1000)
        for n in (4, 6, 8)
    ] + [
        dict(n=8, thrust_model='quadratic', motor=True, excite=True, steps=1000),
        dict(n=8, thrust_model='fsolve', motor=False, excite=True, steps=300),
        # equilibrium reference: the integrator's best case
        dict(n=8, thrust_model='quadratic', motor=False, excite=False, steps=1000),
    ]),
    'step_dynamics': (StepDynamics, [
        dict(n=8, excite=True, steps=1000), dict(n=8, excite=False, steps=1000),
    ]),
    'controller_step': (ControllerStep, [dict(n=8, steps=5000)]),
    'closed_loop': (ClosedLoop, [
        dict(n=8, thrust_model='quadratic', motor=False, log=False, steps=2000),
        dict(n=8, thrust_model='quadratic', motor=False, log=True, steps=2000),
    ]),
    'run_sim': (RunSim, [dict(n=8, steps=1000)]),
    # memory growth of DataLog: bytes/step slope across these
    'log_scaling': (ClosedLoop, [
        dict(n=8, log=True, steps=s) for s in (1000, 5000, 20000)
    ]),
}

# --quick: fewer steps so a smoke test takes seconds
QUICK_STEPS = {'step_speeds': 100, 'step_dynamics': 100, 'controller_step': 500,
               'closed_loop': 200, 'run_sim': 100}
QUICK_SKIP = {'log_scaling'}


def case_id(name: str, params: Dict) -> str:
    shown = ','.join(f'{k}={v}' for k, v in params.items() if k != 'steps')
    # log_scaling is identified by its step count, since steps is the swept axis
    if name == 'log_scaling':
        shown += f',steps={params["steps"]}'
    return f'{name}[{shown}]'


def expand(names=None, quick=False) -> List[tuple]:
    """Return ``[(id, name, params), ...]`` for the selected cases."""
    out = []
    for name, (_, grid) in CASES.items():
        if names and name not in names:
            continue
        if quick and name in QUICK_SKIP:
            continue
        for params in grid:
            params = dict(params)
            if quick:
                params['steps'] = QUICK_STEPS.get(name, params['steps'])
            out.append((case_id(name, params), name, params))
    return out


def build(name: str, params: Dict) -> Runner:
    builder = CASES[name][0]
    kwargs = {k: v for k, v in params.items() if k != 'steps'}
    return builder(**kwargs)
