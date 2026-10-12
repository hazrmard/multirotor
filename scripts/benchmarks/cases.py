"""Benchmark case registry.

A case is a builder ``fn(**params) -> Runner``. A ``Runner`` exposes:

* ``reset()``: restore the initial state (not timed).
* ``run(n)``: advance the simulation by ``n`` steps (timed).
* ``dt``: simulated seconds per step, used for the real-time factor.

``CASES`` maps a case name to its parameter grid. ``QUICK`` overrides the
grid with smaller step counts for smoke tests.
"""
from copy import deepcopy
from itertools import product
from typing import Dict, List

import numpy as np

from multirotor.env import DynamicsMultirotorEnv
from multirotor.helpers import DataLog, vehicle_params_factory
from multirotor.models.tarott18 import MP, PP, SP
from multirotor.optimize import get_controller, run_sim
from multirotor.simulation import Multirotor
from multirotor.vehicle import PropellerParams

M_PROP = 0.5   # kg per propeller
M_BODY = 4.0   # kg
D_PROP = 0.635  # m, arm length (same as the tarott18 octorotor)
ALTITUDE = 5.   # m, hold point for closed-loop cases


def make_vehicle(n: int = 8, thrust: str = 'const', motor: bool = False) -> Multirotor:
    """
    Parameters
    ----------
    n : number of rotors.
    thrust : ``'const'`` uses ``k_thrust * w^2``; ``'blade'`` uses the blade
        element model (``fsolve`` per rotor per right-hand-side call).
    motor : attach the motor model to each propeller.
    """
    if thrust == 'const':
        prop = deepcopy(PP)
    elif thrust == 'blade':
        prop = PropellerParams(
            moment_of_inertia=PP.moment_of_inertia, use_thrust_constant=False,
            diameter=18., pitch=6., k_drag=PP.k_drag, k_thrust=PP.k_thrust)
    else:
        raise ValueError(f'unknown thrust model {thrust!r}')
    if motor:
        prop.motor = deepcopy(MP)
        prop.__post_init__()
    vp = vehicle_params_factory(n, M_PROP, D_PROP, prop, M_BODY)
    return Multirotor(vp, SP)


def hover_speeds(vehicle: Multirotor) -> np.ndarray:
    """Per-rotor speed signal that roughly balances weight (constant-thrust model)."""
    p = vehicle.params
    w = np.sqrt(p.mass * vehicle.simulation.g / (p.nprops * PP.k_thrust))
    return np.full(p.nprops, w, vehicle.dtype)


N_SCHEDULE = 64


def excitation(base: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Deterministic schedule of inputs, ``base * (1 +- frac)``, cycled per step.

    A vehicle fed exactly its hover inputs sits at equilibrium, where ``odeint``
    takes the fewest right-hand-side calls. Perturbed inputs make it work for
    its answer.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-frac, frac, size=(N_SCHEDULE, *np.shape(base)))
    return (np.asarray(base) * (1 + noise)).astype(np.asarray(base).dtype)


def vehicle_points(v: Multirotor) -> List[tuple]:
    """Scope points for the vehicle: ``(level, owner, attribute, scope name)``."""
    pts = [('coarse', v, 'step_speeds', 'step_speeds'),
           ('coarse', v, 'step_dynamics', 'step_dynamics'),
           ('fine', v, 'integrator', 'integrator'),
           ('fine', v, 'dxdt_speeds', 'model'),
           ('fine', v, 'dxdt_dynamics', 'model'),
           ('fine', v, 'get_forces_torques', 'forces_torques')]
    pts += [('fine', p, 'step', 'propeller') for p in v.propellers]
    return pts


def log_points() -> List[tuple]:
    # DataLog is created inside reset()/run_sim, so patch the class.
    return [('coarse', DataLog, 'log', 'log'),
            ('coarse', DataLog, 'done_logging', 'log_done')]


class Runner:
    dt: float = SP.dt

    def reset(self) -> None:
        raise NotImplementedError

    def run(self, n: int) -> None:
        raise NotImplementedError

    def scope_points(self) -> List[tuple]:
        """Callables to time as scopes: ``(level, owner, attr, name)``.

        Levels: ``coarse`` (controller, env, step, logging) and ``fine``
        (integrator, model, propellers; includes coarse).
        """
        return []


class StepSpeeds(Runner):
    """Raw ODE step. Isolates ``Multirotor.step_speeds``.

    ``excite=False`` holds exact hover speeds (equilibrium: cheapest possible
    integration). ``excite=True`` cycles through +-5% perturbed speeds.
    """
    def __init__(self, n=8, thrust='const', motor=False, excite=True):
        self.v = make_vehicle(n, thrust, motor)
        hover = hover_speeds(self.v)
        self.us = excitation(hover, 0.05) if excite else hover[None]
        self.dt = self.v.simulation.dt

    def reset(self):
        self.v.reset()

    def run(self, n):
        step, us = self.v.step_speeds, self.us
        m = len(us)
        for i in range(n):
            step(us[i % m])

    def scope_points(self):
        return vehicle_points(self.v)


class StepDynamics(Runner):
    """Raw ODE step with force/torque inputs. Isolates ``step_dynamics``."""
    def __init__(self, n=8, excite=True):
        self.v = make_vehicle(n)
        hover = np.asarray([0, 0, self.v.params.mass * self.v.simulation.g, 0, 0, 0],
                           self.v.dtype)
        self.us = excitation(hover, 0.05) if excite else hover[None]
        # torques are zero at hover, so perturb them additively
        if excite:
            self.us[:, 3:] = excitation(np.ones(3, self.v.dtype), 0.5)[:, :] * 0.1 - 0.1
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
        return [('coarse', self.c, 'step', 'controller')]


class ClosedLoop(Runner):
    """controller -> allocation -> ``step_speeds``; optionally logging to ``DataLog``."""
    def __init__(self, n=8, thrust='const', motor=False, log=False):
        self.v = make_vehicle(n, thrust, motor)
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
        return ([('coarse', self.c, 'step', 'controller'),
                 ('coarse', self.v, 'allocate_control', 'allocation')]
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
        return ([('coarse', self.c, 'step', 'controller'),
                 ('coarse', self.env, 'step', 'env.step'),
                 ('coarse', self.env, 'reward', 'reward'),
                 ('coarse', self.v, 'allocate_control', 'allocation')]
                + vehicle_points(self.v) + log_points())


# name -> (builder, list of param dicts)
def _grid(**axes) -> List[Dict]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in product(*axes.values())]


CASES: Dict[str, tuple] = {
    'step_speeds': (StepSpeeds, [
        dict(n=n, thrust='const', motor=False, excite=True, steps=1000)
        for n in (4, 6, 8)
    ] + [
        dict(n=8, thrust='const', motor=True, excite=True, steps=1000),
        dict(n=8, thrust='blade', motor=False, excite=True, steps=300),
        # equilibrium reference: the integrator's best case
        dict(n=8, thrust='const', motor=False, excite=False, steps=1000),
    ]),
    'step_dynamics': (StepDynamics, [
        dict(n=8, excite=True, steps=1000), dict(n=8, excite=False, steps=1000),
    ]),
    'controller_step': (ControllerStep, [dict(n=8, steps=5000)]),
    'closed_loop': (ClosedLoop, [
        dict(n=8, thrust='const', motor=False, log=False, steps=2000),
        dict(n=8, thrust='const', motor=False, log=True, steps=2000),
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
