"""
`multirotor.rollout`: the closed loop of the cascaded controller and a vehicle
integrated with RK4, against the loop of `Controller` and `Multirotor` (LSODA).
Built for numba here; the JAX build is tested in `test_rollout_jax`.
"""
import numpy as np
import pytest

import multirotor.controller.pid as pid
import multirotor.rollout as rollout
from scipy.integrate import solve_ivp

from multirotor.simulation import Multirotor, odeint_integrator
from multirotor.vehicle import BatteryParams

from common import make_params
from test_controller import make_controller

BUILT = rollout.build_for('numba')

DT = .01
TARGET = np.array([3., -2., 5., .4])
# RK4 substeps per step, and the largest difference in position (m) from the LSODA
# loop. The motors' dynamics are fast, and need more substeps than the body alone.
SUBSTEPS = {'body': 4, 'motors': 16, 'motors_battery': 16}
POSITION_TOLERANCE = {'body': 1e-4, 'motors': 3e-3, 'motors_battery': 1e-3}



def dop853(fun, x0, dt, args=()):
    "A tight reference integrator, independent of LSODA."
    return solve_ivp(
        lambda t, x: fun(t, x, *args), (0, dt), x0, method='DOP853', rtol=1e-12,
        atol=1e-13).y[:, -1]



def make_vehicle(kind: str, integrator=odeint_integrator) -> Multirotor:
    battery = BatteryParams(
        max_voltage=25.2, capacity=20., internal_resistance=.08,
        ocv_soc=np.linspace(0, 1, 6), ocv_voltage=np.linspace(20., 25.2, 6)
    ) if kind == 'motors_battery' else None
    vp, sp = make_params(8, 'quadratic', kind != 'body', np.float64, DT, battery)
    return Multirotor(vp, sp, integrator=integrator)



def reference(kind: str, steps: int, integrator=odeint_integrator) -> np.ndarray:
    "State and controller output per step, from `Controller` and `Multirotor`."
    vehicle = make_vehicle(kind, integrator)
    controller = make_controller(pid, vehicle)
    out = []
    for _ in range(steps):
        action = controller.step(TARGET)
        vehicle.step_speeds(vehicle.allocate_control(action[0], action[1:]))
        out.append(np.concatenate((vehicle.state, action)))
    return np.array(out)



def arguments(kind: str):
    "The rollout's arguments from a fresh vehicle and controller."
    vehicle = make_vehicle(kind)
    found, state, plant = rollout.plant_arguments(vehicle)
    assert found == kind
    return state, plant, rollout.controller_arguments(make_controller(pid, vehicle), vehicle)



@pytest.mark.parametrize('kind', ('body', 'motors_battery'))
def test_numba_rollout_matches_the_loop(kind):
    steps = 60
    state, plant, ctrl = arguments(kind)
    built = BUILT
    _, trajectory = built.closed_loop(
        getattr(built, 'step_' + kind), state, plant, ctrl,
        np.tile(TARGET, (steps, 1)), DT, SUBSTEPS[kind])
    expected = reference(kind, steps)
    assert np.abs(trajectory - expected)[:, :3].max() < POSITION_TOLERANCE[kind]
    np.testing.assert_allclose(trajectory[:, 9:12], expected[:, 9:12], atol=1e-3)
