"""
Functional tests for the modules that depend on the simulation: environments,
controller, data logging and the control-library system. Each runs once with the
current vehicle and once with the frozen oracle, and the results must match.
"""
import numpy as np
import pytest

from multirotor.env import DynamicsMultirotorEnv, SpeedsMultirotorEnv
from multirotor.helpers import DataLog
from multirotor.optimize import get_controller
from multirotor.simulation import Multirotor

from common import atol_for, hover_signals, make_pair, make_params

STEPS = 100



def test_speeds_env_matches_oracle():
    vp, sp = make_params(8, 'quadratic', False)
    new, old = make_pair(8, 'quadratic', False)
    envs = [SpeedsMultirotorEnv(new), SpeedsMultirotorEnv(old)]
    for env in envs:
        env.seed(0)
        env.reset()
    u = hover_signals(vp, sp) * 1.01
    for i in range(STEPS):
        (s_new, r_new, d_new, _), (s_old, r_old, d_old, _) = [env.step(u) for env in envs]
        np.testing.assert_allclose(s_new, s_old, rtol=0, atol=atol_for(sp.dtype), err_msg=f'step {i}')
        assert r_new == pytest.approx(r_old, abs=1e-6)
        assert d_new == d_old



def test_dynamics_env_with_allocation_matches_oracle():
    new, old = make_pair(8, 'quadratic', False)
    envs = [DynamicsMultirotorEnv(new, allocate=True), DynamicsMultirotorEnv(old, allocate=True)]
    for env in envs:
        env.seed(0)
        env.reset()
    action = np.asarray([0, 0, new.weight * 1.05, 0.01, -0.01, 0.], new.dtype)
    for i in range(STEPS):
        (s_new, *_), (s_old, *_) = [env.step(action.copy()) for env in envs]
        np.testing.assert_allclose(s_new, s_old, rtol=0, atol=atol_for(new.dtype), err_msg=f'step {i}')



def test_closed_loop_matches_oracle():
    new, old = make_pair(8, 'quadratic', False)
    ref = np.asarray([1., 0.5, 2., 0.], new.dtype)
    controllers = [get_controller(new), get_controller(old)]
    for i in range(STEPS):
        for vehicle, ctrl in zip((new, old), controllers):
            action = ctrl.step(reference=ref)
            vehicle.step_speeds(vehicle.allocate_control(action[0], action[1:4]))
        np.testing.assert_allclose(new.state, old.state, rtol=0, atol=atol_for(new.dtype), err_msg=f'step {i}')



def test_datalog_records_distinct_states():
    vp, sp = make_params(8, 'quadratic', False)
    vehicle = Multirotor(vp, sp)
    log = DataLog(vehicle=vehicle)
    u = hover_signals(vp, sp) * 1.05
    for _ in range(5):
        vehicle.step_speeds(u)
        log.log()
    log.done_logging()
    assert len(np.unique(log.z)) == 5, 'each logged state must be a separate snapshot'



@pytest.mark.parametrize('motor', (False, True))
def test_controls_system_builds(motor):
    pytest.importorskip('control')
    vp, sp = make_params(8, 'quadratic', motor, dtype=np.float64)
    vehicle = Multirotor(vp, sp)
    sys = vehicle.nonlinear_speeds_controls_system()
    assert sys.nstates == 12
