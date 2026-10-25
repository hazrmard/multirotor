"""
Parity tests: the current implementation against the frozen oracle in
``tests/legacy``, for the vehicle configurations whose numerics must not change:
vehicles without motors. Vehicles with motors integrate motors and body together
and are checked for accuracy in ``test_accuracy.py`` instead.
"""
import numpy as np
import pytest

from common import (
    EXCITE_FRAC, N_SCHEDULE, assert_snapshots_close, atol_for, hover_signals,
    make_pair, make_params, nominal_speed, schedule, settle_at_hover,
    snapshot
)

DTYPES = (np.float32, np.float64)

# (n, thrust_model, motor, steps). fsolve vehicles are slow, so they run fewer steps.
CONFIGS = [
    (4, 'quadratic', False, N_SCHEDULE),
    (6, 'quadratic', False, N_SCHEDULE),
    (8, 'quadratic', False, N_SCHEDULE),
    (8, 'fsolve', False, 16),
]



def run_parity(n, thrust_model, motor, steps, dtype, excite=True):
    vp, sp = make_params(n, thrust_model, motor, dtype)
    new, old = make_pair(n, thrust_model, motor, dtype)
    speed = nominal_speed(vp, sp)
    for v in (new, old):
        settle_at_hover(v, speed)
    hover = hover_signals(vp, sp)
    us = schedule(hover, EXCITE_FRAC[thrust_model], steps) if excite else np.tile(hover, (steps, 1))
    atol = atol_for(dtype)
    assert_snapshots_close(snapshot(new), snapshot(old), atol, 'initial')
    for i, u in enumerate(us):
        f_new, t_new = new.get_forces_torques(u, new.state)
        f_old, t_old = old.get_forces_torques(u, old.state)
        np.testing.assert_allclose(f_new, f_old, rtol=0, atol=atol, err_msg=f'step {i}: forces')
        np.testing.assert_allclose(t_new, t_old, rtol=0, atol=atol, err_msg=f'step {i}: torques')
        new.step_speeds(u)
        old.step_speeds(u)
        assert_snapshots_close(snapshot(new), snapshot(old), atol, f'step {i}')



@pytest.mark.parametrize('dtype', DTYPES, ids=lambda d: d.__name__)
@pytest.mark.parametrize('n, thrust_model, motor, steps', CONFIGS)
def test_step_speeds_parity(n, thrust_model, motor, steps, dtype):
    run_parity(n, thrust_model, motor, steps, dtype)



@pytest.mark.parametrize('dtype', DTYPES, ids=lambda d: d.__name__)
def test_step_speeds_parity_at_hover(dtype):
    run_parity(8, 'quadratic', False, N_SCHEDULE, dtype, excite=False)



@pytest.mark.parametrize('dtype', DTYPES, ids=lambda d: d.__name__)
def test_step_dynamics_parity(dtype):
    new, old = make_pair(8, 'quadratic', False, dtype)
    hover = np.asarray([0, 0, new.params.mass * new.simulation.g, 0, 0, 0], dtype)
    rng = np.random.default_rng(0)
    us = np.tile(hover, (N_SCHEDULE, 1))
    us[:, 2] *= 1 + rng.uniform(-0.05, 0.05, N_SCHEDULE)
    us[:, 3:] = rng.uniform(-0.05, 0.05, (N_SCHEDULE, 3))
    for i, u in enumerate(us):
        new.step_dynamics(u)
        old.step_dynamics(u)
        assert_snapshots_close(snapshot(new), snapshot(old), atol_for(dtype), f'step {i}')



@pytest.mark.parametrize('motor', (False, True))
def test_apply_speed_and_dxdt_are_pure(motor):
    vp, sp = make_params(8, 'quadratic', motor)
    new, _ = make_pair(8, 'quadratic', motor)
    settle_at_hover(new, nominal_speed(vp, sp))
    u = schedule(hover_signals(vp, sp), 0.05, 1)[0]
    before = snapshot(new)
    for _ in range(2):
        for s, p in zip(u, new.propellers):
            p.apply_speed(s)
        new.dxdt_speeds(0., new.state, u)
        new.get_forces_torques(u, new.state)
    assert_snapshots_close(snapshot(new), before, 0., 'after pure calls')



@pytest.mark.parametrize('excite', (True, False), ids=('excited', 'hover'))
@pytest.mark.parametrize('dtype', DTYPES, ids=lambda d: d.__name__)
@pytest.mark.parametrize('n', (4, 6, 8))
def test_fixed_forces_path_is_bit_identical(n, dtype, excite):
    # Vehicles without motors and with quadratic thrust calculate forces once
    # per step. Their trajectories must equal the oracle's exactly.
    vp, sp = make_params(n, 'quadratic', False, dtype)
    new, old = make_pair(n, 'quadratic', False, dtype)
    assert new._forces_fixed_over_step
    speed = nominal_speed(vp, sp)
    for v in (new, old):
        settle_at_hover(v, speed)
    hover = hover_signals(vp, sp)
    us = schedule(hover, 0.05, N_SCHEDULE) if excite else np.tile(hover, (N_SCHEDULE, 1))
    for i in range(200):
        u = us[i % N_SCHEDULE]
        np.testing.assert_array_equal(new.step_speeds(u), old.step_speeds(u), err_msg=f'step {i}')
        np.testing.assert_array_equal(new.acceleration, old.acceleration, err_msg=f'step {i}')
