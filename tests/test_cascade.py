"""
`multirotor.controller.cascade`, the cascaded controller as pure functions of its
state. The whole controller is tested against the legacy oracle in
``test_controller.py`` and ``test_fleet_controller.py``; this tests the functions
themselves: that they change none of their arguments, what a stage that does not run
repeats, and the stages' edge cases. The JAX build is tested in ``test_cascade_jax.py``.
"""
import copy
import itertools

import numpy as np
import pytest

import multirotor.controller.cascade as cascade
from multirotor.controller import data as K

OPTIONS = list(itertools.product(
    (False, True), (False, True), (False, True), (False, True),
    ((1, 1, 1), (3, 2, 5), (4, 1, 2))))
INERTIA = np.diag([.01, .012, .02])
MASS, WEIGHT, SIM_DT = 1.2, 11.77, .001
SCALE = np.array([2, 2, 2, 1, 1, 1, .3, .3, 1, .5, .5, .5])



def make_params(rng) -> np.ndarray:
    p = np.zeros((K.N_PID_PARAMS, K.N_CASCADE_CHANNELS))
    p[K.PID_K_P] = rng.uniform(.5, 5, 12)
    p[K.PID_K_I] = rng.uniform(0, .5, 12)
    p[K.PID_K_D] = rng.uniform(0, .5, 12)
    p[K.PID_MAX_ERR_I] = np.where(rng.random(12) < .5, np.inf, rng.uniform(.1, 2, 12))
    p[K.PID_LIMIT] = rng.uniform(.3, 3, 12)
    p[K.PID_MAX_ACCELERATION] = rng.uniform(.5, 4, 12)
    p[K.PID_MAX_ACCELERATION, 3:5] = 0.   # the branch without an acceleration limit
    return p



def inputs(rng):
    "A target, a body state, an acceleration and a feed-forward velocity."
    return (rng.normal(size=4) * 3, rng.normal(size=12) * SCALE, rng.normal(size=3),
            rng.normal(size=2))



def leaves(tree):
    "The arrays in nested tuples."
    if isinstance(tree, tuple):
        return [leaf for branch in tree for leaf in leaves(branch)]
    return [tree]



def zero_memo():
    return (np.zeros(2), np.zeros(1), np.zeros(3))



def step(params, x, memo, rng, n, steps=(1, 1, 1), leashing=True, sqrt_pos=True,
         sqrt_att=True, feed_forward=True, inputs_=None):
    reference, body_state, acceleration, velocity = inputs_ or inputs(rng)
    return cascade.cascaded_step(
        params, x, memo, reference, body_state, acceleration, INERTIA, MASS, WEIGHT,
        SIM_DT, n, *steps, velocity, feed_forward, .4, leashing, sqrt_pos, sqrt_att)



@pytest.mark.parametrize('leashing,sqrt_pos,sqrt_att,feed_forward,steps', OPTIONS)
def test_a_step_changes_none_of_its_arguments(leashing, sqrt_pos, sqrt_att, feed_forward, steps):
    "The caller keeps the returned state or drops it, so the step must not write to its own."
    rng = np.random.default_rng(0)
    params, x, memo = make_params(rng), rng.normal(size=(K.N_PID_STATES, 12)) * .1, zero_memo()
    for n in range(10):
        args = inputs(rng)
        before = copy.deepcopy((params, x, memo, args))
        new_x, new_memo, action = step(
            params, x, memo, rng, n, steps, leashing, sqrt_pos, sqrt_att, feed_forward, args)
        for given, kept in zip(leaves((params, x, memo, args)), leaves(before)):
            np.testing.assert_array_equal(given, kept)
        assert np.isfinite(new_x).all() and np.isfinite(action).all()
        x, memo = new_x, new_memo



def test_stages_that_do_not_run_repeat_their_outputs():
    "Between periods, the state is kept and the last thrust and torques are repeated."
    rng = np.random.default_rng(1)
    params, x, memo = make_params(rng), np.zeros((K.N_PID_STATES, 12)), zero_memo()
    x, memo, first = step(params, x, memo, rng, 0, steps=(3, 3, 3))
    for n in (1, 2):
        again_x, again_memo, action = step(params, x, memo, rng, n, steps=(3, 3, 3))
        np.testing.assert_array_equal(again_x, x)
        np.testing.assert_array_equal(action, first)
    changed_x, _, action = step(params, x, memo, rng, 3, steps=(3, 3, 3))
    assert not np.array_equal(changed_x, x)
    assert not np.array_equal(action, first)



def test_the_stages_run_at_their_own_periods():
    "With a long altitude period, only the thrust is kept between its steps."
    rng = np.random.default_rng(2)
    params, memo = make_params(rng), zero_memo()
    x, memo, first = step(params, np.zeros((K.N_PID_STATES, 12)), memo, rng, 0, steps=(1, 1, 4))
    _, _, second = step(params, x, memo, rng, 1, steps=(1, 1, 4))
    assert second[0] == first[0]                   # thrust repeated
    assert not np.array_equal(second[1:], first[1:])   # torques computed again



@pytest.mark.parametrize('err, expected', ((1., 2.), (-1., -2.), (0., 0.)))
def test_sqrt_control_without_p_gain_follows_the_acceleration_limit(err, expected):
    "With k_p=0 the correction is sqrt(2 * limit * |err|), signed. It was 0 for err > 0."
    assert cascade.sqrt_control(err, 0., 2., 0.) == pytest.approx(expected)



def test_sqrt_control_is_linear_near_zero_and_square_root_beyond():
    k_p, limit = 2., 1.
    linear_distance = limit / k_p**2
    assert cascade.sqrt_control(.1, k_p, limit, 0.) == pytest.approx(.1 * k_p)
    err = 3.
    assert cascade.sqrt_control(err, k_p, limit, 0.) == pytest.approx(
        np.sqrt(2. * limit * (err - linear_distance / 2.)))
    assert cascade.sqrt_control(-err, k_p, limit, 0.) == pytest.approx(
        -np.sqrt(2. * limit * (err - linear_distance / 2.)))



def test_sqrt_control_does_not_overshoot_within_a_step():
    "The correction rate is limited to the error divided by the time step."
    assert cascade.sqrt_control(1e-3, 0., 100., .1) == pytest.approx(1e-3 / .1)



def test_leashed_position_stage_does_not_use_the_leashed_error_as_the_previous_error():
    """
    With the leash and without the square-root controller, the leash used to store
    the leashed error as the previous error before the PID law read it, so the
    integral and derivative terms (and the output) depended on whether the state
    was stored. The derivative term must use the previous error that was stored.
    """
    rng = np.random.default_rng(0)
    params = np.zeros((K.N_PID_PARAMS, 2))
    params[K.PID_K_P], params[K.PID_K_I], params[K.PID_K_D] = 2., .5, .3
    params[K.PID_MAX_ERR_I], params[K.PID_LIMIT] = np.inf, 50.
    state = rng.normal(size=(K.N_PID_STATES, 2))
    kept = state.copy()
    body_state, acceleration = rng.normal(size=12), rng.normal(size=3)
    reference, measurement = rng.normal(size=2) * 5, np.zeros(2)
    _, new_state = cascade.position_stage(
        params, state, reference, measurement, body_state, acceleration, .001, .01, True,
        False)
    np.testing.assert_array_equal(state, kept)
    np.testing.assert_allclose(
        new_state[K.PID_ERR_D],
        params[K.PID_K_D] * (reference - state[K.PID_ERR]) / .01)



def test_a_bare_pid_stage_applies_the_pid_law():
    params = np.zeros((K.N_PID_PARAMS, 1))
    params[K.PID_K_P], params[K.PID_K_I], params[K.PID_K_D] = 2., .5, .25
    params[K.PID_MAX_ERR_I] = np.inf
    x = np.zeros((K.N_PID_STATES, 1))
    x[K.PID_ERR], x[K.PID_INTEGRAL] = .5, 1.
    action, new_x = cascade.pid_stage(params, x, np.array([3.]), np.array([1.]), .1)
    err, integral = 2., 1. + .1 * (.5 + 2.) / 2.
    np.testing.assert_allclose(action, 2. * err + .5 * integral + .25 * (err - .5) / .1)
    np.testing.assert_allclose(new_x[K.PID_REFERENCE], 3.)
    np.testing.assert_allclose(new_x[K.PID_ACTION], action)
