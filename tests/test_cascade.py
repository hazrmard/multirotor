"""
`multirotor.controller.cascade`, the cascaded controller as a pure function of its
state, against the in-place kernel `multirotor.controller.kernels.cascaded_pid_step`
(numba). The JAX build is tested in `test_cascade_jax`.
"""
import itertools

import numpy as np
import pytest

import multirotor.controller.cascade as cascade
from multirotor.controller import kernels as K

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
    return (rng.normal(size=4) * 3, rng.normal(size=12) * SCALE, rng.normal(size=3),
            rng.normal(size=2))



@pytest.mark.parametrize('leashing,sqrt_pos,sqrt_att,feed_forward,steps', OPTIONS)
def test_matches_in_place_kernel(leashing, sqrt_pos, sqrt_att, feed_forward, steps):
    rng = np.random.default_rng(0)
    params = make_params(rng)
    if not leashing and rng.random() < .5:
        params[K.PID_K_P, :2] = 0.    # the zero-gain branch of the square-root controller
    x_old = rng.normal(size=(K.N_PID_STATES, 12)) * .1
    x_new, memo = x_old.copy(), (np.zeros(2), np.zeros(1), np.zeros(3))
    loop = [np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(1), np.zeros(3)]
    for n in range(40):
        reference, body_state, acceleration, velocity = inputs(rng)
        action = np.empty(4)
        K.cascaded_pid_step(
            params, x_old, reference, body_state, acceleration, INERTIA, MASS, WEIGHT,
            SIM_DT, n, *steps, velocity, feed_forward, .4, leashing, sqrt_pos, sqrt_att,
            True, *loop, action)
        x_new, memo, expected = cascade.cascaded_step(
            params, x_new, memo, reference, body_state, acceleration, INERTIA, MASS,
            WEIGHT, SIM_DT, n, *steps, velocity, feed_forward, .4, leashing, sqrt_pos,
            sqrt_att)
        np.testing.assert_allclose(x_new, x_old, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(expected, action, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(memo[0], loop[0], rtol=1e-12, atol=1e-12)
