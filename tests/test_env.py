"""
The reward of the environments. The reference is the numpy implementation that
the compiled kernel replaced.
"""
import numpy as np
import pytest

from multirotor.env import DynamicsMultirotorEnv
from multirotor.simulation import Multirotor

from common import make_params


def reference_reward(env, state, nstate):
    "The reward and flags of `BaseMultirotorEnv.reward` as numpy code."
    dist = np.linalg.norm(nstate[:3])
    reached = dist <= env.proximity
    outofbounds = np.any(np.abs(state[:3]) > env.bounding_box / 2)
    outoftime = env.vehicle.t >= env.period
    tipped = np.any(np.abs(state[6:9]) > env.max_angle)
    delta_pos = nstate[:3] - state[:3]
    advance = np.linalg.norm(delta_pos)
    cross = np.linalg.norm(np.cross(delta_pos, env._des_unit_vec))
    delta_turn = np.abs(nstate[8]) - np.abs(state[8])
    reward = ((advance - cross - delta_turn) * env.motion_reward_scaling) \
        - env.vehicle.simulation.dt
    if reached:
        reward += env.bonus
    elif tipped or outofbounds:
        reward -= env.bonus
    elif outoftime:
        reward -= (dist / env.bounding_box) * env.bonus
    return reward, (bool(reached), bool(outofbounds), bool(outoftime), bool(tipped))


@pytest.mark.parametrize('dtype', (np.float32, np.float64), ids=lambda d: d.__name__)
def test_reward_matches_numpy_reference(dtype):
    vp, sp = make_params(8, 'quadratic', False, dtype)
    env = DynamicsMultirotorEnv(Multirotor(vp, sp), allocate=True)
    env.reset()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        state = rng.uniform(-8, 8, 12).astype(dtype)
        state[6:9] *= 0.05
        nstate = (state + rng.normal(0, 0.05, 12)).astype(dtype)
        if rng.random() < 0.15:
            nstate[:3] *= 0.03                   # near the origin
        if rng.random() < 0.15:
            state[6] = 0.5                       # tipped
        if rng.random() < 0.15:
            state[1] = 12.                       # out of bounds
        direction = rng.normal(size=3)
        env._des_unit_vec = (direction / np.linalg.norm(direction)).astype(dtype)
        env.vehicle.t = rng.uniform(0., 1.2 * env.period)
        expected, flags = reference_reward(env, state, nstate)
        reward = env.reward(state, None, nstate)
        assert reward == pytest.approx(expected, rel=1e-5, abs=1e-4)
        assert (env._reached, env._outofbounds, env._outoftime, env._tipped) == flags
        assert env._done == any(flags)
        seen.add(flags)
    # Each way that an episode ends, and none, must occur in the samples
    assert {(True, False, False, False), (False, True, False, False),
            (False, False, True, False), (False, False, False, True),
            (False, False, False, False)} <= seen
