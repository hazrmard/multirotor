"""
Tests for the controller search in ``multirotor.optimize``.
"""
import numpy as np
import optuna

from multirotor.optimize import (
    apply_params, get_controller, make_controller_from_trial, optimize
)
from multirotor.simulation import Multirotor

from common import make_params


class RecordingTrial:
    """A trial that records the range of each suggestion and returns its midpoint."""

    def __init__(self):
        self.ranges = {}

    def suggest_float(self, name, low, high, **kwargs):
        self.ranges[name] = (low, high)
        return (low + high) / 2



def flatten(params: dict, prefix: str='') -> dict:
    flat = {}
    for name, value in params.items():
        if isinstance(value, dict):
            flat.update(flatten(value, prefix + name + '.'))
        else:
            flat[prefix + name] = np.asarray(value)
    return flat



def test_search_ranges_contain_the_default_gains():
    # Defaults fly. A search range that excludes them can make every trial unstable.
    vp, sp = make_params(4, 'quadratic', False, np.float64)
    ctrl = get_controller(Multirotor(vp, sp))
    trial = RecordingTrial()
    make_controller_from_trial(trial)
    defaults = {
        'r_pitch_roll.k_p': ctrl.ctrl_r.k_p[0], 'r_pitch_roll.k_i': ctrl.ctrl_r.k_i[0],
        'r_pitch_roll.k_d': ctrl.ctrl_r.k_d[0],
        'a_pitch_roll.k_p': ctrl.ctrl_a.k_p[0], 'a_pitch_roll.k_i': ctrl.ctrl_a.k_i[0],
        'a_pitch_roll.k_d': ctrl.ctrl_a.k_d[0],
        **{f'{name}.{gain}': getattr(getattr(ctrl, 'ctrl_' + name), gain)[0]
           for name in ('p', 'v', 'z', 'vz') for gain in ('k_p', 'k_i', 'k_d')},
    }
    for name, value in defaults.items():
        low, high = trial.ranges[name]
        assert low <= value <= high, f'{name}={value} is outside [{low}, {high}]'



def test_apply_params_matches_the_trial_parameters():
    # The controller from `apply_params` must have the gains the trial evaluated.
    recorder = RecordingTrial()
    make_controller_from_trial(recorder)
    params = {name: (low + high) / 2 for name, (low, high) in recorder.ranges.items()}
    from_trial = flatten(make_controller_from_trial(optuna.trial.FixedTrial(params)))
    from_apply = flatten(apply_params(None, params))
    assert set(from_trial) == set(from_apply)
    for name in from_trial:
        np.testing.assert_allclose(from_trial[name], from_apply[name], err_msg=name)



def test_optimize_leaves_the_given_controller_unchanged():
    vp, sp = make_params(4, 'quadratic', False, np.float64)
    vehicle = Multirotor(vp, sp)
    ctrl = get_controller(vehicle)
    k_p = ctrl.ctrl_r.k_p.copy()
    optimize(vp, sp, ctrl, ntrials=2, verbosity=optuna.logging.ERROR)
    assert ctrl.vehicle is vehicle
    np.testing.assert_array_equal(ctrl.ctrl_r.k_p, k_p)



def test_default_tilt_limit_is_below_the_tipping_angle_of_the_environments():
    # An equal limit lets a saturated controller end episodes as "tipped".
    from multirotor.env import BaseMultirotorEnv
    from multirotor.optimize import DEFAULTS
    assert DEFAULTS.max_tilt < BaseMultirotorEnv.max_angle



def test_scenarios_are_decided_by_the_seed_and_spread_over_distances():
    from multirotor.optimize import DEFAULTS, make_scenario_suites
    train, heldout = make_scenario_suites(DEFAULTS, seed=3)
    assert train.shape == (DEFAULTS.num_sims, 12)
    assert heldout.shape == (DEFAULTS.num_heldout_sims, 12)
    np.testing.assert_array_equal(train, make_scenario_suites(DEFAULTS, seed=3)[0])
    assert not np.array_equal(train, make_scenario_suites(DEFAULTS, seed=4)[0])
    assert not np.array_equal(train[:, :3], heldout[:len(train), :3])
    half_box = DEFAULTS.bounding_box * 0.75 / 2
    for suite in (train, heldout):
        assert np.all(np.abs(suite[:, :3]) <= half_box)
        assert np.all(np.abs(suite[:, :3]) >= 0.5)       # `proximity` of the envs
        assert np.all(suite[:, 3:] == 0)
        distance = np.linalg.norm(suite[:, :3], axis=1)
        assert distance.min() < 2.5 and distance.max() > 10  # near and far starts



def test_every_trial_flies_the_same_scenarios():
    from multirotor.optimize import make_env, make_objective, run_scenarios, make_scenario_suites
    vp, sp = make_params(4, 'quadratic', False, np.float32)
    params = {name: (low + high) / 2 for name, (low, high) in recorded_ranges().items()}
    objective = make_objective(vp, sp, seed=5)
    first = objective(optuna.trial.FixedTrial(params))
    assert objective(optuna.trial.FixedTrial(params)) == first
    # The value is the mean reward of the controller over the training scenarios.
    env = make_env(vp, sp)
    ctrl = get_controller(env.vehicle)
    apply_params(ctrl, params)
    train, _ = make_scenario_suites(seed=5, dtype=sp.dtype)
    assert first == run_scenarios(env, ctrl, train)



def recorded_ranges() -> dict:
    trial = RecordingTrial()
    make_controller_from_trial(trial)
    return trial.ranges



def test_optimize_scores_the_best_parameters_on_heldout_scenarios():
    from multirotor.optimize import make_env, run_scenarios, make_scenario_suites
    vp, sp = make_params(4, 'quadratic', False, np.float32)
    study = optimize(vp, sp, ntrials=2, verbosity=optuna.logging.ERROR, seed=2)
    env = make_env(vp, sp)
    ctrl = get_controller(env.vehicle)
    apply_params(ctrl, study.best_params)
    _, heldout = make_scenario_suites(seed=2, dtype=sp.dtype)
    assert study.user_attrs['heldout_score'] == run_scenarios(env, ctrl, heldout)



def test_resuming_a_study_with_other_scenarios_is_an_error(monkeypatch):
    import pytest
    import multirotor.optimize as optimize_module
    vp, sp = make_params(4, 'quadratic', False, np.float32)
    study = optuna.create_study(direction='maximize')
    study.set_user_attr('scenarios', dict(seed=1, num_sims=20, bounding_box=20))
    monkeypatch.setattr(optimize_module, 'get_study', lambda *args, **kwargs: study)
    with pytest.raises(ValueError, match='scenarios'):
        optimize(vp, sp, ntrials=1, seed=2, verbosity=optuna.logging.ERROR)
