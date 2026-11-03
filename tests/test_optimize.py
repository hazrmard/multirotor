"""
Tests for the controller search in ``multirotor.optimize``.
"""
import numpy as np
import optuna
import pytest

import multirotor.optimize as optimize_module
from multirotor.env import BaseMultirotorEnv
from multirotor.optimize import (
    DEFAULTS, apply_params, get_controller, make_controller_from_trial, make_env,
    make_objective, make_scenario_suites, optimize, run_scenarios
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



def midpoint_params() -> dict:
    "The parameters of a trial that takes the midpoint of every range."
    trial = RecordingTrial()
    make_controller_from_trial(trial)
    return {name: (low + high) / 2 for name, (low, high) in trial.ranges.items()}



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
    params = midpoint_params()
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
    assert DEFAULTS.max_tilt < BaseMultirotorEnv.max_angle



def test_scenarios_are_decided_by_the_seed_and_spread_over_distances():
    train, heldout = make_scenario_suites(DEFAULTS, seed=3)
    assert train.shape == (DEFAULTS.num_sims, 12)
    assert heldout.shape == (DEFAULTS.num_heldout_sims, 12)
    np.testing.assert_array_equal(train, make_scenario_suites(DEFAULTS, seed=3)[0])
    assert not np.array_equal(train, make_scenario_suites(DEFAULTS, seed=4)[0])
    assert not np.array_equal(train[:, :3], heldout[:len(train), :3])
    half_box = DEFAULTS.bounding_box * 0.75 / 2
    for suite in (train, heldout):
        assert np.all(np.abs(suite[:, :3]) <= half_box + 1e-5)
        assert np.all(np.abs(suite[:, :3]) >= 0.5 - 1e-6)    # `proximity` of the envs
        assert np.all(suite[:, 3:] == 0)
        distance = np.linalg.norm(suite[:, :3], axis=1)
        assert distance.min() < 2.5 and distance.max() > 10  # near and far starts



def test_scenarios_of_a_suite_are_distinct():
    for seed in range(20):
        for suite in make_scenario_suites(DEFAULTS, seed):
            assert len(np.unique(suite[:, :3], axis=0)) == len(suite)



def test_the_score_of_a_trial_does_not_depend_on_its_number():
    vp, sp = make_params(4, 'quadratic', False, np.float32)
    params = midpoint_params()
    objective = make_objective(vp, sp, seed=5)
    scores = [objective(optuna.trial.FixedTrial(params, number=n)) for n in (0, 7)]
    assert scores[0] == scores[1]



def test_optimize_scores_the_best_parameters_on_heldout_scenarios():
    vp, sp = make_params(4, 'quadratic', False, np.float32)
    study = optimize(vp, sp, ntrials=2, verbosity=optuna.logging.ERROR, seed=2)
    env = make_env(vp, sp)
    ctrl = get_controller(env.vehicle)
    apply_params(ctrl, study.best_params)
    _, heldout = make_scenario_suites(seed=2, dtype=sp.dtype)
    assert study.user_attrs['heldout_score'] == run_scenarios(env, ctrl, heldout)



@pytest.mark.parametrize('field, value', (
    ('seed', 1), ('num_sims', 10), ('num_heldout_sims', 10), ('bounding_box', 30)))
def test_resuming_a_study_with_other_scenarios_is_an_error(monkeypatch, field, value):
    vp, sp = make_params(4, 'quadratic', False, np.float32)
    scenarios = dict(
        seed=2, num_sims=DEFAULTS.num_sims, num_heldout_sims=DEFAULTS.num_heldout_sims,
        bounding_box=DEFAULTS.bounding_box)
    scenarios[field] = value
    study = optuna.create_study(direction='maximize')
    study.set_user_attr('scenarios', scenarios)
    monkeypatch.setattr(optimize_module, 'get_study', lambda *args, **kwargs: study)
    with pytest.raises(ValueError, match='scenarios'):
        optimize(vp, sp, ntrials=1, seed=2, verbosity=optuna.logging.ERROR)
