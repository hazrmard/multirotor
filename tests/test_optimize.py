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
