"""
Structural tests: the parameter matrices, components indexing into the vehicle's state, and
copying.
"""
import pickle
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from multirotor import physics
from multirotor.builder import build_vehicle, motor_matrix, propeller_matrix
from multirotor.models.tarott18 import MP, PP
from multirotor.simulation import Multirotor
from multirotor.vehicle import PropellerParams

from common import hover_signals, make_params











def test_reset_is_in_place():
    vp, sp = make_params(8, 'quadratic', True)
    vehicle = Multirotor(vp, sp)
    speed, current = vehicle._speed, vehicle._current
    speed[:] = 100.
    current[:] = 5.
    vehicle.reset()
    assert vehicle._speed is speed and vehicle._current is current
    assert not speed.any() and not current.any()
    assert vehicle.propellers[3].speed == 0. and vehicle.propellers[3].motor.current == 0.



def test_mixed_motor_presence_raises():
    vp, sp = make_params(4, 'quadratic', False)
    vp.propellers[1] = replace(deepcopy(PP), motor=deepcopy(MP))
    with pytest.raises(ValueError, match='motor'):
        build_vehicle(vp, sp)



def test_mixed_thrust_models_raises():
    fsolve_prop = PropellerParams(
        moment_of_inertia=PP.moment_of_inertia, use_thrust_constant=False,
        k_drag=PP.k_drag, k_thrust=PP.k_thrust)
    with pytest.raises(ValueError, match='use_thrust_constant'):
        propeller_matrix([deepcopy(PP), fsolve_prop])



def test_motor_without_drag_raises():
    motor = deepcopy(MP)
    motor.k_drag = None
    with pytest.raises(ValueError, match='k_drag'):
        motor_matrix([motor])



def test_setters_write_through():
    vp, sp = make_params(8, 'quadratic', True)
    vehicle = Multirotor(vp, sp)
    vehicle.propellers[2].motor.speed = 100.
    assert vehicle._speed[2] == 100.
    assert vehicle.propellers[2].speed == 100.
    vehicle.propellers[3].speed = 50.
    assert vehicle.propellers[3].motor.speed == 50.
    # The next step starts from the speed that was set. Motors integrated with
    # the body agree with motors integrated on their own to the solver's
    # tolerance.
    expected = vehicle.propellers[2].apply_speed(0.)
    vehicle.step_speeds(np.zeros(8, vehicle.dtype))
    assert vehicle.propellers[2].speed == pytest.approx(expected, rel=1e-3)



@pytest.mark.parametrize('copy', ['deepcopy', 'pickle'])
def test_copies_have_their_own_state(copy):
    vp, sp = make_params(8, 'quadratic', True)
    vehicle = Multirotor(vp, sp)
    vehicle.step_speeds(hover_signals(vp, sp))
    if copy == 'deepcopy':
        clone = deepcopy(vehicle)
    else:
        clone = pickle.loads(pickle.dumps(vehicle))
    assert clone._speed is not vehicle._speed
    for i, p in enumerate(clone.propellers):
        # The components index into the clone's arrays, not the original's
        assert p._speed is clone._speed and p.motor._speed is clone._speed
        assert p.motor._current is clone._current
    clone.propellers[0].speed = 1.
    assert vehicle.propellers[0].speed != 1.
    np.testing.assert_array_equal(clone.speeds[1:], vehicle.speeds[1:])
