"""
Structural tests: data classes, components indexing into them, and copying.
"""
import pickle
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from multirotor import physics
from multirotor.data import MotorData, PropellerData
from multirotor.models.tarott18 import MP, PP
from multirotor.simulation import Multirotor
from multirotor.vehicle import PropellerParams

from common import hover_signals, make_params



def test_propeller_data_compiles_params():
    props = [deepcopy(PP) for _ in range(3)]
    props[1].k_thrust = 2 * PP.k_thrust
    data = PropellerData(props)
    assert data.params.shape == (physics.N_PROP_PARAMS, 3)
    np.testing.assert_array_equal(
        data.params[physics.PROP_K_THRUST], [PP.k_thrust, 2 * PP.k_thrust, PP.k_thrust])
    assert data.use_thrust_constant
    assert data.motors is None
    assert len(data) == 3



def test_motor_data_compiles_params():
    prop = replace(deepcopy(PP), motor=deepcopy(MP))
    data = PropellerData([prop, prop])
    motors = data.motors
    assert motors.params.shape == (physics.N_MOTOR_PARAMS, 2)
    np.testing.assert_array_equal(motors.params[physics.MOTOR_RESISTANCE], MP.resistance)
    np.testing.assert_array_equal(motors.params[physics.MOTOR_K_DRAG], PP.k_drag)
    assert data.speed is motors.speed, 'propeller speed is stored once, in the motor data'



def test_data_reset_is_in_place():
    prop = replace(deepcopy(PP), motor=deepcopy(MP))
    data = PropellerData([prop, prop])
    speed, current = data.speed, data.motors.current
    speed[:] = 100.
    current[:] = 5.
    data.reset()
    assert data.speed is speed and data.motors.current is current
    assert not speed.any() and not current.any()



def test_mixed_motor_presence_raises():
    with pytest.raises(ValueError, match='motor'):
        PropellerData([deepcopy(PP), replace(deepcopy(PP), motor=deepcopy(MP))])



def test_mixed_thrust_models_raises():
    fsolve_prop = PropellerParams(
        moment_of_inertia=PP.moment_of_inertia, use_thrust_constant=False,
        k_drag=PP.k_drag, k_thrust=PP.k_thrust)
    with pytest.raises(ValueError, match='use_thrust_constant'):
        PropellerData([deepcopy(PP), fsolve_prop])



def test_motor_without_drag_raises():
    motor = deepcopy(MP)
    motor.k_drag = None
    with pytest.raises(ValueError, match='k_drag'):
        MotorData([motor])



def test_setters_write_through():
    vp, sp = make_params(8, 'quadratic', True)
    vehicle = Multirotor(vp, sp)
    vehicle.propellers[2].motor.speed = 100.
    assert vehicle._data.motors.speed[2] == 100.
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
def test_copies_have_their_own_data(copy):
    vp, sp = make_params(8, 'quadratic', True)
    vehicle = Multirotor(vp, sp)
    vehicle.step_speeds(hover_signals(vp, sp))
    if copy == 'deepcopy':
        clone = deepcopy(vehicle)
    else:
        clone = pickle.loads(pickle.dumps(vehicle))
    assert clone._data is not vehicle._data
    assert clone._data.speed is clone._data.motors.speed
    for p in clone.propellers:
        assert p._data is clone._data
        assert p.motor._data is clone._data.motors
    clone.propellers[0].speed = 1.
    assert vehicle.propellers[0].speed != 1.
    np.testing.assert_array_equal(clone.speeds[1:], vehicle.speeds[1:])
