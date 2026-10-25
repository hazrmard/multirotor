"""
Contract tests: what ``step``, ``apply_*`` and ``reset`` do on each component and
on ``Multirotor``. These hold for every implementation, whatever the numerics.
"""
from copy import deepcopy

import numpy as np
import pytest

from multirotor.models.tarott18 import MP, PP
from multirotor.simulation import Battery, Motor, Multirotor, Propeller
from multirotor.vehicle import BatteryParams, SimulationParams

from common import hover_signals, make_params, nominal_speed

SP = SimulationParams(dt=0.01, g=9.81)



def make_propeller(motor: bool) -> Propeller:
    params = deepcopy(PP)
    if motor:
        params.motor = deepcopy(MP)
        params.motor.k_drag = params.k_drag
    return Propeller(params, SP)



@pytest.mark.parametrize('motor', (False, True))
def test_propeller_step_advances_and_returns_speed(motor):
    prop = make_propeller(motor)
    speed = prop.step(300.)
    assert np.ndim(speed) == 0
    assert prop.speed == speed
    assert speed > 0
    if not motor:
        assert speed == 300.



@pytest.mark.parametrize('motor', (False, True))
def test_propeller_apply_speed_is_step_without_state_change(motor):
    prop = make_propeller(motor)
    prop.step(200.)
    before = (prop.speed, None if prop.motor is None else (prop.motor.speed, prop.motor.voltage, prop.motor.current))
    predicted = prop.apply_speed(300.)
    after = (prop.speed, None if prop.motor is None else (prop.motor.speed, prop.motor.voltage, prop.motor.current))
    assert before == after
    assert prop.step(300.) == predicted



@pytest.mark.parametrize('motor', (False, True))
def test_propeller_reset(motor):
    prop = make_propeller(motor)
    prop.step(300.)
    prop.reset()
    assert prop.speed == 0.
    if motor:
        assert (prop.motor.speed, prop.motor.voltage, prop.motor.current) == (0., 0., 0.)



def test_motor_step_apply_reset():
    params = deepcopy(MP)
    params.k_drag = PP.k_drag
    motor = Motor(params, SP)
    signal = 300.
    predicted = motor.apply_speed(signal)
    assert (motor.speed, motor.voltage, motor.current) == (0., 0., 0.)
    speed = motor.step(signal)
    assert np.ndim(speed) == 0
    assert speed == predicted == motor.speed
    assert motor.voltage > 0 and motor.current > 0
    motor.reset()
    assert (motor.speed, motor.voltage, motor.current) == (0., 0., 0.)



def test_motor_step_respects_max_voltage():
    params = deepcopy(MP)
    params.k_drag = PP.k_drag
    motor = Motor(params, SP)
    for _ in range(50):
        motor.step(1000., max_voltage=5.)
    assert motor.voltage <= 5.



def test_battery_reset_and_step():
    battery = Battery(BatteryParams(max_voltage=22.2), SP)
    assert battery.reset() == 22.2
    battery.step()
    assert battery.voltage == 22.2



@pytest.mark.parametrize('motor', (False, True))
def test_multirotor_step_speeds(motor):
    vp, sp = make_params(8, 'quadratic', motor)
    vehicle = Multirotor(vp, sp)
    u = hover_signals(vp, sp)
    before = vehicle.state
    state = vehicle.step_speeds(u)
    assert state.shape == (12,)
    assert state is vehicle.state
    assert vehicle.state is not before, 'state must be a new array after each step'
    assert vehicle.t == pytest.approx(sp.dt)
    vehicle.step_speeds(u)
    assert vehicle.t == pytest.approx(2 * sp.dt)



def test_multirotor_step_dynamics():
    vp, sp = make_params(8, 'quadratic', False)
    vehicle = Multirotor(vp, sp)
    before = vehicle.state
    state = vehicle.step_dynamics(np.asarray([0, 0, vehicle.weight * 1.1, 0, 0, 0], vehicle.dtype))
    assert state.shape == (12,)
    assert vehicle.state is not before
    assert vehicle.t == pytest.approx(sp.dt)
    assert state[2] > 0



@pytest.mark.parametrize('motor', (False, True))
def test_multirotor_reset(motor):
    vp, sp = make_params(8, 'quadratic', motor)
    vehicle = Multirotor(vp, sp)
    for _ in range(5):
        vehicle.step_speeds(hover_signals(vp, sp) * 1.1)
    state = vehicle.reset()
    assert vehicle.t == 0.
    np.testing.assert_array_equal(state, np.zeros(12))
    assert state is vehicle.state
    for p in vehicle.propellers:
        assert p.speed == 0.
        if motor:
            assert (p.motor.speed, p.motor.voltage, p.motor.current) == (0., 0., 0.)
