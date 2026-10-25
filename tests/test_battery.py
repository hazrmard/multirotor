"""
Tests for the battery model: parameters, the battery equations, and a vehicle
flying on a battery.
"""
from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import brentq

from multirotor.data import MotorData
from multirotor.dynamics import rhs_motor, rhs_motor_battery
from multirotor.models.tarott18 import MP, PP
from multirotor.physics import battery_operating_point, battery_terminal, motor_power
from multirotor.simulation import Battery, Multirotor
from multirotor.vehicle import BatteryParams, SimulationParams

from common import N_SCHEDULE, hover_signals, make_params, nominal_speed, schedule, settle_at_hover

SP = SimulationParams(dt=0.01)



def make_vehicle(battery: BatteryParams, dt: float=0.01) -> Multirotor:
    """A vehicle with motors and ``battery``, settled at hover."""
    vp, sp = make_params(8, 'quadratic', True, dtype=np.float64, dt=dt, battery=battery)
    vehicle = Multirotor(vp, sp)
    settle_at_hover(vehicle, nominal_speed(vp, sp))
    return vehicle



def hover(vehicle: Multirotor) -> np.ndarray:
    return hover_signals(vehicle.params, vehicle.simulation)



# Parameters ------------------------------------------------------------------

def test_battery_params_validation():
    with pytest.raises(ValueError, match='both'):
        BatteryParams(max_voltage=22.2, ocv_soc=[0, 1])
    with pytest.raises(ValueError, match='increasing'):
        BatteryParams(capacity=5., ocv_soc=[1, 0], ocv_voltage=[22, 18])
    with pytest.raises(ValueError, match='finite'):
        BatteryParams(capacity=5.)



def test_battery_model_needs_motors_and_quadratic_thrust():
    vp, sp = make_params(8, 'quadratic', False, battery=BatteryParams(max_voltage=22.2, capacity=5.))
    with pytest.raises(ValueError, match='motors'):
        Multirotor(vp, sp)
    vp, sp = make_params(8, 'fsolve', True, battery=BatteryParams(max_voltage=22.2, capacity=5.))
    with pytest.raises(ValueError, match='blade-element'):
        Multirotor(vp, sp)



# Battery equations -----------------------------------------------------------

def test_terminal_voltage():
    assert battery_terminal(100., 20., 0.) == (20., True)
    voltage, ok = battery_terminal(100., 20., 0.1)
    assert ok
    # V = V_oc - R * P / V
    assert voltage == pytest.approx(20. - 0.1 * 100. / voltage, rel=1e-12)



def test_terminal_voltage_overload():
    # Maximum power is V_oc**2 / (4 R) = 1000 W
    assert battery_terminal(1001., 20., 0.1) == (10., False)



def test_operating_point_with_voltage_limited_motors():
    params = deepcopy(MP)
    params.k_drag = PP.k_drag
    motor_params = MotorData([params] * 4).params
    speed = np.full(4, 300.)
    u = np.asarray([300., 500., 700., 900.])   # the larger signals exceed the voltage
    v_oc, resistance = 20., 0.05
    voltage, power, ok = battery_operating_point(speed, u, v_oc, resistance, motor_params)
    assert ok
    assert params.speed_voltage_scaling * u.max() > voltage, 'some motors must be limited'
    # Reference: the voltage V at which V equals the terminal voltage for the
    # power the motors draw at V.
    def residual(v):
        return v - battery_terminal(motor_power(speed, u, v, motor_params), v_oc, resistance)[0]
    expected = brentq(residual, 1e-9, v_oc, xtol=1e-14)
    assert voltage == pytest.approx(expected, rel=1e-10)
    assert power == pytest.approx(motor_power(speed, u, voltage, motor_params), rel=1e-12)



def test_standalone_battery():
    battery = Battery(BatteryParams(
        capacity=1., internal_resistance=0.1, ocv_soc=[0., 1.], ocv_voltage=[18., 22.]), SP)
    assert battery.voltage == 22. and battery.soc == 1.
    battery.step()
    assert battery.charge_drawn == 0.
    battery.step(load=10.)
    assert battery.charge_drawn == pytest.approx(0.1)
    assert battery.current == 10.
    assert battery.soc == pytest.approx(1. - 0.1 / 3600.)
    assert battery.voltage == pytest.approx(battery.open_circuit_voltage() - 1.)
    assert battery.reset() == 22. and battery.charge_drawn == 0.



# Vehicle on a battery --------------------------------------------------------

def test_defaults_keep_the_voltage_limit_model():
    vehicle = make_vehicle(BatteryParams(max_voltage=22.2))
    assert not vehicle.battery.has_model
    assert vehicle._rhs is rhs_motor
    for _ in range(5):
        vehicle.step_speeds(hover(vehicle))
    assert vehicle.battery.voltage == 22.2
    assert vehicle.battery.charge_drawn == 0.
    motors = vehicle._data.motors
    assert vehicle.battery_current == pytest.approx(np.sum(motors.current * motors.voltage) / 22.2)



def test_ideal_battery_matches_voltage_limit_model():
    # No resistance and a flat OCV: the motors see the same voltage limit.
    limit = make_vehicle(BatteryParams(max_voltage=22.2))
    ideal = make_vehicle(BatteryParams(max_voltage=22.2, capacity=5.))
    assert ideal._rhs is rhs_motor_battery
    us = schedule(hover(limit), 0.05, N_SCHEDULE)
    for u in us:
        limit.step_speeds(u)
        ideal.step_speeds(u)
    np.testing.assert_allclose(ideal.state, limit.state, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ideal.speeds, limit.speeds, rtol=1e-6)



def test_charge_under_constant_load():
    vehicle = make_vehicle(BatteryParams(max_voltage=22.2, capacity=5.))
    u = hover(vehicle)
    vehicle.step_speeds(u)
    motors = vehicle._data.motors
    current = np.sum(motors.current * motors.voltage) / 22.2
    steps = 100
    for _ in range(steps):
        vehicle.step_speeds(u)
    assert vehicle.battery.charge_drawn == pytest.approx(current * (steps + 1) * 0.01, rel=1e-3)
    assert vehicle.battery_current == pytest.approx(current, rel=1e-3)



def test_charge_converges_with_time_step():
    # Total charge drawn at dt = 10 ms matches a run at dt = 1 ms.
    battery = BatteryParams(max_voltage=22.2, capacity=5., internal_resistance=0.05)
    coarse, fine = make_vehicle(battery, dt=0.01), make_vehicle(battery, dt=0.001)
    for u in schedule(hover(coarse), 0.05, N_SCHEDULE):
        coarse.step_speeds(u)
        for _ in range(10):
            fine.step_speeds(u)
    assert coarse.battery.charge_drawn == pytest.approx(fine.battery.charge_drawn, rel=5e-3)



def test_voltage_sags_with_current():
    resistance = 0.05
    vehicle = make_vehicle(BatteryParams(max_voltage=22.2, capacity=5., internal_resistance=resistance))
    for _ in range(20):
        vehicle.step_speeds(hover(vehicle))
    battery = vehicle.battery
    assert battery.voltage < battery.open_circuit_voltage()
    assert battery.voltage == pytest.approx(
        battery.open_circuit_voltage() - resistance * battery.current, rel=1e-3)
    assert battery.current == pytest.approx(vehicle.battery_current)



def test_motors_are_limited_by_a_discharged_battery():
    # As the battery discharges, its voltage falls below what the motors
    # request: they are limited to the terminal voltage, lose thrust, and the
    # vehicle descends.
    vehicle = make_vehicle(BatteryParams(
        capacity=0.002, ocv_soc=[0., 1.], ocv_voltage=[6., 22.2]))
    u = hover(vehicle)
    requested = MP.speed_voltage_scaling * u[0]
    for _ in range(200):
        vehicle.step_speeds(u)
    assert vehicle.battery.voltage < requested
    np.testing.assert_allclose(vehicle._data.motors.voltage, vehicle.battery.voltage, rtol=1e-6)
    assert vehicle.inertial_velocity[2] < 0



def test_overload_sets_flag():
    vehicle = make_vehicle(BatteryParams(max_voltage=22.2, capacity=5., internal_resistance=2.))
    vehicle.step_speeds(hover(vehicle) * 1.5)
    assert vehicle.battery.overload



def test_cutoff_latches_until_reset():
    # The voltage sag under load (about 1 V) triggers the cutoff while the
    # open-circuit voltage is still above min_voltage.
    min_voltage = 20.
    vehicle = make_vehicle(BatteryParams(
        capacity=0.005, internal_resistance=0.05, ocv_soc=[0., 1.],
        ocv_voltage=[15., 22.2], min_voltage=min_voltage))
    u = hover(vehicle)
    voltages, cutoffs = [], []
    for _ in range(200):
        vehicle.step_speeds(u)
        voltages.append(vehicle.battery.voltage)
        cutoffs.append(vehicle.battery.cutoff)
        if vehicle.battery.cutoff:
            break
    k = cutoffs.index(True)
    assert voltages[k] < min_voltage <= voltages[k - 1], 'cutoff within one step'
    # From the next step on, the motors get no voltage and draw no current
    vehicle.step_speeds(u)
    assert not vehicle._data.motors.voltage.any()
    assert vehicle.battery.current == 0.
    # The latch holds after the unloaded voltage recovers above min_voltage
    assert vehicle.battery.voltage > min_voltage
    vehicle.step_speeds(u)
    assert vehicle.battery.cutoff
    vehicle.reset()
    assert not vehicle.battery.cutoff and vehicle.battery.voltage == 22.2



def test_current_reporting():
    vehicle = make_vehicle(BatteryParams(max_voltage=22.2, capacity=5., internal_resistance=0.02))
    for _ in range(20):
        vehicle.step_speeds(hover(vehicle))
    motors = vehicle._data.motors
    np.testing.assert_array_equal(vehicle.motor_currents, motors.current)
    # At steady hover the step-average battery current equals the
    # instantaneous one: motor power divided by the terminal voltage.
    assert vehicle.battery_current == pytest.approx(
        np.sum(motors.current * motors.voltage) / vehicle.battery.voltage, rel=5e-3)

    vp, sp = make_params(8, 'quadratic', True)
    assert np.isnan(Multirotor(vp, sp).battery_current), 'no battery voltage'
    vp, sp = make_params(8, 'quadratic', False, battery=BatteryParams(max_voltage=22.2))
    no_motors = Multirotor(vp, sp)
    assert np.isnan(no_motors.battery_current)
    assert no_motors.motor_currents is None
