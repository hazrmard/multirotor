"""
Shared builders for the test suite: vehicles in every configuration, hover
inputs, input schedules, and comparisons against the frozen oracle in
``tests/legacy``.
"""
from copy import deepcopy
from dataclasses import replace
from typing import Tuple

import numpy as np

from multirotor.helpers import find_nominal_speed, vehicle_params_factory
from multirotor.models.tarott18 import MP, PP
from multirotor.simulation import Multirotor
from multirotor.vehicle import BatteryParams, MotorParams, PropellerParams, SimulationParams, VehicleParams

from legacy.simulation import Multirotor as LegacyMultirotor, Propeller as LegacyPropeller

M_PROP = 0.5    # kg per propeller
M_BODY = 4.0    # kg
D_PROP = 0.635  # m, arm length (same as the tarott18 octorotor)
N_SCHEDULE = 64 # steps in one input schedule

THRUST_MODELS = ('quadratic', 'fsolve')
EXCITE_FRAC = {'quadratic': 0.05, 'fsolve': 0.002}
"Input perturbation per thrust model (the benchmark's values)."



def make_params(
    n: int=8, thrust_model: str='quadratic', motor: bool=False,
    dtype: type=np.float32, dt: float=0.01, battery: BatteryParams=None
) -> Tuple[VehicleParams, SimulationParams]:
    """
    Make the parameters of a vehicle with ``n`` identical propellers.

    Parameters
    ----------
    n : int, optional
        Number of propellers, by default 8
    thrust_model : str, optional
        ``'quadratic'`` for ``k_thrust * speed**2``, or ``'fsolve'`` for the
        blade-element model, by default 'quadratic'
    motor : bool, optional
        Whether every propeller has the ``tarott18`` motor, by default False
    dtype : type, optional
        The simulation data type, by default np.float32
    dt : float, optional
        The simulation time step, by default 0.01
    battery : BatteryParams, optional
        The battery parameters, by default None

    Returns
    -------
    Tuple[VehicleParams, SimulationParams]
        The vehicle and simulation parameters.
    """
    if thrust_model == 'quadratic':
        prop = deepcopy(PP)
    elif thrust_model == 'fsolve':
        prop = PropellerParams(
            moment_of_inertia=PP.moment_of_inertia, use_thrust_constant=False,
            diameter=18., pitch=6., k_drag=PP.k_drag, k_thrust=PP.k_thrust)
    else:
        raise ValueError(thrust_model)
    if motor:
        prop = replace(prop, motor=deepcopy(MP))
    vp = vehicle_params_factory(n, M_PROP, D_PROP, prop, M_BODY)
    vp.battery = battery
    sp = SimulationParams(dt=dt, g=9.81, dtype=dtype)
    return vp, sp



def make_pair(*args, **kwargs) -> Tuple[Multirotor, LegacyMultirotor]:
    """
    Make the same vehicle twice: with the current implementation and with the
    frozen oracle. Arguments are passed to `make_params`.

    Returns
    -------
    Tuple[Multirotor, LegacyMultirotor]
        The current and the legacy vehicle.
    """
    vp, sp = make_params(*args, **kwargs)
    return Multirotor(vp, sp), LegacyMultirotor(vp, sp)



def nominal_speed(vp: VehicleParams, sp: SimulationParams) -> float:
    """
    Propeller speed (rad/s) at which each propeller carries its share of the
    weight, for either thrust model.
    """
    weight_per_rotor = vp.mass * sp.g / len(vp.propellers)
    thrust = LegacyPropeller(vp.propellers[0], sp).thrust
    return float(find_nominal_speed(lambda w: thrust(float(np.ravel(w)[0])), weight_per_rotor))



def motor_signal(motor: MotorParams, speed: float) -> float:
    """
    Speed signal that holds a motor at ``speed`` (rad/s) in steady state.
    """
    current = (motor.k_drag * speed**2 + motor.k_df * speed) / motor.k_torque
    voltage = motor.k_emf * speed + motor.resistance * current
    return voltage / motor.speed_voltage_scaling



def hover_signals(vp: VehicleParams, sp: SimulationParams) -> np.ndarray:
    """
    Per-propeller speed signals that hold the vehicle at hover.
    """
    speed = nominal_speed(vp, sp)
    motor = vp.propellers[0].motor
    u = speed if motor is None else motor_signal(motor, speed)
    return np.full(len(vp.propellers), u, sp.dtype)



def schedule(
    base: np.ndarray, frac: float, steps: int=N_SCHEDULE, seed: int=0
) -> np.ndarray:
    """
    Deterministic inputs ``base * (1 +- frac)``, one row per step.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-frac, frac, size=(steps, *np.shape(base)))
    return (np.asarray(base) * (1 + noise)).astype(np.asarray(base).dtype)



def settle_at_hover(vehicle, speed: float) -> None:
    """
    Start the propellers (and motors) of a current or legacy vehicle at
    ``speed`` instead of at rest.
    """
    for p in vehicle.propellers:
        p.speed = speed
        if p.motor is not None:
            p.motor.speed = speed



def atol_for(dtype: type) -> float:
    """
    Tolerance for comparing a current and a legacy vehicle.
    """
    return 1e-6 if dtype == np.float32 else 1e-10



def snapshot(vehicle) -> dict:
    """
    The observable state of a current or legacy vehicle, as arrays.
    """
    snap = dict(
        state=np.asarray(vehicle.state, float),
        acceleration=np.asarray(vehicle.acceleration, float),
        t=np.asarray(vehicle.t, float),
        speeds=np.asarray([p.speed for p in vehicle.propellers], float),
    )
    if vehicle.propellers[0].motor is not None:
        snap['motor_speeds'] = np.asarray([p.motor.speed for p in vehicle.propellers], float)
        snap['motor_voltages'] = np.asarray([p.motor.voltage for p in vehicle.propellers], float)
        snap['motor_currents'] = np.asarray([p.motor.current for p in vehicle.propellers], float)
    return snap



def assert_snapshots_close(new: dict, old: dict, atol: float, context: str='') -> None:
    """
    Assert that two `snapshot` results match within ``atol``.
    """
    assert new.keys() == old.keys(), context
    for key in new:
        np.testing.assert_allclose(
            new[key], old[key], rtol=0, atol=atol, err_msg=f'{context}: {key}')
