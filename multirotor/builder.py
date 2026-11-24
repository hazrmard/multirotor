"""
Build the arrays that the simulation takes from a vehicle's parameters.

`build_vehicle` turns a `multirotor.vehicle.VehicleParams` into the tuple that
`multirotor.rollout` and the compiled steps take: `multirotor.tuples.BodyVehicle`,
`MotorsVehicle` or `BatteryVehicle`. It is a function of the parameters and nothing else.
With numpy values it returns numpy arrays. With JAX arrays or tracers it uses JAX
operations, so it can be differentiated: the gradient of a loss with respect to the arm
length, the propeller's diameter or a motor's resistance passes through it.

The parameter matrices have one row per parameter and one column per propeller or motor.
The ``PROP_*``, ``MOTOR_*`` and ``BATTERY_*`` constants of `multirotor.physics` name the
rows, so ``prop_params[PROP_K_THRUST]`` has one value per propeller and lines up with the
propellers' speeds.
"""
from typing import List, Optional, Sequence, Union

import numpy as np

from . import physics
from .tuples import BatteryVehicle, BodyVehicle, ControllerModel, MotorsVehicle
from .vehicle import BatteryParams, MotorParams, PropellerParams, SimulationParams, VehicleParams, xp_of

Vehicle = Union[BodyVehicle, MotorsVehicle, BatteryVehicle]



def _stack(xp, values: Sequence[float]):
    "A float vector of the values: numbers, or tracers. None is NaN, a missing parameter."
    return xp.stack([xp.asarray(np.nan if v is None else v, dtype=float) for v in values])



def _pack(xp, size: int, rows: dict):
    "A matrix of ``size`` rows, from the rows given by index. The others are zero."
    n = len(next(iter(rows.values())))
    return xp.stack([rows.get(i, xp.zeros(n)) for i in range(size)])



def propeller_matrix(propellers: List[PropellerParams], xp=None) -> np.ndarray:
    """
    The propellers' parameters as a matrix.

    Parameters
    ----------
    propellers : List[PropellerParams]
        The propellers.
    xp : module, optional
        ``numpy`` or ``jax.numpy``. By default, ``jax.numpy`` if any value is a JAX array.

    Returns
    -------
    np.ndarray
        Shape (N_PROP_PARAMS, n). A missing ``k_thrust`` or ``k_drag`` is NaN.

    Raises
    ------
    ValueError
        If the propellers do not all use the same thrust model.
    """
    if len({bool(p.use_thrust_constant) for p in propellers}) > 1:
        raise ValueError('All propellers must have the same use_thrust_constant.')
    xp = xp or xp_of(propellers)
    def column(name: str):
        return _stack(xp, [getattr(p, name) for p in propellers])
    return _pack(xp, physics.N_PROP_PARAMS, {
        physics.PROP_K_THRUST: column('k_thrust'),
        physics.PROP_K_DRAG: column('k_drag'),
        physics.PROP_MOMENT_OF_INERTIA: column('moment_of_inertia'),
        physics.PROP_RADIUS: column('R'),
        physics.PROP_AREA: column('A'),
        physics.PROP_LIFT_SLOPE: column('a'),
        physics.PROP_BLADES: column('b'),
        physics.PROP_CHORD: column('c'),
        physics.PROP_EFFICIENCY: column('eta'),
        physics.PROP_THETA0: column('theta0'),
        physics.PROP_THETA1: column('theta1'),
    })



def motor_matrix(motors: List[MotorParams], xp=None) -> np.ndarray:
    """
    The motors' parameters as a matrix.

    Parameters
    ----------
    motors : List[MotorParams]
        The motors. Their ``k_drag`` is set by their propeller, if they have one.
    xp : module, optional
        ``numpy`` or ``jax.numpy``. By default, ``jax.numpy`` if any value is a JAX array.

    Returns
    -------
    np.ndarray
        Shape (N_MOTOR_PARAMS, n).

    Raises
    ------
    ValueError
        If a motor has no ``k_drag`` or no ``k_torque``.
    """
    for m in motors:
        if m.k_drag is None:
            raise ValueError(
                'MotorParams.k_drag is None. Set PropellerParams.k_drag, '
                'which is copied to the motor.')
        if m.k_torque is None:
            raise ValueError('MotorParams needs k_torque or k_emf.')
    xp = xp or xp_of(motors)
    def column(name: str):
        return _stack(xp, [getattr(m, name) for m in motors])
    return _pack(xp, physics.N_MOTOR_PARAMS, {
        physics.MOTOR_RESISTANCE: column('resistance'),
        physics.MOTOR_K_EMF: column('k_emf'),
        physics.MOTOR_K_TORQUE: column('k_torque'),
        physics.MOTOR_K_DF: column('k_df'),
        physics.MOTOR_K_DRAG: column('k_drag'),
        physics.MOTOR_MOMENT_OF_INERTIA: column('moment_of_inertia'),
        physics.MOTOR_SPEED_VOLTAGE_SCALING: column('speed_voltage_scaling'),
        physics.MOTOR_MAX_CURRENT: column('max_current'),
    })



def battery_arrays(battery: BatteryParams, xp=None) -> tuple:
    """
    The arrays of a battery model, for `multirotor.physics`.

    Parameters
    ----------
    battery : BatteryParams
        A battery with a model (a capacity).
    xp : module, optional
        ``numpy`` or ``jax.numpy``. By default, ``jax.numpy`` if any value is a JAX array.

    Returns
    -------
    tuple
        The parameter vector of length N_BATTERY_PARAMS (the capacity in coulombs, the
        internal resistance and the cutoff voltage), the state of charge of the
        open-circuit voltage curve, and its voltage.
    """
    xp = xp or xp_of(battery)
    values = {
        physics.BATTERY_CAPACITY: 3600. * battery.capacity,
        physics.BATTERY_INTERNAL_RESISTANCE: battery.internal_resistance,
        physics.BATTERY_MIN_VOLTAGE: battery.min_voltage}
    vector = xp.stack([
        xp.asarray(values.get(i, 0.), dtype=float) for i in range(physics.N_BATTERY_PARAMS)])
    if battery.ocv_soc is None:
        # Constant open-circuit voltage
        return vector, xp.asarray([0., 1.]), xp.stack([
            xp.asarray(battery.max_voltage, dtype=float)] * 2)
    return vector, xp.asarray(battery.ocv_soc, dtype=float), \
        xp.asarray(battery.ocv_voltage, dtype=float)



def voltage_limit(battery: Optional[BatteryParams], xp=None) -> float:
    """
    The voltage that a full battery supplies to the motors: its open-circuit voltage with a
    battery model, otherwise its ``max_voltage`` (infinite without a battery).
    """
    if battery is None:
        return np.inf
    if battery.capacity is None:
        return battery.max_voltage
    xp = xp or xp_of(battery)
    _, soc, voltage = battery_arrays(battery, xp)
    return xp.interp(1., soc, voltage)



def build_vehicle(params: VehicleParams, simulation: SimulationParams) -> Vehicle:
    """
    The tuple that the rollout and the compiled steps take, from a vehicle's parameters.

    Parameters
    ----------
    params : VehicleParams
        The vehicle. Its values can be numpy arrays, JAX arrays or tracers.
    simulation : SimulationParams
        The simulation parameters. ``g`` is used.

    Returns
    -------
    BodyVehicle, MotorsVehicle or BatteryVehicle
        Without motors, with motors, or with motors and a battery model. The motors'
        voltage limit is that of a full battery, and the disturbance is zero.

    Raises
    ------
    ValueError
        If only some propellers have a motor, if they do not use the same thrust model,
        if a motor lacks a coefficient, or if a battery model has no motors to draw current.
    """
    xp = xp_of(params)
    propellers = params.propellers
    has_motor = {p.motor is not None for p in propellers}
    if len(has_motor) > 1:
        raise ValueError('Either all propellers must have a motor, or none.')
    has_motor = has_motor.pop()
    battery = params.battery
    has_battery = battery is not None and battery.capacity is not None
    if has_battery and not has_motor:
        raise ValueError('A battery model needs motors to draw current.')

    fields = dict(
        g=xp.asarray(simulation.g, dtype=float), mass=xp.asarray(params.mass, dtype=float),
        inertia=xp.asarray(params.inertia_matrix, dtype=float),
        inertia_inverse=xp.asarray(params.inertia_matrix_inverse, dtype=float),
        prop_params=propeller_matrix(propellers, xp),
        clockwise=xp.asarray(params.clockwise, dtype=float),
        positions=xp.asarray(params.propeller_vectors, dtype=float),
        disturbance=xp.zeros(6))
    if not has_motor:
        return BodyVehicle(**fields)
    fields.update(
        motor_params=motor_matrix([p.motor for p in propellers], xp),
        max_voltage=xp.asarray(voltage_limit(battery, xp), dtype=float))
    if not has_battery:
        return MotorsVehicle(**fields)
    vector, soc, voltage = battery_arrays(battery, xp)
    return BatteryVehicle(**fields, battery_params=vector, ocv_soc=soc, ocv_voltage=voltage)



def controller_model(params: VehicleParams, simulation: SimulationParams) -> ControllerModel:
    """
    What a controller tuned for a vehicle believes about it: its mass, weight and inertia.
    Pass it to a rollout as ``model`` to fly a vehicle whose parameters differ.

    Parameters
    ----------
    params : VehicleParams
        The vehicle the controller was tuned for. Its values can be tracers.
    simulation : SimulationParams
        The simulation parameters. ``g`` is used.

    Returns
    -------
    ControllerModel
    """
    xp = xp_of(params)
    mass = xp.asarray(params.mass, dtype=float)
    return ControllerModel(
        mass, simulation.g * mass, xp.asarray(params.inertia_matrix, dtype=float))
