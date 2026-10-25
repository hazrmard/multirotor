"""
Passive containers that hold the parameters and state of a vehicle's propellers
and motors as arrays, one column or element per propeller. They compute nothing
and advance nothing: the classes in `multirotor.simulation` read and write them,
and the kernels in `multirotor.physics` and `multirotor.dynamics` take their
arrays as inputs.
"""
from typing import List

import numpy as np

from .vehicle import MotorParams, PropellerParams
from . import physics



def _value(x: float) -> float:
    """Missing (None) parameters are stored as NaN."""
    return np.nan if x is None else float(x)



class MotorData:
    """
    The parameters and state of ``n`` motors.

    Attributes
    ----------
    params : np.ndarray
        Parameter matrix, shape (N_MOTOR_PARAMS, n). Rows are named by the
        ``MOTOR_*`` constants in `multirotor.physics`.
    speed : np.ndarray
        Motor speeds (rad/s), shape (n,).
    voltage : np.ndarray
        Motor voltages (V), shape (n,).
    current : np.ndarray
        Motor currents (A), shape (n,).
    """

    def __init__(self, params: List[MotorParams]) -> None:
        """
        Parameters
        ----------
        params : List[MotorParams]
            The parameters of each motor.
        """
        n = len(params)
        self.params = np.zeros((physics.N_MOTOR_PARAMS, n))
        for i, p in enumerate(params):
            if p.k_drag is None:
                raise ValueError(
                    'MotorParams.k_drag is None. Set PropellerParams.k_drag, '
                    'which is copied to the motor.')
            if p.k_torque is None:
                raise ValueError('MotorParams needs k_torque or k_emf.')
            self.params[physics.MOTOR_RESISTANCE, i] = p.resistance
            self.params[physics.MOTOR_K_EMF, i] = p.k_emf
            self.params[physics.MOTOR_K_TORQUE, i] = p.k_torque
            self.params[physics.MOTOR_K_DF, i] = p.k_df
            self.params[physics.MOTOR_K_DRAG, i] = p.k_drag
            self.params[physics.MOTOR_MOMENT_OF_INERTIA, i] = p.moment_of_inertia
            self.params[physics.MOTOR_SPEED_VOLTAGE_SCALING, i] = p.speed_voltage_scaling
            self.params[physics.MOTOR_MAX_CURRENT, i] = p.max_current
        self.speed = np.zeros(n)
        self.voltage = np.zeros(n)
        self.current = np.zeros(n)
        # Angular acceleration from the last step, used by the explicit motor
        # update (`physics.motor_step_explicit`).
        self._last_acc = np.zeros(n)


    def __len__(self) -> int:
        return self.speed.shape[0]


    def reset(self) -> None:
        """Set all state to zero, in place."""
        self.speed[:] = 0.
        self.voltage[:] = 0.
        self.current[:] = 0.
        self._last_acc[:] = 0.



class PropellerData:
    """
    The parameters and state of ``n`` propellers, and of their motors if they
    have any. Propellers must be homogeneous: either all have a motor or none
    does, and all use the same thrust model.

    Attributes
    ----------
    params : np.ndarray
        Parameter matrix, shape (N_PROP_PARAMS, n). Rows are named by the
        ``PROP_*`` constants in `multirotor.physics`.
    use_thrust_constant : bool
        Whether thrust is ``k_thrust * speed**2`` (True) or from the
        blade-element model (False).
    speed : np.ndarray
        Propeller speeds (rad/s), shape (n,). With motors, this is the same
        array object as ``motors.speed``, so each speed is stored once.
    motors : MotorData
        The motors' data, or None if the propellers have no motors.
    """

    def __init__(self, params: List[PropellerParams]) -> None:
        """
        Parameters
        ----------
        params : List[PropellerParams]
            The parameters of each propeller.

        Raises
        ------
        ValueError
            If the propellers are not homogeneous.
        """
        has_motor = {p.motor is not None for p in params}
        if len(has_motor) > 1:
            raise ValueError('Either all propellers must have a motor, or none.')
        use_thrust_constant = {bool(p.use_thrust_constant) for p in params}
        if len(use_thrust_constant) > 1:
            raise ValueError('All propellers must have the same use_thrust_constant.')
        self.use_thrust_constant = use_thrust_constant.pop()

        n = len(params)
        self.params = np.zeros((physics.N_PROP_PARAMS, n))
        for i, p in enumerate(params):
            self.params[physics.PROP_K_THRUST, i] = _value(p.k_thrust)
            self.params[physics.PROP_K_DRAG, i] = _value(p.k_drag)
            self.params[physics.PROP_MOMENT_OF_INERTIA, i] = p.moment_of_inertia
            self.params[physics.PROP_RADIUS, i] = p.R
            self.params[physics.PROP_AREA, i] = p.A
            self.params[physics.PROP_LIFT_SLOPE, i] = p.a
            self.params[physics.PROP_BLADES, i] = p.b
            self.params[physics.PROP_CHORD, i] = p.c
            self.params[physics.PROP_EFFICIENCY, i] = p.eta
            self.params[physics.PROP_THETA0, i] = p.theta0
            self.params[physics.PROP_THETA1, i] = p.theta1

        if has_motor.pop():
            self.motors = MotorData([p.motor for p in params])
            self.speed = self.motors.speed
        else:
            self.motors = None
            self.speed = np.zeros(n)


    def __len__(self) -> int:
        return self.speed.shape[0]


    def reset(self) -> None:
        """Set all state to zero, in place."""
        self.speed[:] = 0.
        if self.motors is not None:
            self.motors.reset()
