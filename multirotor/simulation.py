from typing import Callable, List, NamedTuple, Tuple
from copy import deepcopy

import numpy as np
from scipy.integrate import odeint

from .vehicle import MotorParams, PropellerParams, SimulationParams, VehicleParams, BatteryParams
from .coords import body_to_inertial, direction_cosine_matrix, rotating_frame_derivative, angular_to_euler_rate
from .physics import (
    thrust, thrusts_physics, body_torques, motor_electrical, battery_ocv,
    battery_operating_point,
    PROP_K_THRUST, PROP_K_DRAG, BATTERY_CAPACITY, BATTERY_INTERNAL_RESISTANCE,
    N_BATTERY_PARAMS
)
from .data import MotorData, PropellerData
from .dynamics import (
    rhs_body, rhs_motor, rhs_motor_battery, rhs_motor_blade_element,
    rhs_motor_only, N_BODY_STATES
)
from .helpers import control_allocation_matrix


Integrator = Callable[[Callable, np.ndarray, float, tuple], np.ndarray]
"""
A solver for one simulation step: ``integrator(fun, x0, dt, args) -> x``.

Advances the initial state ``x0`` from ``t=0`` to ``t=dt`` under
``dx/dt = fun(t, x, *args)`` and returns the state at ``t=dt``.
"""


def odeint_integrator(
    fun: Callable, x0: np.ndarray, dt: float, args: tuple = ()
) -> np.ndarray:
    """
    The default :data:`Integrator`: LSODA via `scipy.integrate.odeint`.

    Parameters
    ----------
    fun : Callable
        The derivative function, with signature ``fun(t, x, *args)``.
    x0 : np.ndarray
        The state at the start of the step.
    dt : float
        The step duration.
    args : tuple, optional
        Extra arguments passed to ``fun``.

    Returns
    -------
    np.ndarray
        The state at the end of the step.
    """
    return odeint(fun, x0, (0, dt), args=args, rtol=1e-4, atol=1e-4, tfirst=True)[-1]



class KernelParams(NamedTuple):
    """
    float64 copies of a vehicle's parameters, in the order the right-hand sides
    in `multirotor.dynamics` take them after their per-step arguments.
    """
    g: float
    mass: float
    inertia_matrix: np.ndarray
    inertia_matrix_inverse: np.ndarray
    prop_params: np.ndarray
    "The `PropellerData.params` matrix."
    motor_params: np.ndarray
    "The `MotorData.params` matrix, or None without motors."
    clockwise: np.ndarray
    positions: np.ndarray
    "Position of each propeller relative to the center of mass, shape (3, n)."



class Propeller:
    """
    Models the thrust and aerodynamics of the propeller blades spinning at a
    certain rate.

    The propeller's state lives in a `PropellerData` object, at index ``i``. A
    standalone propeller owns a data object of size 1. Propellers of a
    `Multirotor` share the vehicle's data object, each at its own index.
    """

    def __init__(
        self, params: PropellerParams, simulation: SimulationParams,
        integrator: Integrator = odeint_integrator
    ) -> None:
        """
        Parameters
        ----------
        params : PropellerParams
            The propeller parameters. They are copied.
        simulation : SimulationParams
            The simulation parameters.
        integrator : Integrator, optional
            The solver used to advance the motor, if any. By default
            `odeint_integrator`.
        """
        params = deepcopy(params)
        self._attach(PropellerData([params]), 0, params, simulation, integrator)


    @classmethod
    def _view(
        cls, data: PropellerData, i: int, params: PropellerParams,
        simulation: SimulationParams, integrator: Integrator
    ) -> 'Propeller':
        """
        Create a propeller over an existing data object, at index ``i``.

        Parameters
        ----------
        data : PropellerData
            The shared data object.
        i : int
            The index of this propeller in ``data``.
        params : PropellerParams
            The parameters of this propeller.
        simulation : SimulationParams
            The simulation parameters.
        integrator : Integrator
            The solver used to advance the motor, if any.

        Returns
        -------
        Propeller
        """
        propeller = cls.__new__(cls)
        propeller._attach(data, i, params, simulation, integrator)
        return propeller


    def _attach(
        self, data: PropellerData, i: int, params: PropellerParams,
        simulation: SimulationParams, integrator: Integrator
    ) -> None:
        self.params: PropellerParams = params
        self.simulation: SimulationParams = simulation
        self._data: PropellerData = data
        self._i: int = i

        # Thrust constant uses a simpler quardatic relationship to calculate
        # propeller thrust.
        if self.params.use_thrust_constant:
            self.thrust: Callable = self._thrust_constant
        else:
            self.thrust: Callable = self._thrust_physics
        # If not motor is provided, then speed signals take effect instantaneously
        if data.motors is not None:
            self.motor: Motor = Motor._view(
                data.motors, i, params.motor, simulation, integrator)
        else:
            self.motor: Motor = None


    @property
    def speed(self) -> float:
        "Radians per second"
        return float(self._data.speed[self._i])
    @speed.setter
    def speed(self, speed: float):
        self._data.speed[self._i] = speed


    def reset(self):
        self.speed = 0.
        if self.motor is not None:
            self.motor.reset()


    def apply_speed(self, u: float, **kwargs) -> float:
        """
        Calculate the actual speed of the propeller after the speed signal is
        given. This method is *pure* and does not change the state of the propeller.
        It is used by the multirotor's dxdt_* methods to calculate derivatives.

        Parameters
        ----------
        u : float
            Radians per second speed command

        Returns
        -------
        float
            The actual speed
        """
        if self.motor is not None:
            return self.motor.apply_speed(u, **kwargs)
        return u


    def step(self, u: float, **kwargs) -> float:
        """
        Step through the speed command. This method changes the state of the
        propeller.

        Parameters
        ----------
        u : float
            Speed command in radians per second.

        Returns
        -------
        float
            The actual speed achieved.
        """
        if self.motor is not None:
            self.motor.step(u, **kwargs)
        else:
            self.speed = u
        return self.speed


    def _thrust_constant(self, speed,airstream_velocity: np.ndarray=np.zeros(3)) -> float:
        return self.params.k_thrust * speed**2


    def _thrust_physics(self, speed, airstream_velocity: np.ndarray=np.zeros(3)) -> float:
        p = self.params
        return thrust(
            speed, airstream_velocity,
            p.R, p.A, self.simulation.rho, p.a, p.b, p.c, p.eta, p.theta0, p.theta1
        )


    @property
    def state(self) -> float:
        return self.speed


    @property
    def rpm(self) -> float:
        return self.speed * 60. / (2. * np.pi)



class Motor:
    """
    Models the electronic and mechanical characterists of the rotation of the
    propeller shaft.

    The motor's state lives in a `MotorData` object, at index ``i``. A
    standalone motor owns a data object of size 1. Motors of a `Multirotor`
    share the vehicle's data object, each at its own index.
    """

    def __init__(
        self, params: MotorParams, simulation: SimulationParams,
        integrator: Integrator = odeint_integrator
    ) -> None:
        """
        Parameters
        ----------
        params : MotorParams
            The motor parameters. They are copied.
        simulation : SimulationParams
            The simulation parameters.
        integrator : Integrator, optional
            The solver used to advance the motor speed. By default
            `odeint_integrator`.
        """
        params = deepcopy(params)
        self._attach(MotorData([params]), 0, params, simulation, integrator)


    @classmethod
    def _view(
        cls, data: MotorData, i: int, params: MotorParams,
        simulation: SimulationParams, integrator: Integrator
    ) -> 'Motor':
        """
        Create a motor over an existing data object, at index ``i``.

        Parameters
        ----------
        data : MotorData
            The shared data object.
        i : int
            The index of this motor in ``data``.
        params : MotorParams
            The parameters of this motor.
        simulation : SimulationParams
            The simulation parameters.
        integrator : Integrator
            The solver used to advance the motor speed.

        Returns
        -------
        Motor
        """
        motor = cls.__new__(cls)
        motor._attach(data, i, params, simulation, integrator)
        return motor


    def _attach(
        self, data: MotorData, i: int, params: MotorParams,
        simulation: SimulationParams, integrator: Integrator
    ) -> None:
        self.params: MotorParams = params
        self.simulation: SimulationParams = simulation
        self.integrator: Integrator = integrator
        self._data: MotorData = data
        self._i: int = i


    @property
    def speed(self) -> float:
        "Radians per second"
        return float(self._data.speed[self._i])
    @speed.setter
    def speed(self, speed: float):
        self._data.speed[self._i] = speed


    @property
    def voltage(self) -> float:
        return float(self._data.voltage[self._i])
    @voltage.setter
    def voltage(self, voltage: float):
        self._data.voltage[self._i] = voltage


    @property
    def current(self) -> float:
        return float(self._data.current[self._i])
    @current.setter
    def current(self, current: float):
        self._data.current[self._i] = current


    def reset(self) -> float:
        self.speed = 0.
        self.voltage = 0.
        self.current = 0.


    def current_average(self, max_voltage: float) -> float:
        """
        Average current consumption given the duty cycle of the speed controller.
        Duty cycle depends on max voltage which causes 100% duty cycle.

        Parameters
        ----------
        max_voltage : float
            The peak voltage corresponding to 100% duty cycle.

        Returns
        -------
        float
            Current consumption
        """
        duty_cycle = self.voltage / max_voltage
        return self.current * duty_cycle


    def apply_speed(self, u: float, max_voltage: float=np.inf) -> float:
        """
        Apply a voltage speed signal to the motor. This method is pure and doesn't
        change the state of the motor.

        Parameters
        ----------
        u : float
            Voltage signal.
        max_voltage : float, optional
            The maximum voltage supply from power source. By default infinite.

        Returns
        -------
        float
            The speed of the motor (rad /s)
        """
        return float(self._integrate(u, max_voltage)[0])


    def step(self, u: float, max_voltage: float=np.inf) -> float:
        """
        Apply a voltage speed signal to the motor. This method changes the state
        of the motor.

        Parameters
        ----------
        u : float
            Speed signal (rad/s).
        max_voltage : float, optional
            The maximum voltage supply from power source. By default infinite.

        Returns
        -------
        float
            The speed of the motor (rad /s)
        """
        speed = self._integrate(u, max_voltage)
        # Voltage and current are evaluated at the speed at the end of the step
        voltage, current = motor_electrical(
            speed, np.asarray([u], dtype=float), max_voltage, self._params_column())
        self.speed, self.voltage, self.current = speed[0], voltage[0], current[0]
        return self.speed


    def _params_column(self) -> np.ndarray:
        """This motor's column of the parameter matrix, shape (N_MOTOR_PARAMS, 1)."""
        return self._data.params[:, self._i:self._i + 1]


    def _integrate(self, u: float, max_voltage: float) -> np.ndarray:
        """
        Speed of this motor after one time step, without changing its state.
        Returns an array of shape (1,).
        """
        speed = self._data.speed[self._i:self._i + 1].copy()
        args = (np.asarray([u], dtype=float), float(max_voltage), self._params_column())
        return self.integrator(rhs_motor_only, speed, self.simulation.dt, args)



class Battery:
    """
    Models the battery of the Multirotor.

    Without a battery model (``BatteryParams.capacity`` is None), the battery is
    a fixed voltage limit for the motors. With one, it tracks the charge drawn,
    and its terminal voltage sags with the current drawn and as it discharges.
    Once all of its charge is drawn, it is empty: it has no voltage and supplies
    no current until it is reset.
    `Multirotor` integrates the battery together with the body and motors, and
    updates it after each step.
    """


    def __init__(self, params: BatteryParams, simulation: SimulationParams) -> None:
        self.params = deepcopy(params)
        self.simulation = simulation
        self.voltage: float = None
        "Terminal voltage (V)"
        self.current: float = 0.
        "Current (A), averaged over the last step"
        self.charge_drawn: float = 0.
        "Charge (C) drawn since the battery was full"
        self.cutoff: bool = False
        "Whether the terminal voltage fell below min_voltage, cutting off the motors"
        self.overload: bool = False
        "Whether the last step demanded more power than the battery can deliver"

        if self.has_model:
            # Parameters for the kernels in `multirotor.dynamics`
            self._params_vector = np.zeros(N_BATTERY_PARAMS)
            self._params_vector[BATTERY_CAPACITY] = 3600. * self.params.capacity
            self._params_vector[BATTERY_INTERNAL_RESISTANCE] = self.params.internal_resistance
            if self.params.ocv_soc is None:
                # Constant open-circuit voltage
                self._ocv_soc = np.asarray([0., 1.])
                self._ocv_voltage = np.full(2, float(self.params.max_voltage))
            else:
                self._ocv_soc = self.params.ocv_soc
                self._ocv_voltage = self.params.ocv_voltage
        self.reset()


    @property
    def has_model(self) -> bool:
        """Whether the battery's charge and voltage are modeled."""
        return self.params.capacity is not None


    @property
    def empty(self) -> bool:
        """Whether all of the battery's charge has been drawn."""
        return self.has_model and self.soc <= 0.


    @property
    def soc(self) -> float:
        """State of charge, from 0 (empty) to 1 (full)."""
        if not self.has_model:
            return 1.
        return 1. - self.charge_drawn / self._params_vector[BATTERY_CAPACITY]


    @property
    def motor_voltage_limit(self) -> float:
        """The maximum voltage the battery supplies to the motors."""
        return 0. if self.cutoff else self.voltage


    def open_circuit_voltage(self, charge_drawn: float=None) -> float:
        """
        Open-circuit voltage (V) after drawing ``charge_drawn`` coulombs from a
        full battery. By default, the charge drawn so far.
        """
        if not self.has_model:
            return self.params.max_voltage
        charge_drawn = self.charge_drawn if charge_drawn is None else charge_drawn
        soc = 1. - charge_drawn / self._params_vector[BATTERY_CAPACITY]
        return battery_ocv(soc, self._ocv_soc, self._ocv_voltage)


    def reset(self):
        """
        Return the battery to full charge.

        Returns
        -------
        float
            The terminal voltage.
        """
        self.charge_drawn = 0.
        self.current = 0.
        self.cutoff = False
        self.overload = False
        self.voltage = self.open_circuit_voltage()
        return self.voltage


    @property
    def state(self) -> float:
        return self.voltage


    def step(self, load: float=None):
        """
        Draw a constant current from the battery for one time step. Without a
        load, or without a battery model, this does nothing.

        Parameters
        ----------
        load : float, optional
            The current (A) drawn, by default None
        """
        if load is None or not self.has_model:
            return
        # An empty battery supplies no more charge
        charge = min(load * self.simulation.dt, self._charge_remaining())
        self.charge_drawn += charge
        self.current = charge / self.simulation.dt
        if self.empty:
            self.voltage = 0.
            return
        resistance = self._params_vector[BATTERY_INTERNAL_RESISTANCE]
        self.voltage = self.open_circuit_voltage() - resistance * self.current
        self._check_cutoff()


    def _charge_remaining(self) -> float:
        """Charge (C) left in the battery."""
        return self._params_vector[BATTERY_CAPACITY] - self.charge_drawn


    def _kernel_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The battery's arguments for `dynamics.rhs_motor_battery`."""
        return self._params_vector, self._ocv_soc, self._ocv_voltage


    def _end_step(
        self, charge: float, motor_speeds: np.ndarray, u: np.ndarray,
        motor_params: np.ndarray
    ) -> float:
        """
        Update the battery after `Multirotor` integrated one time step.

        Parameters
        ----------
        charge : float
            Charge (C) drawn during the step.
        motor_speeds : np.ndarray
            Motor speeds at the end of the step.
        u : np.ndarray
            Speed signals during the step.
        motor_params : np.ndarray
            The `MotorData.params` matrix.

        Returns
        -------
        float
            The motors' voltage limit at the end of the step.
        """
        # The integrator can overshoot the capacity by its tolerance
        charge = min(charge, self._charge_remaining())
        self.charge_drawn += charge
        self.current = charge / self.simulation.dt
        open_circuit_voltage = self.open_circuit_voltage()
        if self.cutoff or self.empty:
            self.voltage = open_circuit_voltage
            return 0.
        self.voltage, _, ok = battery_operating_point(
            motor_speeds, u, open_circuit_voltage,
            self._params_vector[BATTERY_INTERNAL_RESISTANCE], motor_params)
        self.overload = not ok
        motor_voltage_limit = self.voltage
        # The cutoff takes effect from the next step
        self._check_cutoff()
        return motor_voltage_limit


    def _check_cutoff(self) -> None:
        """Cut off the motors if the terminal voltage is below min_voltage."""
        if self.voltage < self.params.min_voltage:
            self.cutoff = True



class Multirotor:
    """
    The multirotor class models dynamics and control allocation of a vehicle.
    """

    def __init__(
        self, params: VehicleParams, simulation: SimulationParams,
        integrator: Integrator = odeint_integrator
    ) -> None:
        """
        Parameters
        ----------
        params : VehicleParams
            The vehicle parameters. These completely describe the vehicle's properties.
            The parameters are copied by this class, so any changes made to the params
            object is isolated from this instance.
        simulation : SimulationParams
            The simulation parameters.
        integrator : Integrator, optional
            The solver used to advance the state by one time step in
            `step_dynamics()` and `step_speeds()`. By default
            `odeint_integrator` (LSODA, rtol=atol=1e-4).
        """
        self.params: VehicleParams = deepcopy(params)
        self.simulation: SimulationParams = simulation
        self.integrator: Integrator = integrator
        self.dtype = self.params.inertia_matrix.dtype if simulation.dtype is None \
                     else simulation.dtype
        self.state: np.ndarray = None
        self.t: float = 0.
        self._dxdt = None

        # Parameters and state of all propellers (and motors) as arrays. Each
        # Propeller indexes into it.
        self._data = PropellerData(self.params.propellers)
        self.propellers: List[Propeller] = [
            Propeller._view(self._data, i, params, self.simulation, self.integrator)
            for i, params in enumerate(self.params.propellers)
        ]
        # Without motors, propeller speed equals the speed signal, which is
        # fixed over a time step. With thrust = k_thrust * speed**2, forces and
        # torques then don't depend on the vehicle's state during the step.
        self._forces_fixed_over_step = \
            self._data.motors is None and self._data.use_thrust_constant
        # The right-hand side used when forces are fixed over the step. Kept as
        # an attribute so it can be wrapped, e.g. for profiling.
        self._rhs_body = rhs_body
        if self.params.battery is not None:
            self.battery = Battery(self.params.battery, self.simulation)
        else:
            self.battery = Battery(BatteryParams(max_voltage=np.inf), self.simulation)
        if self.battery.has_model and self._data.motors is None:
            raise ValueError('A battery model needs motors to draw current.')
        if self.battery.has_model and not self._data.use_thrust_constant:
            raise ValueError('A battery model is not supported with blade-element thrust.')
        # With motors, motors and body are integrated together. The integrated
        # state is the 12 body states followed by the motor speeds and, with a
        # battery model, the charge drawn during the step.
        if self._data.motors is None:
            self._rhs = None
        elif self.battery.has_model:
            self._rhs = rhs_motor_battery
        elif self._data.use_thrust_constant:
            self._rhs = rhs_motor
        else:
            self._rhs = rhs_motor_blade_element
        # Copied before reset() casts the parameters to the vehicle's dtype
        self._kernel_params = KernelParams(
            g=float(self.simulation.g),
            mass=float(self.params.mass),
            inertia_matrix=self.params.inertia_matrix.astype(float),
            inertia_matrix_inverse=self.params.inertia_matrix_inverse.astype(float),
            prop_params=self._data.params,
            motor_params=None if self._data.motors is None else self._data.motors.params,
            clockwise=np.asarray(self.params.clockwise, dtype=float),
            positions=self.params.propeller_vectors.astype(float),
        )
        self.alloc, self.alloc_inverse = control_allocation_matrix(self.params)
        self.reset()


    def reset(self) -> np.ndarray:
        """
        Reset the state of the vehicle. This includes resetting each propeller
        and re-calculating inertia and allocation matrices.

        Can simulate dynamics with propellers with/out motors.

        Returns
        -------
        np.ndarray
            The state of the vehicle.
        """
        self.t = 0.
        self._data.reset()
        self.battery.reset()

        self.alloc = self.alloc.astype(self.dtype)
        self.alloc_inverse = self.alloc_inverse.astype(self.dtype)
        self.params.propeller_vectors = self.params.propeller_vectors.astype(self.dtype)
        self.params.inertia_matrix = self.params.inertia_matrix.astype(self.dtype)
        self.params.inertia_matrix_inverse = self.params.inertia_matrix_inverse.astype(self.dtype)
        self._clockwise = np.asarray(self.params.clockwise)
        self.state = np.zeros(12, dtype=self.dtype)
        self._dxdt = np.zeros_like(self.state)
        return self.state


    @property
    def position(self):
        """Position in the inertial frame."""
        return self.state[0:3]


    @property
    def velocity(self) -> np.ndarray:
        """Body-frame velocity"""
        return self.state[3:6]


    @property
    def inertial_velocity(self) -> np.ndarray:
        """Velocity in the intertial frame."""
        dcm = direction_cosine_matrix(*self.orientation)
        v_inertial = body_to_inertial(self.velocity, dcm)
        return v_inertial


    @property
    def acceleration(self) -> np.ndarray:
        """Body-frame acceleration"""
        return self._dxdt[3:6]


    @property
    def inertial_acceleration(self) -> np.ndarray:
        """Acceleration in the intertial frame."""
        dcm = direction_cosine_matrix(*self.orientation)
        a_inertial = body_to_inertial(self.acceleration, dcm)
        return a_inertial



    @property
    def orientation(self) -> np.ndarray:
        """Euler rotations (roll, pitch, yaw)"""
        return self.state[6:9]


    @property
    def angular_rate(self) -> np.ndarray:
        """Angular rate of body frame axes (not same as rate of roll, pitch, yaw)"""
        return self.state[9:12]


    @property
    def euler_rate(self) -> np.ndarray:
        """Euler rate of vehicle d(roll, pitch, yaw)/dt"""
        return angular_to_euler_rate(self.angular_rate, self.orientation)


    @property
    def weight(self) -> float:
        return self.simulation.g * self.params.mass


    @property
    def motor_currents(self) -> np.ndarray:
        """
        Current (A) drawn by each motor, on the motor side of its speed
        controller. None if the vehicle has no motors.
        """
        if self._data.motors is None:
            return None
        return self._data.motors.current.copy()


    @property
    def battery_current(self) -> float:
        """
        Current (A) drawn from the battery, on the battery side of the speed
        controllers, which are modeled as lossless.

        With a battery model, the average over the last step. Without one, the
        motors' power divided by ``BatteryParams.max_voltage``. NaN if the
        vehicle has no motors, or has no battery model and an infinite
        ``max_voltage``.
        """
        if self.battery.has_model:
            return self.battery.current
        motors = self._data.motors
        max_voltage = self.battery.params.max_voltage
        if motors is None or not np.isfinite(max_voltage):
            return np.nan
        return np.sum(motors.current * motors.voltage) / max_voltage


    @property
    def speeds(self) -> np.ndarray:
        return np.asarray(self._data.speed, self.dtype)
    @speeds.setter
    def speeds(self, speeds: np.ndarray):
        self._step_propellers(speeds)


    def _motor_speeds_after_step(self, u: np.ndarray) -> np.ndarray:
        """
        Speeds of all motors after one time step under speed signals ``u``,
        integrating the motors on their own, without changing their state.
        """
        motors = self._data.motors
        args = (
            np.asarray(u, dtype=float), float(self.battery.motor_voltage_limit),
            motors.params
        )
        return self.integrator(
            rhs_motor_only, motors.speed.copy(), self.simulation.dt, args)


    def _apply_speeds(self, u: np.ndarray) -> np.ndarray:
        """
        Propeller speeds after one time step under speed signals ``u``, without
        changing the state of the propellers. Without motors, speed signals take
        effect instantaneously.
        """
        if self._data.motors is None:
            return np.asarray(u)
        return self._motor_speeds_after_step(u)


    def _set_motor_state(
        self, speed: np.ndarray, u: np.ndarray, max_voltage: float
    ) -> None:
        """
        Set motor speeds, and the voltages and currents at those speeds under
        speed signals ``u`` and voltage limit ``max_voltage``.
        """
        motors = self._data.motors
        motors.speed[:] = speed
        motors.voltage[:], motors.current[:] = motor_electrical(
            motors.speed, np.asarray(u, dtype=float), float(max_voltage),
            motors.params)


    def _step_propellers(self, u: np.ndarray) -> None:
        """
        Advance all propellers (and motors) by one time step under speed signals
        ``u``, on their own.
        """
        if self._data.motors is None:
            self._data.speed[:] = u
        else:
            self._set_motor_state(
                self._motor_speeds_after_step(u), u, self.battery.motor_voltage_limit)


    def _thrusts(self, speeds: np.ndarray, airstream_velocities: np.ndarray) -> np.ndarray:
        """
        Thrust (N) of each propeller.

        Parameters
        ----------
        speeds : np.ndarray
            Propeller speeds (rad/s), shape (n,).
        airstream_velocities : np.ndarray
            Airstream velocity at each propeller, shape (3, n). Only used by the
            blade-element thrust model.

        Returns
        -------
        np.ndarray
            Thrust of each propeller, shape (n,).
        """
        if self._data.use_thrust_constant:
            return self._data.params[PROP_K_THRUST] * speeds**2
        return thrusts_physics(
            speeds, airstream_velocities, self._data.params, self.simulation.rho)


    def _body_kernel_params(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Gravity, mass, inertia matrix and its inverse: the parameters `rhs_body` takes."""
        p = self._kernel_params
        return p.g, p.mass, p.inertia_matrix, p.inertia_matrix_inverse


    def get_forces_torques(self, speeds: np.ndarray, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the forces and torques acting on the vehicle's center of gravity
        given its current state and speed of propellers.

        Parameters
        ----------
        speeds : np.ndarray
            Propeller speeds (rad/s)
        state : np.ndarray
            State of the vehicle (position, velocity, orientation, angular rate)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The forces and torques acting on the body.
        """
        linear_vel_body = state[:3]
        angular_vel_body = state[3:6]
        airstream_velocity_inertial = rotating_frame_derivative(
            self.params.propeller_vectors,
            linear_vel_body,
            angular_vel_body)

        speeds = self._apply_speeds(speeds)
        thrusts = self._thrusts(speeds, airstream_velocity_inertial).astype(self.dtype)
        drag_torques = self._data.params[PROP_K_DRAG] * speeds**2
        torques = body_torques(
            thrusts, drag_torques, self._clockwise, self.params.propeller_vectors)
        forces = np.zeros(3, dtype=self.dtype)
        forces[2] = thrusts.sum()
        return forces, torques.astype(self.dtype)


    def dxdt_dynamics(self, t: float, x: np.ndarray, u: np.ndarray, params=None):
        """
        Calculate the rate of change of state given the dynamics (forces, torques)
        acting on the system.

        Parameters
        ----------
        t : float
            Time. Currently this function is time invariant.
        x : np.ndarray
            State of the vehicle.
        u : np.ndarray
            A 6-vector of forces and torques.

        Returns
        -------
        np.ndarray
            The rate of change of state.
        """
        # This method must not have any side-effects. It should not change the
        # state of the vehicle. This method is called multiple times from the 
        # same state by the odeint() function, and the results should be consistent.
        # Do not need to get forces/torques on body, since the action array
        # already is a 6d vector of forces/torques.
        # forces, torques = self.get_forces_torques(u, x)
        u = np.asarray(u, dtype=float)
        return rhs_body(
            t, np.asarray(x, dtype=float), u[:3], u[3:], *self._body_kernel_params())


    def dxdt_speeds(
        self, t: float, x: np.ndarray, u: np.ndarray,
        disturb_forces: np.ndarray=0., disturb_torques: np.ndarray=0., params=None
    ):
        """
        Calculate the rate of change of state given the propeller speeds on the
        system (rad/s).

        Only the 12 body states are modeled here. With motors, each motor's
        speed is held at `Propeller.apply_speed` of its signal: the speed it
        would reach after one time step. `step_speeds` instead lets motor speeds
        evolve during the step, so for vehicles with motors this derivative
        (and `nonlinear_speeds_controls_system`, which uses it) leaves out the
        motors' dynamics.

        Parameters
        ----------
        t : float
            Time. Currently this function is time invariant.
        x : np.ndarray
            State of the vehicle.
        u : np.ndarray
            A p-vector of propeller speeds (rad/s), where p=number of propellers.
        disturb_forces : np.ndarray, optional
            Disturbinng x,y,z forces in the vehicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.

        Returns
        -------
        np.ndarray
            The rate of change of state.
        """
        # This method must not have any side-effects. It should not change the
        # state of the vehicle. This method is called multiple times from the 
        # same state by the odeint() function, and the results should be consistent.
        forces, torques = self.get_forces_torques(u, x)
        return rhs_body(
            t, np.asarray(x, dtype=float),
            forces.astype(float) + disturb_forces,
            torques.astype(float) + disturb_torques,
            *self._body_kernel_params())


    def step_dynamics(self, u: np.ndarray) -> np.ndarray:
        """
        Given the 6-vector of x,y,z-forces and roll,pitch,yaw-torques, calculate
        the next state of the vehicle.

        Parameters
        ----------
        u : np.ndarray
            The 6-vector, where the first 3 elements are forces (N) and the next 3
            elements are the torques (Nm)

        Returns
        -------
        np.ndarray
            The new state of the vehicle.
        """
        self.t += self.simulation.dt
        u = np.asarray(u, dtype=float)
        args = (u[:3], u[3:]) + self._body_kernel_params()
        x0 = self.state.astype(float)
        self._dxdt = self._rhs_body(self.t, x0, *args).astype(self.dtype)
        self.state = self.integrator(
            self._rhs_body, x0, self.simulation.dt, args
        ).astype(self.dtype)
        # TODO: inverse solve for speed = forces to set propeller speeds
        return self.state


    def step_speeds(
        self, u: np.ndarray, disturb_forces: np.ndarray=0.,
        disturb_torques: np.ndarray=0.
    ) -> np.ndarray:
        """
        Given the n-vector of propeller speed signals, calculate
        the next state of the vehicle. Where n is number of propellers.

        With motors, the body and the motors are integrated together, so the
        forces during the step follow the motor speeds as they change.

        Parameters
        ----------
        u : np.ndarray
            The speed signals to be sent to each propeller's step() method. Can
            be the actual speed (rad/s) or the voltage signal (V) if a motor
            is used and MotorParams.speed_voltage_scaling constant is set.
        disturb_forces : np.ndarray, optional
            Disturbinng x,y,z forces in the vehicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.

        Returns
        -------
        np.ndarray
            The new state of the vehicle.
        """
        self.t += self.simulation.dt
        if self._data.motors is not None:
            self._step_with_motors(u, disturb_forces, disturb_torques)
        elif self._forces_fixed_over_step:
            self._step_with_fixed_forces(u, disturb_forces, disturb_torques)
        else:
            self._step_with_varying_forces(u, disturb_forces, disturb_torques)
        return self.state


    def _step_with_fixed_forces(
        self, u: np.ndarray, disturb_forces: np.ndarray,
        disturb_torques: np.ndarray
    ) -> None:
        """
        Advance a vehicle without motors and with quadratic thrust by one time
        step. Forces and torques depend only on the speed signals, which are
        fixed over the step, so they are calculated once instead of at every
        evaluation of the derivative.

        Parameters
        ----------
        u : np.ndarray
            Speed signals (rad/s).
        disturb_forces : np.ndarray
            Disturbing x,y,z forces in the vehicle's local frame.
        disturb_torques : np.ndarray
            Disturbing x,y,z torques in the vehicle's local frame.
        """
        forces, torques = self.get_forces_torques(u, self.state)
        forces = forces.astype(float) + disturb_forces
        torques = torques.astype(float) + disturb_torques
        args = (forces, torques) + self._body_kernel_params()
        x0 = self.state.astype(float)
        self._dxdt = self._rhs_body(self.t, x0, *args).astype(self.dtype)
        self.state = self.integrator(
            self._rhs_body, x0, self.simulation.dt, args
        ).astype(self.dtype)
        self._step_propellers(u)


    def _step_with_varying_forces(
        self, u: np.ndarray, disturb_forces: np.ndarray,
        disturb_torques: np.ndarray
    ) -> None:
        """
        Advance a vehicle without motors and with blade-element thrust by one
        time step. Thrust depends on the airstream, so forces are recalculated
        at every evaluation of the derivative.

        Parameters
        ----------
        u : np.ndarray
            Speed signals (rad/s).
        disturb_forces : np.ndarray
            Disturbing x,y,z forces in the vehicle's local frame.
        disturb_torques : np.ndarray
            Disturbing x,y,z torques in the vehicle's local frame.
        """
        x0 = self.state.astype(float)
        self._dxdt = self.dxdt_speeds(
            t=self.t, x=x0, u=u,
            disturb_forces=disturb_forces, disturb_torques=disturb_torques
        ).astype(self.dtype)
        self.state = self.integrator(
            self.dxdt_speeds, x0, self.simulation.dt,
            (u, disturb_forces, disturb_torques)
        ).astype(self.dtype)
        self._step_propellers(u)


    def _step_with_motors(
        self, u: np.ndarray, disturb_forces: np.ndarray,
        disturb_torques: np.ndarray
    ) -> None:
        """
        Advance the body and the motors by one time step, integrating them
        together.

        Parameters
        ----------
        u : np.ndarray
            Speed signals.
        disturb_forces : np.ndarray
            Disturbing x,y,z forces in the vehicle's local frame.
        disturb_torques : np.ndarray
            Disturbing x,y,z torques in the vehicle's local frame.
        """
        motors = self._data.motors
        battery = self.battery
        u = np.asarray(u, dtype=float)
        disturbance = np.zeros(6)
        disturbance[:3] += disturb_forces
        disturbance[3:] += disturb_torques
        x0 = np.concatenate((self.state.astype(float), motors.speed))
        if battery.has_model:
            # One more state: the charge drawn during this step
            x0 = np.append(x0, 0.)
            args = (u, battery.cutoff, disturbance) + self._kernel_params \
                   + battery._kernel_params() + (battery.charge_drawn,)
        elif self._data.use_thrust_constant:
            args = (u, float(battery.motor_voltage_limit), disturbance) \
                   + self._kernel_params
        else:
            # Blade-element thrust also needs the air density
            args = (u, float(battery.motor_voltage_limit), disturbance) \
                   + self._kernel_params + (float(self.simulation.rho),)
        self._dxdt = self._rhs(self.t, x0, *args)[:N_BODY_STATES].astype(self.dtype)
        x = self.integrator(self._rhs, x0, self.simulation.dt, args)

        self.state = x[:N_BODY_STATES].astype(self.dtype)
        motor_speeds = x[N_BODY_STATES:N_BODY_STATES + len(motors)]
        if battery.has_model:
            motor_voltage_limit = battery._end_step(x[-1], motor_speeds, u, motors.params)
        else:
            motor_voltage_limit = battery.motor_voltage_limit
        self._set_motor_state(motor_speeds, u, motor_voltage_limit)


    def allocate_control(self, thrust: float, torques: np.ndarray) -> np.ndarray:
        """
        Allocate control to propellers by converting prescribed forces and torqes
        into propeller speeds. Uses the control allocation matrix.

        Parameters
        ----------
        thrust : float
            The thrust in the body z-direction.
        torques : np.ndarray
            The roll, pitch, yaw torques required about (x, y, z) axes.

        Returns
        -------
        np.ndarray
            The prescribed propeller speeds (rad /s)
        """
        # TODO: njit it? np.linalg.lstsq can be compiled
        vec = np.asarray([thrust, *torques], self.dtype)
        # return np.sqrt(np.linalg.lstsq(self.alloc, vec, rcond=None)[0])
        return np.sqrt(
            np.clip(self.alloc_inverse @ vec, a_min=0., a_max=None)
        )


    def nonlinear_dynamics_controls_system(
        self, linearize: bool=False, perturbation: float=1e-1,
        about_state: np.ndarray=0, about_action: np.ndarray=0
    ):
        """
        Create a system representation using the python controls library. The
        system takes net forces and torques as input.

        Parameters
        ----------
        linearize : bool, optional
            Whether to linearize the system about a state/action, by default False
        perturbation : float, optional
            The change in time to use to calculate dx/dt, by default 1e-1
        about_state : np.ndarray, optional
            The state about which to linearize, by default 0
        about_action : np.ndarray, optional
            The action about which to linearize, by default 0

        Returns
        -------
        Union[control.LinearIOSystem, control.NonlinearIOSystem]
            The system object
        """
        import control
        sys = control.NonlinearIOSystem(
            updfcn=self.dxdt_dynamics,
            inputs=['fx','fy','fz','tx','ty','tz'],
            states=['x','y','z',
                    'vx','vy','vz',
                    'roll','pitch','yaw',
                    'xrate', 'yrate', 'zrate']
        )
        if linearize:
            x0 = (np.zeros(12) if about_state==0 else about_state)
            u0 = (np.zeros(6) if about_action==0 else about_action)
            sys = sys.linearize(eps=perturbation, x0=x0, u0=u0)
        return sys
    

    def nonlinear_speeds_controls_system(
        self, linearize: bool=False, perturbation: float=1e-1,
        about_state: np.ndarray=0, about_action: np.ndarray=0
    ):
        """
        Create a system representation using the python controls library. The
        system takes propeller speed signals as input.

        The system has the 12 body states only. For vehicles with motors it
        leaves out the motors' dynamics; see `dxdt_speeds`.

        Parameters
        ----------
        linearize : bool, optional
            Whether to linearize the system about a state/action, by default False
        perturbation : float, optional
            The change in time to use to calculate dx/dt, by default 1e-1
        about_state : np.ndarray, optional
            The state about which to linearize, by default 0
        about_action : np.ndarray, optional
            The action about which to linearize, by default 0

        Returns
        -------
        Union[control.LinearIOSystem, control.NonlinearIOSystem]
            The system object
        """
        import control
        sys = control.NonlinearIOSystem(
            updfcn=self.dxdt_speeds,
            inputs=['w%d' % i for i in range(len(self.propellers))],
            states=['x','y','z',
                    'vx','vy','vz',
                    'roll','pitch','yaw',
                    'xrate', 'yrate', 'zrate']
        )
        if linearize:
            x0 = (np.zeros(12) if about_state==0 else about_state)
            u0 = (np.zeros(len(self.propellers)) if about_action==0 else about_action)
            sys = sys.linearize(eps=perturbation, x0=x0, u0=u0)
        return sys
