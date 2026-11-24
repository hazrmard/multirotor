from typing import Any, Callable, List, Optional, Tuple, Union
from copy import deepcopy

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._compile import step_function
from .vehicle import MotorParams, PropellerParams, SimulationParams, VehicleParams, BatteryParams
from .coords import body_to_inertial, direction_cosine_matrix, rotating_frame_derivative, angular_to_euler_rate
from .physics import (
    thrust_blade_element_model, thrusts_blade_element_model, battery_end_step,
    forces_torques, motor_acc, motor_electrical,
    PROP_K_THRUST, PROP_K_DRAG, BATTERY_CAPACITY, BATTERY_INTERNAL_RESISTANCE,
    BATTERY_MIN_VOLTAGE, N_BATTERY_PARAMS
)
from .builder import battery_arrays, build_vehicle, motor_matrix, propeller_matrix
from .dynamics import (
    rhs_body, rhs_body_with_motors, rhs_body_with_motors_and_battery, rhs_motor_only,
    N_BODY_STATES
)
from .helpers import control_allocation_matrix
from .integrator import DEFAULT_SUBSTEPS, integrate
from .tuples import BatteryVehicle, BodyVehicle, MotorsVehicle, VehicleState, kind_of


# ---------------------------------------------------------------------------
# The compiled steps. They are compiled once, here, and take the vehicle's parameters as
# arguments, so that all vehicles share them. A call from Python costs more than the step
# itself, and most of that cost is per array passed and returned. So each step takes what
# changes from call to call as one array, the vehicle's parameters as JAX arrays made once,
# and returns its results as one array.

_rhs_body = step_function(rhs_body)
_motor_acc = step_function(motor_acc)


@step_function
def _motor_electrical(
    speed: jax.Array, u: jax.Array, max_voltage: float, motor_params: jax.Array
) -> jax.Array:
    """Voltage and current of each motor, as the two rows of one array, shape (2, n)."""
    return jnp.stack(motor_electrical(speed, u, max_voltage, motor_params))


@partial(step_function, static_argnames=('substeps',))
def _step_motor_only(
    inputs: jax.Array, motor_params: jax.Array, dt: float, substeps: int
) -> jax.Array:
    """
    Advance motors on their own by one time step.

    ``inputs``: The motors' speeds (n), their speed signals (n) and the voltage limit (1).
    Returns: Their speed, and their voltage and current at that speed, as the three rows of
      one array, shape (3, n).
    """
    n = motor_params.shape[1]
    speed, u, max_voltage = inputs[:n], inputs[n:2 * n], inputs[2 * n]
    speed, _ = integrate(
        lambda speed: rhs_motor_only(speed, u, max_voltage, motor_params), speed, dt, substeps)
    voltage, current = motor_electrical(speed, u, max_voltage, motor_params)
    return jnp.stack((speed, voltage, current))


@partial(step_function, static_argnames=('substeps',))
def _step_forces(inputs: jax.Array, vehicle: BodyVehicle, dt: float, substeps: int) -> jax.Array:
    """
    Advance the body by one time step under fixed forces and torques.

    ``inputs``: The 12 body states, the forces (3) and the torques (3).
    Returns: The 12 body states, then their rate of change at the start of the step.
    """
    x, forces, torques = inputs[:12], inputs[12:15], inputs[15:18]
    body, dxdt = integrate(lambda x: rhs_body(x, forces, torques, vehicle), x, dt, substeps)
    return jnp.concatenate((body, dxdt))


@partial(step_function, static_argnames=('substeps',))
def _step_body(inputs: jax.Array, vehicle: BodyVehicle, dt: float, substeps: int) -> jax.Array:
    """
    Advance a vehicle without motors and with quadratic thrust by one time step: the forces
    follow from the speed signals, which are fixed over the step.

    ``inputs``: The 12 body states, the speed signals (n), and the force and torque
      disturbing the body (6), used in place of the vehicle's.
    Returns: As `_step_forces`.
    """
    n = vehicle.prop_params.shape[1]
    x, u, disturbance = inputs[:12], inputs[12:12 + n], inputs[12 + n:]
    forces, torques = forces_torques(
        u, vehicle.prop_params, vehicle.clockwise, vehicle.positions)
    forces, torques = forces + disturbance[:3], torques + disturbance[3:]
    body, dxdt = integrate(lambda x: rhs_body(x, forces, torques, vehicle), x, dt, substeps)
    return jnp.concatenate((body, dxdt))


@partial(step_function, static_argnames=('substeps',))
def _step_motors(inputs: jax.Array, vehicle: MotorsVehicle, dt: float, substeps: int) -> jax.Array:
    """
    Advance the body and the motors together by one time step.

    ``inputs``: The body states and motor speeds (12 + n), the speed signals (n), the force
      and torque disturbing the body (6) and the motors' voltage limit (1); the last two are
      used in place of the vehicle's.
    Returns: The body states and motor speeds (12 + n), the rate of change of the body states
      at the start (12), and the motors' voltages (n) and currents (n) at the end.
    """
    n = vehicle.prop_params.shape[1]
    x, u = inputs[:12 + n], inputs[12 + n:12 + 2 * n]
    max_voltage = inputs[18 + 2 * n]
    vehicle = vehicle._replace(disturbance=inputs[12 + 2 * n:18 + 2 * n])
    x_new, dxdt = integrate(
        lambda x: rhs_body_with_motors(x, u, max_voltage, vehicle), x, dt, substeps)
    voltage, current = motor_electrical(
        x_new[N_BODY_STATES:], u, max_voltage, vehicle.motor_params)
    return jnp.concatenate((x_new, dxdt[:N_BODY_STATES], voltage, current))


@partial(step_function, static_argnames=('substeps',))
def _step_motors_battery(
    inputs: jax.Array, vehicle: BatteryVehicle, dt: float, substeps: int
) -> jax.Array:
    """
    Advance the body, the motors and the battery together by one time step.

    ``inputs``: The body states and motor speeds (12 + n), the speed signals (n), the force
      and torque disturbing the body (6), used in place of the vehicle's, and the battery's
      charge drawn (C) and whether it is cut off and overloaded, as floats (3).
    Returns: The body states and motor speeds (12 + n), the rate of change of the body states
      at the start (12), the six fields of the battery's `multirotor.physics.BatteryEnd`, and
      the motors' voltages (n) and currents (n) at the end.
    """
    n = vehicle.prop_params.shape[1]
    width = 12 + n
    x, u = inputs[:width], inputs[width:width + n]
    vehicle = vehicle._replace(disturbance=inputs[width + n:width + n + 6])
    battery = inputs[width + n + 6:]
    charge_drawn, cutoff, overload = battery[0], battery[1] > .5, battery[2] > .5
    # One more integrated state: the charge drawn during this step
    x_new, dxdt = integrate(
        lambda x: rhs_body_with_motors_and_battery(x, u, cutoff, charge_drawn, vehicle),
        jnp.concatenate((x, jnp.zeros(1))), dt, substeps)
    speeds = x_new[N_BODY_STATES:width]
    end = battery_end_step(
        x_new[width], charge_drawn, cutoff, overload, speeds, u, vehicle.motor_params,
        vehicle.battery_params, vehicle.ocv_soc, vehicle.ocv_voltage, dt)
    voltage, current = motor_electrical(speeds, u, end.voltage_limit, vehicle.motor_params)
    return jnp.concatenate((
        x_new[:width], dxdt[:N_BODY_STATES], jnp.stack([1. * field for field in end]),
        voltage, current))


def _rk4(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dt: float, substeps: int) -> np.ndarray:
    """
    Advance ``x`` by ``dt`` in ``substeps`` RK4 steps of ``dx/dt = f(x)``, in numpy. For the
    vehicles with blade-element thrust, whose right-hand side is not compiled.
    """
    h = dt / substeps
    for _ in range(substeps):
        k1 = f(x)
        k2 = f(x + h / 2. * k1)
        k3 = f(x + h / 2. * k2)
        k4 = f(x + h * k3)
        x = x + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return x


class Propeller:
    """
    Models the thrust and aerodynamics of the propeller blades spinning at a
    certain rate.

    The propeller's speed is element ``i`` of an array. A standalone propeller owns an
    array of size 1. Propellers of a `Multirotor` index into the vehicle's array, each at
    its own index, so the vehicle and its propellers share the state. (Sharing the array,
    not a view of it, keeps the sharing in copies of the vehicle.)
    """

    def __init__(
        self, params: PropellerParams, simulation: SimulationParams,
        substeps: int = DEFAULT_SUBSTEPS['motors']
    ) -> None:
        """
        Parameters
        ----------
        params : PropellerParams
            The propeller parameters. They are copied.
        simulation : SimulationParams
            The simulation parameters.
        substeps : int, optional
            RK4 steps per time step when advancing the motor, if any.
        """
        params = deepcopy(params)
        motor = None if params.motor is None else Motor(params.motor, simulation, substeps)
        self._attach(
            params, simulation, motor, np.zeros(1) if motor is None else motor._speed, 0)


    @classmethod
    def _view(
        cls, params: PropellerParams, simulation: SimulationParams, motor: Optional['Motor'],
        speed: np.ndarray, i: int
    ) -> 'Propeller':
        """Create a propeller whose speed is ``speed[i]``."""
        propeller = cls.__new__(cls)
        propeller._attach(params, simulation, motor, speed, i)
        return propeller


    def _attach(
        self, params: PropellerParams, simulation: SimulationParams, motor: Optional['Motor'],
        speed: np.ndarray, i: int
    ) -> None:
        """Set the propeller's parameters, motor and the array that holds its speed."""
        self.params: PropellerParams = params
        self.simulation: SimulationParams = simulation
        self._speed: np.ndarray = speed
        self._i: int = i

        # Thrust constant uses a simpler quardatic relationship to calculate
        # propeller thrust.
        if self.params.use_thrust_constant:
            self.thrust: Callable = self._thrust_constant
        else:
            self.thrust: Callable = self._thrust_blade_element_model
        # If not motor is provided, then speed signals take effect instantaneously
        self.motor: Optional[Motor] = motor


    @property
    def speed(self) -> float:
        "Radians per second"
        return float(self._speed[self._i])
    @speed.setter
    def speed(self, speed: float) -> None:
        self._speed[self._i] = speed


    def reset(self) -> None:
        "Stop the propeller, and its motor."
        self.speed = 0.
        if self.motor is not None:
            self.motor.reset()


    def apply_speed(self, u: float, **kwargs: Any) -> float:
        """
        Calculate the speed of the propeller after the speed signal takes effect.
        This method is *pure*: it leaves the state of the propeller unchanged.
        The multirotor's dxdt_* methods use it to calculate derivatives.

        Parameters
        ----------
        u : float
            Radians per second speed command
        **kwargs : Any
            Passed to `Motor.apply_speed`, if there is a motor.

        Returns
        -------
        float
            The actual speed
        """
        if self.motor is not None:
            return self.motor.apply_speed(u, **kwargs)
        return u


    def step(self, u: float, **kwargs: Any) -> float:
        """
        Step through the speed command. This method changes the state of the
        propeller.

        Parameters
        ----------
        u : float
            Speed command in radians per second.
        **kwargs : Any
            Passed to `Motor.step`, if there is a motor.

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


    def _thrust_constant(
        self, speed: float, airstream_velocity: np.ndarray = np.zeros(3)
    ) -> float:
        """
        Thrust that is ``k_thrust * speed**2``.

        ``speed``: The propeller's speed (rad/s).
        Returns: The thrust (N).
        """
        return self.params.k_thrust * speed**2


    def _thrust_blade_element_model(
        self, speed: float, airstream_velocity: np.ndarray = np.zeros(3)
    ) -> float:
        """
        Thrust from the blade-element model.

        ``speed``: The propeller's speed (rad/s).
        ``airstream_velocity``: The airstream velocity at the propeller, shape (3,).
        Returns: The thrust (N).
        """
        p = self.params
        return thrust_blade_element_model(
            speed, airstream_velocity,
            p.R, p.A, self.simulation.rho, p.a, p.b, p.c, p.eta, p.theta0, p.theta1
        )


    @property
    def state(self) -> float:
        "The propeller's state: its speed (rad/s)."
        return self.speed


    @property
    def rpm(self) -> float:
        "The propeller's speed in revolutions per minute."
        return self.speed * 60. / (2. * np.pi)



class Motor:
    """
    Models the electronic and mechanical characterists of the rotation of the
    propeller shaft.

    The motor's speed, voltage and current are element ``i`` of arrays, and its parameters
    are column ``i`` of a matrix. A standalone motor owns arrays of size 1. Motors of a
    `Multirotor` index into the vehicle's arrays, each at its own index.
    """

    def __init__(
        self, params: MotorParams, simulation: SimulationParams,
        substeps: int = DEFAULT_SUBSTEPS['motors']
    ) -> None:
        """
        Parameters
        ----------
        params : MotorParams
            The motor parameters. They are copied.
        simulation : SimulationParams
            The simulation parameters.
        substeps : int, optional
            RK4 steps per time step when advancing the motor speed.
        """
        params = deepcopy(params)
        self._attach(
            params, simulation, substeps, motor_matrix([params], np),
            np.zeros(1), np.zeros(1), np.zeros(1), 0)


    @classmethod
    def _view(
        cls, params: MotorParams, simulation: SimulationParams, substeps: int,
        matrix: np.ndarray, speed: np.ndarray, voltage: np.ndarray, current: np.ndarray,
        i: int
    ) -> 'Motor':
        """
        Create the motor at index ``i`` of the motors' parameter matrix, shape
        (N_MOTOR_PARAMS, n), and of the arrays of their speed, voltage and current.
        """
        motor = cls.__new__(cls)
        motor._attach(params, simulation, substeps, matrix, speed, voltage, current, i)
        return motor


    def _attach(
        self, params: MotorParams, simulation: SimulationParams, substeps: int,
        matrix: np.ndarray, speed: np.ndarray, voltage: np.ndarray, current: np.ndarray,
        i: int
    ) -> None:
        """Set the motor's parameters, and the arrays that hold its parameters and state."""
        self.params: MotorParams = params
        self.simulation: SimulationParams = simulation
        self.substeps: int = substeps
        self._matrix: np.ndarray = matrix
        self._speed: np.ndarray = speed
        self._voltage: np.ndarray = voltage
        self._current: np.ndarray = current
        self._i: int = i


    @property
    def speed(self) -> float:
        "Radians per second"
        return float(self._speed[self._i])
    @speed.setter
    def speed(self, speed: float) -> None:
        self._speed[self._i] = speed


    @property
    def voltage(self) -> float:
        "The voltage (V) across the motor at the last step."
        return float(self._voltage[self._i])
    @voltage.setter
    def voltage(self, voltage: float) -> None:
        self._voltage[self._i] = voltage


    @property
    def current(self) -> float:
        "The current (A) through the motor at the last step."
        return float(self._current[self._i])
    @current.setter
    def current(self, current: float) -> None:
        self._current[self._i] = current


    def reset(self) -> float:
        """
        Stop the motor, with no voltage and no current.

        Returns
        -------
        float
            The motor's speed, 0.
        """
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
        Apply a voltage speed signal to the motor. This method is pure: it leaves
        the state of the motor unchanged.

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
        return float(self._integrate(u, max_voltage)[0, 0])


    def step(self, u: float, max_voltage: float=np.inf) -> float:
        """
        Apply a voltage speed signal to the motor. This method updates the state
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
        # Voltage and current are evaluated at the speed at the end of the step
        self.speed, self.voltage, self.current = self._integrate(u, max_voltage)[:, 0]
        return self.speed


    def _params_column(self) -> np.ndarray:
        """This motor's column of the parameter matrix, shape (N_MOTOR_PARAMS, 1)."""
        return self._matrix[:, self._i:self._i + 1]


    def _integrate(self, u: float, max_voltage: float) -> np.ndarray:
        """
        This motor after one time step, without changing its state.

        ``max_voltage``: The voltage limit (V).
        Returns: The speed (rad/s), and the voltage (V) and current (A) at that speed, shape
          (3, 1).
        """
        inputs = np.asarray((self._speed[self._i], u, max_voltage), float)
        return np.asarray(_step_motor_only(
            inputs, self._params_column(), float(self.simulation.dt), substeps=self.substeps))


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
        """
        Parameters
        ----------
        params : BatteryParams
            The battery's parameters. A battery without a model has a constant voltage.
        simulation : SimulationParams
            The simulation parameters.
        """
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
            # Parameters for the equations in `multirotor.physics`
            self._params_vector, self._ocv_soc, self._ocv_voltage = battery_arrays(
                self.params, np)
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


    def open_circuit_voltage(self, charge_drawn: Optional[float] = None) -> float:
        """
        Open-circuit voltage (V) after drawing ``charge_drawn`` coulombs from a
        full battery. By default, the charge drawn so far.

        Parameters
        ----------
        charge_drawn : float, optional
            The charge (C) drawn from a full battery.

        Returns
        -------
        float
            The open-circuit voltage (V).
        """
        if not self.has_model:
            return self.params.max_voltage
        charge_drawn = self.charge_drawn if charge_drawn is None else charge_drawn
        soc = 1. - charge_drawn / self._params_vector[BATTERY_CAPACITY]
        # As `multirotor.physics.battery_ocv`: an empty battery has no voltage
        return 0. if soc <= 0. else float(np.interp(soc, self._ocv_soc, self._ocv_voltage))


    def reset(self) -> float:
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
        "The battery's state: its terminal voltage (V)."
        return self.voltage


    def step(self, load: Optional[float] = None) -> None:
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


    def _end_step(self, end: np.ndarray) -> float:
        """
        Update the battery after `Multirotor` integrated one time step.

        ``end``: The six fields of `multirotor.physics.BatteryEnd`, as floats.
        Returns: The motors' voltage limit (V) for the next step.
        """
        self.charge_drawn, self.current, self.voltage = (float(v) for v in end[:3])
        self.cutoff, self.overload = bool(end[3]), bool(end[4])
        return float(end[5])


    def _check_cutoff(self) -> None:
        """Cut off the motors if the terminal voltage is below min_voltage."""
        if self.voltage < self._params_vector[BATTERY_MIN_VOLTAGE]:
            self.cutoff = True



class Multirotor:
    """
    The multirotor class models dynamics and control allocation of a vehicle.
    """

    def __init__(
        self, params: VehicleParams, simulation: SimulationParams,
        substeps: Optional[int] = None
    ) -> None:
        """
        Parameters
        ----------
        params : VehicleParams
            The vehicle parameters. These completely describe the vehicle's properties.
            `Multirotor` copies the parameters at construction. Changing ``params``
            afterwards does not affect this instance; build a new `Multirotor`
            to apply the change.
        simulation : SimulationParams
            The simulation parameters.
        substeps : int, optional
            RK4 steps per simulation time step in `step_dynamics()` and `step_speeds()`.
            By default `multirotor.rollout.DEFAULT_SUBSTEPS` for the kind of vehicle: more
            with motors, whose dynamics are faster than the body's.
        """
        self.params: VehicleParams = deepcopy(params)
        self.simulation: SimulationParams = simulation
        self.dtype = self.params.inertia_matrix.dtype if simulation.dtype is None \
                     else simulation.dtype
        self.state: np.ndarray = None
        self.t: float = 0.
        self._dxdt = None

        # The parameters as arrays, made once and fixed: the steps read this copy, and
        # `rollout_params` returns a copy of it with the battery's current voltage limit.
        # The voltage limit and the disturbance in it are placeholders: a step passes the
        # current ones.
        self._vehicle = jax.tree_util.tree_map(
            lambda a: float(a) if np.ndim(a) == 0 else np.asarray(a, float),
            build_vehicle(self.params, simulation))
        n = len(self.params.propellers)
        has_motors = not isinstance(self._vehicle, BodyVehicle)
        self._use_thrust_constant = bool(self.params.propellers[0].use_thrust_constant)
        self.substeps: int = DEFAULT_SUBSTEPS['motors' if has_motors else 'body'] \
                             if substeps is None else int(substeps)
        # The state of the propellers (and motors), one element each. A propeller and its
        # motor have the same speed. Each `Propeller` and `Motor` indexes into these arrays.
        self._has_motors: bool = has_motors
        self._speed = np.zeros(n)
        self._voltage = np.zeros(n)
        self._current = np.zeros(n)
        self.propellers: List[Propeller] = []
        for i, params in enumerate(self.params.propellers):
            motor = Motor._view(
                params.motor, self.simulation, self.substeps, self._vehicle.motor_params,
                self._speed, self._voltage, self._current, i) if has_motors else None
            self.propellers.append(Propeller._view(params, self.simulation, motor, self._speed, i))
        # Without motors, propeller speed equals the speed signal, which is
        # fixed over a time step. With thrust = k_thrust * speed**2, forces and
        # torques then don't depend on the vehicle's state during the step.
        self._forces_fixed_over_step = not has_motors and self._use_thrust_constant

        if self.params.battery is not None:
            self.battery = Battery(self.params.battery, self.simulation)
        else:
            self.battery = Battery(BatteryParams(max_voltage=np.inf), self.simulation)
        if self.battery.has_model and not self._use_thrust_constant:
            raise ValueError('A battery model is not supported with blade-element thrust.')
        # The same parameters as JAX arrays, made once: the compiled steps take them, and a
        # numpy array would be converted at every call
        self._compiled_vehicle = jax.tree_util.tree_map(jnp.asarray, self._vehicle)
        self._compiled_motor_params = self._compiled_vehicle.motor_params if has_motors else None
        self._dt = jnp.asarray(float(self.simulation.dt))
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
        self._speed[:] = 0.
        self._voltage[:] = 0.
        self._current[:] = 0.
        self.battery.reset()

        self.alloc = self.alloc.astype(self.dtype)
        self.alloc_inverse = self.alloc_inverse.astype(self.dtype)
        self._positions = np.asarray(self.params.propeller_vectors).astype(self.dtype)
        self.params.inertia_matrix = np.asarray(self.params.inertia_matrix).astype(self.dtype)
        self._clockwise = np.asarray(self.params.clockwise)
        self.state = np.zeros(12, dtype=self.dtype)
        self._dxdt = np.zeros_like(self.state)
        return self.state


    @property
    def position(self) -> np.ndarray:
        """Position in the inertial frame."""
        return self.state[0:3]


    @property
    def velocity(self) -> np.ndarray:
        """Body-frame velocity"""
        return self.state[3:6]


    @property
    def inertial_velocity(self) -> np.ndarray:
        """Velocity in the inertial frame."""
        dcm = direction_cosine_matrix(*self.orientation)
        v_inertial = body_to_inertial(self.velocity, dcm)
        return v_inertial


    @property
    def acceleration(self) -> np.ndarray:
        """Body-frame acceleration"""
        return self._dxdt[3:6]


    @property
    def inertial_acceleration(self) -> np.ndarray:
        """Acceleration in the inertial frame."""
        dcm = direction_cosine_matrix(*self.orientation)
        a_inertial = body_to_inertial(self.acceleration, dcm)
        return a_inertial


    def rollout_params(self) -> Union[BodyVehicle, MotorsVehicle, BatteryVehicle]:
        """
        The parameters of this vehicle as the tuple that `multirotor.rollout` takes,
        with float64 copies of the arrays: `BodyVehicle`, `MotorsVehicle` (with motors) or
        `BatteryVehicle` (with motors and a battery model). The type says which of
        ``closed_loop_body``, ``closed_loop_motors`` and ``closed_loop_motors_battery``
        takes it (`multirotor.tuples.kind_of`).

        Returns
        -------
        BodyVehicle, MotorsVehicle or BatteryVehicle
            The parameters.

        Raises
        ------
        ValueError
            For blade-element thrust, which the rollout does not support.
        """
        if not self._use_thrust_constant:
            raise ValueError('Blade-element thrust is not supported.')
        v = self._vehicle
        copy = type(v)(*(float(f) if np.ndim(f) == 0 else np.array(f, float) for f in v))
        if self._has_motors:
            copy = copy._replace(max_voltage=float(self.battery.motor_voltage_limit))
        return copy


    def rollout_state(self) -> VehicleState:
        """
        The state of this vehicle as the `multirotor.tuples.VehicleState` that
        `multirotor.rollout` starts from, with float64 copies of the arrays.

        Returns
        -------
        VehicleState
            The state.
        """
        x = np.array(self.state, float)
        aux = np.zeros(3)
        if self._has_motors:
            x = np.concatenate((x, self._speed.astype(float)))
            if self.battery.has_model:
                b = self.battery
                aux = np.array((b.charge_drawn, float(b.cutoff), float(b.overload)))
        return VehicleState(x, aux, np.array(self.acceleration, float), float(self.t))


    def set_rollout_state(self, state: VehicleState) -> None:
        """
        Continue from a `multirotor.tuples.VehicleState`, such as the one a rollout returns.

        The motors' voltages and currents and the battery's voltage and current are outputs
        of a step, not state. They are updated by the next step.

        Parameters
        ----------
        state : VehicleState
            The state to continue from.

        Raises
        ------
        ValueError
            If the state is not that of this kind of vehicle.
        """
        x = np.asarray(state.x, float)
        expected = self.rollout_state().x.shape
        if x.shape != expected:
            raise ValueError(f'The vehicle state has shape {x.shape}; expected {expected}.')
        n_body = self.state.shape[0]
        self.state = x[:n_body].astype(self.dtype)
        self._dxdt[3:6] = np.asarray(state.acceleration, float)
        self.t = float(state.t)
        if self._has_motors:
            self._speed[:] = x[n_body:]
            if self.battery.has_model:
                battery = self.battery
                battery.charge_drawn = float(state.aux[0])
                battery.cutoff, battery.overload = bool(state.aux[1]), bool(state.aux[2])
                battery.current = 0.
                battery.voltage = 0. if battery.empty else battery.open_circuit_voltage()



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
        "The vehicle's weight (N)."
        return self.simulation.g * self.params.mass


    @property
    def motor_currents(self) -> np.ndarray:
        """
        Current (A) drawn by each motor, on the motor side of its speed
        controller. None if the vehicle has no motors.
        """
        if not self._has_motors:
            return None
        return self._current.copy()


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
        max_voltage = self.battery.params.max_voltage
        if not self._has_motors or not np.isfinite(max_voltage):
            return np.nan
        return np.sum(self._current * self._voltage) / max_voltage


    @property
    def speeds(self) -> np.ndarray:
        """Propeller speeds (rad/s). A copy, not a view of the internal array."""
        return np.array(self._speed, self.dtype)
    @speeds.setter
    def speeds(self, speeds: np.ndarray) -> None:
        self._step_propellers(speeds)


    def _motors_after_step(self, u: np.ndarray) -> np.ndarray:
        """
        All motors after one time step under speed signals ``u``, integrating the motors on
        their own, without changing their state.

        ``u``: The speed signals, shape (n,).
        Returns: The motor speeds (rad/s), and the voltages (V) and currents (A) at those
          speeds, shape (3, n).
        """
        inputs = np.concatenate(
            (self._speed, u, (self.battery.motor_voltage_limit,)), dtype=float)
        return np.asarray(_step_motor_only(
            inputs, self._compiled_motor_params, self._dt, substeps=self.substeps))


    def _apply_speeds(self, u: np.ndarray) -> np.ndarray:
        """
        Propeller speeds after one time step under speed signals ``u``, without
        changing the state of the propellers. Without motors, speed signals take
        effect instantaneously.

        ``u``: The speed signals, shape (n,).
        Returns: The propeller speeds (rad/s), shape (n,).
        """
        if not self._has_motors:
            return np.asarray(u)
        return self._motors_after_step(u)[0]





    def _step_propellers(self, u: np.ndarray) -> None:
        """
        Advance all propellers (and motors) by one time step under speed signals
        ``u``, on their own.

        ``u``: The speed signals, shape (n,).
        """
        if not self._has_motors:
            self._speed[:] = u
        else:
            self._speed[:], self._voltage[:], self._current[:] = self._motors_after_step(u)


    def _thrusts(self, speeds: np.ndarray, airstream_velocities: np.ndarray) -> np.ndarray:
        """
        Thrust (N) of each propeller.

        ``speeds``: Propeller speeds (rad/s), shape (n,).
        ``airstream_velocities``: Airstream velocity at each propeller, shape (3, n). Only used
          by the blade-element thrust model.
        Returns: Thrust of each propeller, shape (n,).
        """
        if self._use_thrust_constant:
            return self._vehicle.prop_params[PROP_K_THRUST] * speeds**2
        return thrusts_blade_element_model(
            speeds, airstream_velocities, self._vehicle.prop_params, self.simulation.rho)


    def get_forces_torques(self, speeds: np.ndarray, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the forces and torques acting on the vehicle's center of gravity
        given its state and the propeller speeds.

        Parameters
        ----------
        speeds : np.ndarray
            Propeller speeds (rad/s)
        state : np.ndarray
            State of the vehicle, shape (12,): position ``[0:3]``, body velocity
            ``[3:6]``, orientation ``[6:9]``, body angular rate ``[9:12]``

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The forces and torques acting on the body.
        """
        linear_vel_body = state[3:6]
        angular_vel_body = state[9:12]
        airstream_velocity_inertial = rotating_frame_derivative(
            self._positions,
            linear_vel_body,
            angular_vel_body)

        speeds = self._apply_speeds(speeds)
        thrusts = self._thrusts(speeds, airstream_velocity_inertial).astype(self.dtype)
        drag_torques = self._vehicle.prop_params[PROP_K_DRAG] * speeds**2
        # As `multirotor.physics.body_torques`: thrust acts along the body z-axis, and each
        # propeller's drag torque about it, opposite to its rotation
        positions = self._positions
        torques = np.asarray((
            (positions[1] * thrusts).sum(), -(positions[0] * thrusts).sum(),
            (self._clockwise * drag_torques).sum()))
        forces = np.zeros(3, dtype=self.dtype)
        forces[2] = thrusts.sum()
        return forces, torques.astype(self.dtype)


    def dxdt_dynamics(
        self, t: float, x: np.ndarray, u: np.ndarray, params: Optional[dict] = None
    ) -> np.ndarray:
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
        params : dict, optional
            Ignored. Present so that the method has the signature that the ``control``
            library calls.

        Returns
        -------
        np.ndarray
            The rate of change of state.
        """
        # This method must not have any side-effects. It should not change the
        # state of the vehicle. This method is called multiple times from the 
        # same state by the integrator, and the results should be consistent.
        # Do not need to get forces/torques on body, since the action array
        # already is a 6d vector of forces/torques.
        # forces, torques = self.get_forces_torques(u, x)
        u = np.asarray(u, dtype=float)
        return np.asarray(_rhs_body(np.asarray(x, dtype=float), u[:3], u[3:], self._compiled_vehicle))


    def dxdt_speeds(
        self, t: float, x: np.ndarray, u: np.ndarray,
        disturb_forces: np.ndarray=0., disturb_torques: np.ndarray=0.,
        params: Optional[dict] = None
    ) -> np.ndarray:
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
            Disturbing x,y,z forces in the vehicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.
        params : dict, optional
            Ignored. Present so that the method has the signature that the ``control``
            library calls.

        Returns
        -------
        np.ndarray
            The rate of change of state.
        """
        # This method must not have any side-effects. It should not change the
        # state of the vehicle. This method is called multiple times from the 
        # same state by the integrator, and the results should be consistent.
        forces, torques = self.get_forces_torques(u, x)
        return np.asarray(_rhs_body(
            np.asarray(x, dtype=float), forces.astype(float) + disturb_forces,
            torques.astype(float) + disturb_torques, self._compiled_vehicle))


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
        inputs = np.concatenate((self.state, u), dtype=float)
        out = np.asarray(_step_forces(
            inputs, self._compiled_vehicle, self._dt, substeps=self.substeps))
        self.state = out[:N_BODY_STATES].astype(self.dtype)
        self._dxdt = out[N_BODY_STATES:].astype(self.dtype)
        # TODO: inverse solve for speed = forces to set propeller speeds
        return self.state


    def step_speeds(
        self, u: np.ndarray, disturb_forces: np.ndarray=0.,
        disturb_torques: np.ndarray=0.
    ) -> np.ndarray:
        """
        Calculate the next state of the vehicle from the speed signals of its
        ``n`` propellers.

        With motors, the body and the motors are integrated together, so the
        forces during the step follow the motor speeds as they change.

        Parameters
        ----------
        u : np.ndarray
            The speed signals, shape (n,). Each goes to a propeller's ``step()``.
            A signal is a speed (rad/s), or a voltage (V) when the propeller has a
            motor and ``MotorParams.speed_voltage_scaling`` is set.
        disturb_forces : np.ndarray, optional
            Disturbing x,y,z forces in the vehicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.

        Returns
        -------
        np.ndarray
            The new state of the vehicle.
        """
        self.t += self.simulation.dt
        if self._has_motors:
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

        ``u``: Speed signals (rad/s).
        """
        inputs = np.concatenate(
            (self.state, u, self._disturbance(disturb_forces, disturb_torques)), dtype=float)
        out = np.asarray(_step_body(
            inputs, self._compiled_vehicle, self._dt, substeps=self.substeps))
        self.state = out[:N_BODY_STATES].astype(self.dtype)
        self._dxdt = out[N_BODY_STATES:].astype(self.dtype)
        self._step_propellers(u)


    @staticmethod
    def _disturbance(disturb_forces: np.ndarray, disturb_torques: np.ndarray) -> np.ndarray:
        """The disturbing forces and torques as one array, shape (6,)."""
        disturbance = np.zeros(6)
        disturbance[:3] += disturb_forces
        disturbance[3:] += disturb_torques
        return disturbance


    def _step_with_varying_forces(
        self, u: np.ndarray, disturb_forces: np.ndarray,
        disturb_torques: np.ndarray
    ) -> None:
        """
        Advance a vehicle without motors and with blade-element thrust by one
        time step. Thrust depends on the airstream, so forces are recalculated
        at every evaluation of the derivative, in numpy.

        ``u``: Speed signals (rad/s).
        """
        def dxdt(x: np.ndarray) -> np.ndarray:
            return self.dxdt_speeds(
                t=self.t, x=x, u=u, disturb_forces=disturb_forces,
                disturb_torques=disturb_torques)
        x0 = self.state.astype(float)
        self._dxdt = dxdt(x0).astype(self.dtype)
        self.state = _rk4(dxdt, x0, self.simulation.dt, self.substeps).astype(self.dtype)
        self._step_propellers(u)


    def _dxdt_with_motors_blade_element(
        self, x: np.ndarray, u: np.ndarray, max_voltage: float, disturbance: np.ndarray
    ) -> np.ndarray:
        """
        Rate of change of the body and its motors, for vehicles whose thrust comes from the
        blade-element model. Thrust depends on the airstream at each propeller, so it is
        solved numerically at each evaluation, in numpy.

        ``x``: The 12 body states, then the motor speeds, shape (12 + n,).
        """
        body_state, speeds = x[:N_BODY_STATES], x[N_BODY_STATES:]
        # Same airstream inputs as `get_forces_torques`: the body-frame velocity and
        # angular rate.
        positions = self._vehicle.positions
        airstream_velocities = rotating_frame_derivative(
            positions, body_state[3:6], body_state[9:12])
        thrusts = thrusts_blade_element_model(
            speeds, airstream_velocities, self._vehicle.prop_params, float(self.simulation.rho))
        drag_torques = self._vehicle.prop_params[PROP_K_DRAG] * speeds**2
        torques = np.asarray((
            (positions[1] * thrusts).sum(), -(positions[0] * thrusts).sum(),
            (self._vehicle.clockwise * drag_torques).sum()))
        forces = np.asarray([0., 0., thrusts.sum()])
        return np.concatenate((
            np.asarray(_rhs_body(
                body_state, forces + disturbance[:3], torques + disturbance[3:],
                self._compiled_vehicle)),
            np.asarray(_motor_acc(speeds, u, max_voltage, self._compiled_motor_params))))


    def _step_with_motors(
        self, u: np.ndarray, disturb_forces: np.ndarray,
        disturb_torques: np.ndarray
    ) -> None:
        """
        Advance the body and the motors by one time step, integrating them
        together.
        """
        battery = self.battery
        n_motors = len(self._speed)
        u = np.asarray(u, dtype=float)
        dt = float(self.simulation.dt)
        disturbance = self._disturbance(disturb_forces, disturb_torques)
        max_voltage = float(battery.motor_voltage_limit)
        x0 = np.concatenate((self.state.astype(float), self._speed))
        n = x0.shape[0]
        if not self._use_thrust_constant:
            def dxdt(x: np.ndarray) -> np.ndarray:
                return self._dxdt_with_motors_blade_element(x, u, max_voltage, disturbance)
            self._dxdt = dxdt(x0)[:N_BODY_STATES].astype(self.dtype)
            x = _rk4(dxdt, x0, dt, self.substeps)
            self._speed[:] = x[N_BODY_STATES:]
            self._voltage[:], self._current[:] = np.asarray(_motor_electrical(
                self._speed.astype(float), u, max_voltage, self._compiled_motor_params))
        elif battery.has_model:
            inputs = np.concatenate((
                x0, u, disturbance,
                (battery.charge_drawn, battery.cutoff, battery.overload)), dtype=float)
            x = np.asarray(_step_motors_battery(
                inputs, self._compiled_vehicle, self._dt, substeps=self.substeps))
            self._dxdt = x[n:n + N_BODY_STATES].astype(self.dtype)
            electrical = n + N_BODY_STATES + 6
            battery._end_step(x[n + N_BODY_STATES:electrical])
            self._speed[:] = x[N_BODY_STATES:n]
            self._voltage[:] = x[electrical:electrical + n_motors]
            self._current[:] = x[electrical + n_motors:]
        else:
            inputs = np.concatenate((x0, u, disturbance, (max_voltage,)), dtype=float)
            x = np.asarray(_step_motors(
                inputs, self._compiled_vehicle, self._dt, substeps=self.substeps))
            self._dxdt = x[n:n + N_BODY_STATES].astype(self.dtype)
            electrical = n + N_BODY_STATES
            self._speed[:] = x[N_BODY_STATES:n]
            self._voltage[:] = x[electrical:electrical + n_motors]
            self._current[:] = x[electrical + n_motors:]
        self.state = x[:N_BODY_STATES].astype(self.dtype)


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
        vec = np.asarray([thrust, *torques], self.dtype)
        # return np.sqrt(np.linalg.lstsq(self.alloc, vec, rcond=None)[0])
        return np.sqrt(
            np.clip(self.alloc_inverse @ vec, a_min=0., a_max=None)
        )


    def nonlinear_dynamics_controls_system(
        self, linearize: bool=False, perturbation: float=1e-1,
        about_state: np.ndarray=0, about_action: np.ndarray=0
    ) -> Any:
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
    ) -> Any:
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
