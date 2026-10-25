from typing import Callable, List, Tuple
from copy import deepcopy

import numpy as np
from scipy.integrate import odeint

from .vehicle import MotorParams, PropellerParams, SimulationParams, VehicleParams, BatteryParams
from .coords import body_to_inertial, direction_cosine_matrix, rotating_frame_derivative, angular_to_euler_rate
from .physics import (
    thrust, thrusts_physics, body_torques, motor_step_explicit,
    PROP_K_THRUST, PROP_K_DRAG
)
from .data import MotorData, PropellerData
from .dynamics import rhs_body
from .helpers import control_allocation_matrix


Integrator = Callable[[Callable, np.ndarray, float, tuple], np.ndarray]
"""
A solver for one simulation step: ``integrator(fun, y0, dt, args) -> y``.

Advances the initial state ``y0`` from ``t=0`` to ``t=dt`` under
``dy/dt = fun(t, y, *args)`` and returns the state at ``t=dt``.
"""


def odeint_integrator(
    fun: Callable, y0: np.ndarray, dt: float, args: tuple = ()
) -> np.ndarray:
    """
    The default :data:`Integrator`: LSODA via `scipy.integrate.odeint`.

    Parameters
    ----------
    fun : Callable
        The derivative function, with signature ``fun(t, y, *args)``.
    y0 : np.ndarray
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
    return odeint(fun, y0, (0, dt), args=args, rtol=1e-4, atol=1e-4, tfirst=True)[-1]



class Propeller:
    """
    Models the thrust and aerodynamics of the propeller blades spinning at a
    certain rate.

    The propeller's state lives in a `PropellerData` object, at index ``i``. A
    standalone propeller owns a data object of size 1. Propellers of a
    `Multirotor` share the vehicle's data object, each at its own index.
    """

    def __init__(
        self, params: PropellerParams, simulation: SimulationParams
    ) -> None:
        params = deepcopy(params)
        self._attach(PropellerData([params]), 0, params, simulation)


    @classmethod
    def _view(
        cls, data: PropellerData, i: int, params: PropellerParams,
        simulation: SimulationParams
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

        Returns
        -------
        Propeller
        """
        propeller = cls.__new__(cls)
        propeller._attach(data, i, params, simulation)
        return propeller


    def _attach(
        self, data: PropellerData, i: int, params: PropellerParams,
        simulation: SimulationParams
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
            self.motor: Motor = Motor._view(data.motors, i, params.motor, simulation)
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

    def __init__(self, params: MotorParams, simulation: SimulationParams) -> None:
        params = deepcopy(params)
        self._attach(MotorData([params]), 0, params, simulation)


    @classmethod
    def _view(
        cls, data: MotorData, i: int, params: MotorParams,
        simulation: SimulationParams
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

        Returns
        -------
        Motor
        """
        motor = cls.__new__(cls)
        motor._attach(data, i, params, simulation)
        return motor


    def _attach(
        self, data: MotorData, i: int, params: MotorParams,
        simulation: SimulationParams
    ) -> None:
        self.params: MotorParams = params
        self.simulation: SimulationParams = simulation
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
        self._data._last_acc[self._i] = 0.


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
        speed, _, _, _ = self._advance(u, max_voltage)
        return float(speed)


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
        speed, voltage, current, acc = self._advance(u, max_voltage)
        self.speed, self.voltage, self.current = speed, voltage, current
        self._data._last_acc[self._i] = acc
        return self.speed


    def _advance(self, u: float, max_voltage: float) -> Tuple[float, float, float, float]:
        """
        Speed, voltage, current and angular acceleration of this motor after one
        time step, without changing its state.
        """
        i = slice(self._i, self._i + 1)
        speed, voltage, current, acc = motor_step_explicit(
            self._data.speed[i], self._data._last_acc[i], np.asarray([u]),
            max_voltage, self._data.params[:, i], self.simulation.dt
        )
        return speed[0], voltage[0], current[0], acc[0]



class Battery:
    """
    Models the state of charge of the battery of the Multirotor.
    """


    def __init__(self, params: BatteryParams, simulation: SimulationParams) -> None:
        self.params = deepcopy(params)
        self.simulation = simulation


    def reset(self):
        self.voltage = self.params.max_voltage
        return self.voltage


    @property
    def state(self) -> float:
        return self.voltage


    def step(self):
        pass



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
        self.dxdt_decimals = max(1, 1 - int(np.log10(self.simulation.dt)))

        # Parameters and state of all propellers (and motors) as arrays. Each
        # Propeller indexes into it.
        self._data = PropellerData(self.params.propellers)
        self.propellers: List[Propeller] = [
            Propeller._view(self._data, i, params, self.simulation)
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
        if self.battery is not None:
            self.battery.reset()

        self.alloc = self.alloc.astype(self.dtype)
        self.alloc_inverse = self.alloc_inverse.astype(self.dtype)
        self.params.propeller_vectors = self.params.propeller_vectors.astype(self.dtype)
        self.params.inertia_matrix = self.params.inertia_matrix.astype(self.dtype)
        self.params.inertia_matrix_inverse = self.params.inertia_matrix_inverse.astype(self.dtype)
        self._clockwise = np.asarray(self.params.clockwise)
        self._body_params = (
            self.simulation.g, self.params.mass,
            self.params.inertia_matrix, self.params.inertia_matrix_inverse
        )
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
    def current_average(self) -> float:
        """Duty-cycle adjusted currrent draw from battery."""
        peak_voltage = self.battery.params.max_voltage
        motors = self._data.motors
        duty_cycle = motors.voltage / peak_voltage
        return np.sum(duty_cycle * motors.current)


    @property
    def speeds(self) -> np.ndarray:
        return np.asarray(self._data.speed, self.dtype)
    @speeds.setter
    def speeds(self, speeds: np.ndarray):
        self._step_propellers(speeds)


    def _motor_step(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Speeds, voltages, currents and angular accelerations of all motors after
        one time step under speed signals ``u``, without changing their state.
        """
        motors = self._data.motors
        return motor_step_explicit(
            motors.speed, motors._last_acc, np.asarray(u), self.battery.voltage,
            motors.params, self.simulation.dt
        )


    def _apply_speeds(self, u: np.ndarray) -> np.ndarray:
        """
        Propeller speeds after one time step under speed signals ``u``, without
        changing the state of the propellers. Without motors, speed signals take
        effect instantaneously.
        """
        if self._data.motors is None:
            return np.asarray(u)
        speed, _, _, _ = self._motor_step(u)
        return speed


    def _step_propellers(self, u: np.ndarray) -> None:
        """
        Advance all propellers (and motors) by one time step under speed signals
        ``u``.
        """
        motors = self._data.motors
        if motors is None:
            self._data.speed[:] = u
        else:
            speed, voltage, current, acc = self._motor_step(u)
            motors.speed[:] = speed
            motors.voltage[:] = voltage
            motors.current[:] = current
            motors._last_acc[:] = acc


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
            # k_thrust takes the dtype of the speeds, which reproduces the
            # arithmetic of a Python float times a numpy scalar.
            k_thrust = self._data.params[PROP_K_THRUST].astype(speeds.dtype)
            return k_thrust * speeds**2
        return thrusts_physics(
            speeds, airstream_velocities, self._data.params, self.simulation.rho)


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
        torque_vec = body_torques(
            thrusts, drag_torques, self._clockwise, self.params.propeller_vectors)
        forces = np.zeros(3, dtype=self.dtype)
        forces[2] = thrusts.sum()
        torques = torque_vec.sum(axis=1)
        return forces, torques


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
        return rhs_body(
            t, x, u[:3], u[3:], self._body_params, self.dtype, self.dxdt_decimals)


    def dxdt_speeds(
        self, t: float, x: np.ndarray, u: np.ndarray,
        disturb_forces: np.ndarray=0., disturb_torques: np.ndarray=0., params=None
    ):
        """
        Calculate the rate of change of state given the propeller speeds on the
        system (rad/s).

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
            t, x, forces + disturb_forces, torques + disturb_torques,
            self._body_params, self.dtype, self.dxdt_decimals)


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
        args = (u[:3], u[3:], self._body_params, self.dtype, self.dxdt_decimals)
        self._dxdt = self._rhs_body(self.t, self.state, *args)
        self.state = self.integrator(
            self._rhs_body, self.state, self.simulation.dt, args
        )
        self.state = np.around(self.state, 4).astype(self.dtype)
        # TODO: inverse solve for speed = forces to set propeller speeds
        return self.state


    def step_speeds(
        self, u: np.ndarray, disturb_forces: np.ndarray=0.,
        disturb_torques: np.ndarray=0.
    ) -> np.ndarray:
        """
        Given the n-vector of propeller speed signals, calculate
        the next state of the vehicle. Where n is number of propellers.

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
        if self._forces_fixed_over_step:
            # Forces and torques depend only on the speed signals, which are
            # fixed over the step, so they are calculated once instead of at
            # every evaluation of the derivative.
            forces, torques = self.get_forces_torques(u, self.state)
            args = (
                forces + disturb_forces, torques + disturb_torques,
                self._body_params, self.dtype, self.dxdt_decimals
            )
            self._dxdt = self._rhs_body(self.t, self.state, *args)
            self.state = self.integrator(
                self._rhs_body, self.state, self.simulation.dt, args
            )
        else:
            self._dxdt = self.dxdt_speeds(
                t=self.t, x=self.state, u=u,
                disturb_forces=disturb_forces, disturb_torques=disturb_torques
            )
            self.state = self.integrator(
                self.dxdt_speeds, self.state, self.simulation.dt,
                (u, disturb_forces, disturb_torques)
            )
        self.state = np.around(self.state, 4).astype(self.dtype)
        # print('post-x', self.t // self.simulation.dt, self.state.dtype)
        self._step_propellers(u)
        self.battery.step()
        return self.state


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
