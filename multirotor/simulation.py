from typing import Callable, List, Tuple
from copy import deepcopy

import numpy as np
from scipy.integrate import odeint, trapezoid

from .vehicle import MotorParams, PropellerParams, SimulationParams, VehicleParams, BatteryParams
from .coords import body_to_inertial, direction_cosine_matrix, rotating_frame_derivative, angular_to_euler_rate
from .physics import thrust, torque, apply_forces_torques
from .helpers import control_allocation_matrix



class Propeller:
    """
    Models the thrust and aerodynamics of the propeller blades spinning at a 
    certain rate.
    """

    def __init__(
        self, params: PropellerParams, simulation: SimulationParams
    ) -> None:
        self.params: PropellerParams = deepcopy(params)
        self.simulation: SimulationParams = simulation
        self.speed: float = 0.
        "Radians per second"

        # Thrust constant uses a simpler quardatic relationship to calculate
        # propeller thrust.
        if self.params.use_thrust_constant:
            self.thrust: Callable = self._thrust_constant
        else:
            self.thrust: Callable = self._thrust_physics
        # If not motor is provided, then speed signals take effect instantaneously
        if params.motor is not None:
            self.motor = Motor(params.motor, self.simulation)
        else:
            self.motor: Motor = None


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
            self.speed = self.motor.step(u, **kwargs)
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
    """

    def __init__(self, params: MotorParams, simulation: SimulationParams) -> None:
        self.params = deepcopy(params)
        self.simulation = simulation
        self.speed: float = 0.
        "Radians per second"
        self.voltage = 0.
        self.current = 0.
        self._last_angular_acc = 0.


    def reset(self) -> float:
        self.speed = 0.
        self.voltage = 0.
        self.current = 0.
        self._last_angular_acc = 0.


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
        # This method simply calls step() but restores the state of the object
        # afterwards, thus making it a "pure" function.
        voltage, current, last_acc, last_speed = \
            self.voltage, self.current, self._last_angular_acc, self.speed

        speed = self.step(u, max_voltage=max_voltage)

        self.voltage, self.current, self._last_angular_acc, self.speed = \
            voltage, current, last_acc, last_speed
        return speed


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
        self.voltage = np.clip(self.params.speed_voltage_scaling * u, 0, max_voltage)
        self.current = np.clip((self.voltage - self.speed * self.params.k_emf) / self.params.resistance, 0, self.params.max_current)
        torque = self.params.k_torque * self.current
        # Subtract drag torque and dynamic friction from electrical torque
        net_torque = torque - \
                     self.params.k_df * self.speed - \
                     self.params.k_drag * self.speed**2
        accs = (self._last_angular_acc, net_torque / self.params.moment_of_inertia)
        self.speed += \
        trapezoid(
            accs,
            dx=self.simulation.dt
        )
        self._last_angular_acc = accs[1]
        return self.speed



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

    def __init__(self, params: VehicleParams, simulation: SimulationParams) -> None:
        """
        Parameters
        ----------
        params : VehicleParams
            The vehicle parameters. These completely describe the vehicle's properties.
            The parameters are copied by this class, so any changes made to the params
            object is isolated from this instance.
        simulation : SimulationParams
            The simulation parameters.
        """
        self.params: VehicleParams = deepcopy(params)
        self.simulation: SimulationParams = simulation
        self.dtype = self.params.inertia_matrix.dtype if simulation.dtype is None \
                     else simulation.dtype
        self.state: np.ndarray = None
        self.t: float = 0.
        self._dxdt = None
        self.dxdt_decimals = max(1, 1 - int(np.log10(self.simulation.dt)))

        self.propellers: List[Propeller] = []
        for params in self.params.propellers:
            self.propellers.append(Propeller(params, self.simulation))

        if self.params.battery is not None:
            self.battery = Battery(self.params.battery, self.simulation)
        else:
            self.battery = Battery(BatteryParams(max_voltage=np.inf), self.simulation)
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
        for p in self.propellers:
            p.reset()
        if self.battery is not None:
            self.battery.reset()

        self.alloc, self.alloc_inverse = control_allocation_matrix(self.params)
        self.alloc = self.alloc.astype(self.dtype)
        self.params.propeller_vectors = self.params.propeller_vectors.astype(self.dtype)
        self.alloc_inverse = self.alloc_inverse.astype(self.dtype)
        self.params.inertia_matrix = self.params.inertia_matrix.astype(self.dtype)
        self.params.inertia_matrix_inverse = self.params.inertia_matrix_inverse.astype(self.dtype)
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
        currents = np.asarray([p.motor.current for p in self.propellers])
        voltages = np.asarray([p.motor.voltage for p in self.propellers])
        duty_cycle = voltages / peak_voltage
        return np.sum(duty_cycle * currents)


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

        thrust_vec = np.zeros((3, len(self.propellers)), dtype=self.dtype)
        torque_vec = np.zeros_like(thrust_vec)

        for i, (speed, prop, clockwise) in enumerate(zip(
                speeds,
                self.propellers,
                self.params.clockwise)
        ):
            last_speed = prop.speed
            speed = prop.apply_speed(speed, max_voltage=self.battery.voltage)
            angular_acc = (speed - last_speed) / self.simulation.dt
            thrust_vec[2, i] = prop.thrust(
                speed, airstream_velocity_inertial[:, i]
            )
            torque_vec[:, i] = torque(
                self.params.propeller_vectors[:,i], thrust_vec[:,i],
                prop.params.moment_of_inertia, angular_acc,
                prop.params.k_drag, speed,
                clockwise
            )
        forces = thrust_vec.sum(axis=1)
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
        dxdt = apply_forces_torques(
            u[:3], u[3:], x.astype(self.dtype), self.simulation.g,
            self.params.mass, self.params.inertia_matrix, self.params.inertia_matrix_inverse)
        return np.around(dxdt, self.dxdt_decimals)


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
        forces, torques = self.get_forces_torques(
            u, x)
        # print('dxdt-x', self.t // self.simulation.dt, x.dtype)
        dxdt = apply_forces_torques(
            forces+disturb_forces, torques+disturb_torques, x.astype(self.dtype), self.simulation.g,
            self.params.mass, self.params.inertia_matrix, self.params.inertia_matrix_inverse)
        # print('dxdt', self.t // self.simulation.dt, dxdt.dtype)
        return np.around(dxdt, self.dxdt_decimals)


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
        self._dxdt = self.dxdt_dynamics(t=self.t, x=self.state, u=u)
        self.state = odeint(
            self.dxdt_dynamics, self.state, (0, self.simulation.dt),
            args=(u,),
            rtol=1e-4, atol=1e-4, tfirst=True
        )[-1]
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
        self._dxdt = self.dxdt_speeds(
            t=self.t, x=self.state, u=u,
            disturb_forces=disturb_forces, disturb_torques=disturb_torques
        )
        # print('pre-x', self.t // self.simulation.dt, self.state.dtype)
        self.state = odeint(
            self.dxdt_speeds, self.state, (0, self.simulation.dt),
            args=(u, disturb_forces, disturb_torques),
            rtol=1e-4, atol=1e-4, tfirst=True
        )[-1]
        self.state = np.around(self.state, 4).astype(self.dtype)
        # print('post-x', self.t // self.simulation.dt, self.state.dtype)
        for u_, prop in zip(u, self.propellers):
            prop.step(u_, max_voltage=self.battery.voltage)
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
