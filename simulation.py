from typing import List, Tuple

import numpy as np
from numpy import (cos, sin)
from scipy.integrate import odeint

from vehicle import PropellerParams, SimulationParams, VehicleParams
from coords import body_to_inertial, rotating_frame_derivative, angular_to_euler_rate
from physics import thrust, torque, apply_forces_torques



class Propeller:

    def __init__(self, params: PropellerParams, simulation: SimulationParams) -> None:
        self.params: PropellerParams = params
        self.simulation: SimulationParams = simulation
        self.speed: float = None
        "Revolutions per minute"
        self._induced_velocity_guess = 0.1
        "Guess for the induced velocity about the propeller"


    def reset(self):
        self.speed = 0.
        self._induced_velocity_guess = 0.1


    def step(self, u: float) -> float:
        return self.apply_speed(u)


    def apply_speed(self, speed: float) -> float:
        self.speed = speed
        return self.speed


    def thrust(self, speed=None, airstream_velocity: np.ndarray=np.zeros(3)) -> float:
        p = self.params
        speed = speed or self.speed
        return thrust(
            speed, airstream_velocity,
            p.R, p.A, self.simulation.rho, p.a, p.b, p.c, p.eta, p.theta0, p.theta1,
            self._induced_velocity_guess)


    @property
    def state(self) -> float:
        return self.speed



class Multirotor:

    def __init__(self, params: VehicleParams, simulation: SimulationParams) -> None:
        self.params: VehicleParams = params
        self.simulation: SimulationParams = simulation
        self.state: np.ndarray = None
        self.propellers: List[Propeller] = None
        self.propeller_vectors: np.matrix = None
        self.t: float = None
        self.reset()


    def reset(self):
        self.t = 0.
        self.propellers = []
        for params in self.params.propellers:
            self.propellers.append(Propeller(params, self.simulation))

        x = cos(self.params.angles) * self.params.distances
        y = sin(self.params.angles) * self.params.distances
        z = np.zeros_like(y)
        self.propeller_vectors = np.vstack((x, y, z))

        self.inertial_matrix_inverse = np.asmatrix(np.linalg.inv(self.params.inertia_matrix))

        self.state = np.zeros(12)


    @property
    def position(self) -> np.ndarray:
        """Navigation coordinates (height = - z coordinate)"""
        return np.asarray([self.state[9], self.state[10], -self.state[11]])


    @property
    def velocity(self) -> np.ndarray:
        """Body-frame velocity"""
        return self.state[:3]


    @property
    def navigation_velocity(self) -> np.ndarray:
        v_inertial = body_to_inertial(self.velocity)
        v_inertial[2] *= -1 # convert inertial to navigation frame (h = -z)
        return v_inertial


    @property
    def orientation(self) -> np.ndarray:
        """Euler rotations (roll, pitch, yaw)"""
        return self.state[6:9]


    @property
    def angular_velocity(self) -> np.ndarray:
        """Angular rate of body frame axes (not same as rate of roll, pitch, yaw)"""
        return self.state[3:6]


    @property
    def euler_rate(self) -> np.ndarray:
        """Euler rate of vehicle d(roll, pitch, yaw)/dt"""
        return angular_to_euler_rate(self.angular_velocity, self.orientation)


    def get_forces_torques(self, speeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        linear_vel_body = self.state[:3]
        angular_vel_body = self.state[3:6]
        airstream_velocity_inertial = rotating_frame_derivative(
            self.propeller_vectors,
            linear_vel_body,
            angular_vel_body)

        thrust_vec = np.zeros((3, len(self.propellers)))
        torque_vec = np.zeros_like(thrust_vec)

        for i, (speed, prop) in enumerate(zip(
                speeds,
                self.propellers)):
            speed = prop.apply_speed(speed)
            thrust_vec[2, i] = -thrust(
                        speed, airstream_velocity_inertial[:, i], prop.params.R,
                        prop.params.A, self.simulation.rho, prop.params.a,
                        prop.params.b, prop.params.c, prop.params.eta,
                        prop.params.theta0, prop.params.theta1,
                        prop._induced_velocity_guess)
            
            # Store solution as initial guess for next step's solution
            torque_vec[:, i] = torque(self.propeller_vectors[:,i], -thrust_vec[2,i])

        forces = thrust_vec.sum(axis=1)
        torques = torque_vec.sum(axis=1)
        return forces, torques


    def dxdt(self, x: np.ndarray, t: float, u: np.ndarray):
        forces, torques = self.get_forces_torques(u)
        xdot = apply_forces_torques(
            forces, torques, self.state, self.simulation.g,
            self.params.mass, self.params.inertia_matrix, self.params.inertia_matrix_inverse)
        return xdot


    def step(self, u: np.ndarray):
        self.t += self.simulation.dt
        self.state = odeint(self.dxdt, self.state, (0, self.simulation.dt), args=(u,))[-1]
        return self.state
