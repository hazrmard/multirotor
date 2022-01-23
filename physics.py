from typing import List

import numpy as np
from numpy import (cos, sin)
from scipy.optimize import fsolve

from vehicle import PropellerParams, SimulationParams, VehicleParams
from coords import rotating_frame_derivative



def thrustEqn(vi, *prop_params):
    
    # Unpack parameters
    R,A,rho,a,b,c,eta,theta0,theta1,u,v,w,Omega = prop_params
    
    # Calculate local airflow velocity at propeller with vi, V'
    Vprime = np.sqrt(u**2 + v**2 + (w - vi)**2)
    
    # Calculate Thrust averaged over one revolution of propeller using vi
    Thrust = 1/4 * rho * a * b * c * R * \
        ( (w - vi) * Omega * R + 2/3 * (Omega * R)**2 * (theta0 + 3/4 * theta1) + \
          (u**2 + v**2) * (theta0 + 1/2 * theta1) )
    
    # Calculate residual for equation: Thrust = mass flow rate * delta Velocity
    residual = eta * 2 * vi * rho * A * Vprime - Thrust
    return residual



class Propeller:

    def __init__(self, params: PropellerParams) -> None:
        self.params: PropellerParams = params
        self.speed: float = None
        "Revolutions per minute"
        self._last_induced_velocity = 0.
        "Guess for the induced velocity about the propeller"

        self.theta0 = 2*np.arctan2(self.params.p_pitch, (2 * np.pi * 3/4 * self.params.p_diameter/2))
        self.theta1 = -4 / 3 * np.arctan2(self.params.p_pitch, 2 * np.pi * 3/4 * self.params.p_diameter/2)


    def step(self, u: float, dt: float) -> float:
        return self.apply_speed(u)


    def apply_speed(self, speed: float) -> float:
        self.speed = speed
        return self.speed


    def thrust(self, airstream_velocity: np.ndarray=(0,0,0)) -> float:
        u, v, w = airstream_velocity
        # Inputs: Current state x[k], Commanded Propeller RPM inputs u[k],
        #         Propeller location distances dx, dy (m)
        # Returns: Thrust vector for 4 propellers (Newtons)
        
        # Convert commanded RPM to rad/s
        Omega = 2 * np.pi / 60 * self.speed
        
        #Collect propeller config, state, and input parameters
        prop_params = (self.params.R,self.params.A,self.params.rho,self.params.a,self.params.b,self.params.c,self.params.eta,self.theta0,self.theta1,u,v,w,Omega)
        
        # Numerically solve for propeller induced velocity, vi
        # using nonlinear root finder, fsolve, and prop_params
        vi = fsolve(thrustEqn, self._last_induced_velocity, args=prop_params)
        self._last_induced_velocity = vi
        
        # Plug vi back into Thrust equation to solve for T
        Vprime = np.sqrt(u**2 + v**2 + (w - vi)**2)
        Thrust = self.params.eta * 2 * vi * self.params.rho * self.params.A * Vprime
        
        return Thrust


    def torque(self, position_vector: np.ndarray, thrust: float) -> float:
        thrust = np.asarray([0, 0, -thrust])
        return np.cross(position_vector, thrust)

    @property
    def state(self) -> float:
        return self.speed



class Multirotor:

    def __init__(self, vehicle: VehicleParams, simulation: SimulationParams) -> None:
        self.vehicle: VehicleParams = vehicle
        self.simulation: SimulationParams = simulation
        self.state: np.ndarray = None
        self.propellers: List[Propeller] = None
        self.propeller_vectors: np.matrix = None


    def reset(self):
        self.propellers = []
        for params in self.vehicle.propellers:
            self.propellers.append(Propeller(params))
        x = cos(self.vehicle.angles) * self.vehicle.distances
        y = sin(self.vehicle.angles) * self.vehicle.distances
        z = np.zeros_like(y)
        self.propeller_vectors = np.asmatrix(np.vstack((x, y, z)))
        self.inertial_matrix_inverse = np.asmatrix(np.linalg.inv(self.vehicle.inertia_matrix))
        self.state = np.zeros(12)


    @property
    def position(self) -> np.ndarray:
        """Navigation coordinates (height = - z coordinate)"""
        return np.asfarray(self.state[9], self.state[10], -self.state[11])


    @property
    def velocity(self) -> np.ndarray:
        """Body-frame velocity"""
        return self.state[:3]


    @property
    def orientation(self) -> np.ndarray:
        """Euler rotations (roll, pitch, yaw)"""
        return self.state[6:9]


    @property
    def angular_velocity(self) -> np.ndarray:
        """Angular rate of body frame axes (not same as rate of roll, pitch, yaw)"""
        return self.state[3:6]



    def apply_forces_torques(self, forces: np.ndarray, torques: np.ndarray):
        # Store state variables in a readable format
        x = self.state
        ub = x[0]
        vb = x[1]
        wb = x[2]
        p = x[3]
        q = x[4]
        r = x[5]
        phi = x[6]
        theta = x[7]
        psi = x[8]
        xI = x[9]
        yI = x[10]
        hI = x[11]
        
        # Pre-calculate trig values
        cphi = np.cos(phi);   sphi = np.sin(phi)
        cthe = np.cos(theta); sthe = np.sin(theta)
        cpsi = np.cos(psi);   spsi = np.sin(psi)

        fx, fy, fz = forces
        tx, ty, tz = torques
        I = self.vehicle.inertia_matrix
        I_inv = self.inertial_matrix_inverse
        
        # Calculate the derivative of the state matrix using EOM
        xdot = np.zeros_like(self.state)
        
        xdot[0] = -self.simulation.g * sthe + r * vb - q * wb  # = udot
        xdot[1] = self.simulation.g * sphi * cthe - r * ub + p * wb # = vdot
        xdot[2] = 1/self.vehicle.mass * (fz) + self.simulation.g * cphi * cthe + q * ub - p * vb # = wdot


        # xdot[3] = 1/Ixx * (tx + (Iyy - Izz) * q * r)  # = pdot
        # xdot[4] = 1/Iyy * (ty + (Izz - Ixx) * p * r)  # = qdot
        # xdot[5] = 1/Izz * (tz + (Ixx - Iyy) * p * q)  # = rdot
        xdot[3:6] = I_inv @ (torques - np.cross(x[3:6], I @ x[3:6]))

        xdot[6] = p + (q*sphi + r*cphi) * sthe / cthe  # = phidot
        xdot[7] = q * cphi - r * sphi  # = thetadot
        xdot[8] = (q * sphi + r * cphi) / cthe  # = psidot
        
        xdot[9] = cthe*cpsi*ub + (-cphi * spsi + sphi*sthe*cpsi) * vb + \
            (sphi*spsi+cphi*sthe*cpsi) * wb  # = xIdot
            
        xdot[10] = cthe*spsi * ub + (cphi*cpsi+sphi*sthe*spsi) * vb + \
            (-sphi*cpsi+cphi*sthe*spsi) * wb # = yIdot
            
        xdot[11] = (-sthe * ub + sphi*cthe * vb + cphi*cthe * wb) # = zIdot
        
        return xdot


    def apply_propeller_speeds(self, speeds: np.ndarray):
        linear_vel_body = self.state[:3]
        angular_vel_body = self.state[3:6]
        airstream_velocity_inertial = rotating_frame_derivative(
            self.propeller_vectors,
            linear_vel_body,
            angular_vel_body)
        thrust = np.zeros(3, len(self.propellers))
        torque = np.zeros_like(thrust)
        for i, speed, prop in enumerate(zip(
                speeds,
                self.propellers)):
            speed = prop.apply_speed(speed)
            thrust[2, i] = -prop.thrust(airstream_velocity_inertial[:, i])
            torque[:, i] = prop.torque(self.propeller_vectors[:,i], -thrust[2:,i])
        forces = thrust.sum(axis=1)
        torques = np.cross(self.propeller_vectors, thrust)
        return self.apply_forces_torques(forces, torques)


    def step(self):
        pass



def stateDerivative(x,u,physics=None,geometry=None):
    # Inputs: state vector (x), input vector (u)
    # Returns: time derivative of state vector (xdot)
    
    #  State Vector Reference:
    #idx  0, 1, 2, 3, 4, 5,  6,   7,   8,   9, 10, 11
    #x = [u, v, w, p, q, r, phi, the, psi, xE, yE, hE]
    
    # Store state variables in a readable format
    ub = x[0]
    vb = x[1]
    wb = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    phi = x[6]
    theta = x[7]
    psi = x[8]
    xE = x[9]
    yE = x[10]
    hE = x[11]
    
    # Calculate forces and moments from propeller inputs (u)
    F1 = fthrust(x, u[0],  dx,  dy)
    F2 = fthrust(x, u[1], -dx, -dy)
    F3 = fthrust(x, u[2],  dx, -dy)
    F4 = fthrust(x, u[3], -dx,  dy)
    Fz = F1 + F2 + F3 + F4
    L = (F2 + F3) * dy - (F1 + F4) * dy
    M = (F1 + F3) * dx - (F2 + F4) * dx
    N = -T(F1,dx,dy) - T(F2,dx,dy) + T(F3,dx,dy) + T(F4,dx,dy)
    
    # Pre-calculate trig values
    cphi = np.cos(phi);   sphi = np.sin(phi)
    cthe = np.cos(theta); sthe = np.sin(theta)
    cpsi = np.cos(psi);   spsi = np.sin(psi)
    
    # Calculate the derivative of the state matrix using EOM
    xdot = np.zeros(12)
    
    xdot[0] = -g * sthe + r * vb - q * wb  # = udot
    xdot[1] = g * sphi*cthe - r * ub + p * wb # = vdot
    xdot[2] = 1/m * (-Fz) + g*cphi*cthe + q * ub - p * vb # = wdot
    xdot[3] = 1/Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
    xdot[4] = 1/Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
    xdot[5] = 1/Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
    xdot[6] = p + (q*sphi + r*cphi) * sthe / cthe  # = phidot
    xdot[7] = q * cphi - r * sphi  # = thetadot
    xdot[8] = (q * sphi + r * cphi) / cthe  # = psidot
    
    xdot[9] = cthe*cpsi*ub + (-cphi*spsi + sphi*sthe*cpsi) * vb + \
        (sphi*spsi+cphi*sthe*cpsi) * wb  # = xEdot
        
    xdot[10] = cthe*spsi * ub + (cphi*cpsi+sphi*sthe*spsi) * vb + \
        (-sphi*cpsi+cphi*sthe*spsi) * wb # = yEdot
        
    xdot[11] = -1*(-sthe * ub + sphi*cthe * vb + cphi*cthe * wb) # = hEdot
    
    return xdot
