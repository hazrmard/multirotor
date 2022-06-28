from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .simulation import Multirotor
from .coords import euler_to_angular_rate



@dataclass
class PIDController:
    """
    Proportional Integral Derivative controller. Tracks a reference signal and outputs
    a control signal to minimize output. The equation is:

        err = reference - measurement

        u = k_p * err + k_d * d(err)/dt + k_i * int(err . dt)
    """

    k_p: np.ndarray
    "Proportional constant"
    k_i: np.ndarray
    "Integral constant"
    k_d: np.ndarray
    "Derivative constant"
    max_err_i: np.ndarray
    "Maximum accumulated error"
    dt: float
    "Simulation parameters to set timestep"

    def __post_init__(self):
        self.err_p = np.zeros_like(self.k_p)
        self.err_i = np.zeros_like(self.k_i)
        self.err_d = np.zeros_like(self.k_d)
        self.err = np.zeros_like(self.k_p)
        if self.max_err_i is None:
            self.max_err_i = np.inf


    def reset(self):
        self.err *= 0
        self.err_p *= 0
        self.err_i *= 0
        self.err_d *= 0


    def step(self, reference: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        err = reference - measurement
        self.err_p = err
        self.err_i = np.clip(
            self.err_i + trapezoid((self.err, err), dx=self.dt, axis=0),
            a_min=-self.max_err_i, a_max=self.max_err_i
        )
        self.err_d = (err - self.err) / self.dt
        self.err = err
        return self.k_p * self.err_p + self.k_i * self.err_i + self.k_d * self.err_d



@dataclass
class PosController(PIDController):
    """
    Position controller. Takes reference x/y position and outputs reference 
    pitch and roll angles for x and y motion, respectively.

    Uses vector from current to reference position as an approximation of 
    reference velocity. Compares against measured velocity. The deficit is used
    to change pitch and roll angles to increase and decrease velocity.

    Able to limit maximum velocity and tilt angles when tracking reference waypoints.
    """

    vehicle: Multirotor
    max_tilt: float = np.pi / 15
    "Maximum tilt angle in radians"
    max_velocity: float = 1.0
    "Maximum velocity in m/s"



    def __post_init__(self):
        self.k_p = np.ones(2) * np.asarray(self.k_p)
        self.k_i = np.ones(2) * np.asarray(self.k_i)
        self.k_d = np.ones(2) * np.asarray(self.k_d)
        super().__post_init__()


    def step(self, reference, measurement):
        roll, pitch, yaw = self.vehicle.orientation
        delta_x, delta_y = reference - measurement
        rot = np.asarray([
            [np.cos(yaw),   np.sin(yaw)],
            [-np.sin(yaw),  np.cos(yaw)],
        ])
        # convert reference x/y to body frame, given yaw
        # Using rotation matrix. For a positive yaw, the target x,y will appear
        # desired/reference change in x/y i.e. velocity
        ref_delta_xy = rot @ np.asarray([delta_x, delta_y])
        # TODO: Explicitly track velocity, instead of deltas
        ref_vel_xy = ref_delta_xy / self.vehicle.simulation.dt
        abs_max_vel = np.abs((ref_vel_xy / (np.linalg.norm(ref_vel_xy) + 1e-6)) * self.max_velocity)
        ref_vel_xy = np.clip(ref_vel_xy, a_min=-abs_max_vel, a_max=abs_max_vel)
        # actual/measured velocity
        mea_delta_xy = mea_vel_xy = self.vehicle.velocity[:2]
        # desired pitch, roll
        ctrl = super().step(reference=ref_vel_xy, measurement=mea_delta_xy)
        # ctrl[0] -> x dir -> pitch -> forward
        # ctrl[1] -> y dir -> roll -> lateral
        ctrl[0:2] = np.clip(ctrl[0:2], a_min=-self.max_tilt, a_max=self.max_tilt)
        ctrl[1] *= -1 # +y motion requires negative roll
        return ctrl # desired pitch, roll



@dataclass
class AttController(PIDController):
    """
    Attitude controller. Tracks reference roll, pitch, yaw angles and outputs
    the necessary moments about each x,y,z axes to achieve them.

    Uses change in orientation from measured to reference as approximate reference
    angular rate. Compares against measured angular rate. Outputs required change
    in angular rate (angular acceleration) as moments.
    """

    vehicle: Multirotor


    def __post_init__(self):
        self.k_p = np.ones(3) * np.asarray(self.k_p)
        self.k_i = np.ones(3) * np.asarray(self.k_i)
        self.k_d = np.ones(3) * np.asarray(self.k_d)
        super().__post_init__()


    def step(self, reference, measurement):
        # desired change in orientation i.e. angular velocity
        ref_delta = euler_to_angular_rate(reference - measurement, self.vehicle.orientation)
        # actual change in orientation
        mea_delta = self.vehicle.angular_rate
        # prescribed change in velocity i.e. angular acceleration
        ctrl = super().step(reference=ref_delta, measurement=mea_delta)
        # torque = moment of inertia . angular_acceleration
        return self.vehicle.params.inertia_matrix.dot(ctrl)



@dataclass
class AltController(PIDController):
    """
    Altitude Controller. Tracks z-position and outputs thrust force needed.

    Uses change in z-position as approximate vertical velocity. Compares against
    measured velocity. Outputs the change in velocity (acceleration) as thrust force,
    given orientation of vehicle.
    """

    vehicle: Multirotor

    def step(self, reference, measurement):
            roll, pitch, yaw = self.vehicle.orientation
            # desired change in z i.e. velocity
            ref_delta_z = reference - measurement
            # actual change in z
            mea_delta_z = self.vehicle.world_velocity[2]
            # change in delta_z i.e. change in velocity i.e. acceleration
            ctrl = super().step(reference=ref_delta_z, measurement=mea_delta_z)
            # change in z-velocity i.e. acceleration
            ctrl = self.vehicle.params.mass * (
                    ctrl / (np.cos(roll) * np.cos(pitch))
                ) + \
                self.vehicle.weight
            return ctrl # thrust force



class Controller:
    """
    The cascaded PID controller. Tracks position and yaw, and outputs thrust and
    moments needed.

        (x,y) --> Position Ctrl --> (Angles) --> Attitude Ctrl --> (Moments)
        (z)   --> Attitude Ctrl --> (Forces)
    """

    def __init__(self, ctrl_p: PosController, ctrl_a: AttController, ctrl_z: AltController):
        self.ctrl_p = ctrl_p
        self.ctrl_a = ctrl_a
        self.ctrl_z = ctrl_z
        self.vehicle = self.ctrl_a.vehicle
        assert self.ctrl_a.vehicle is self.ctrl_p.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_z.vehicle, "Vehicle instances different."


    def reset(self):
        self.ctrl_a.reset()
        self.ctrl_p.reset()
        self.ctrl_z.reset()


    def step(self, reference, measurement=None):
        # x,y,z,yaw
        pitch_roll = self.ctrl_p.step(reference[:2], self.vehicle.position[:2])
        ref_orientation = np.asarray([pitch_roll[1], pitch_roll[0], reference[3]])
        torques = self.ctrl_a.step(ref_orientation, self.vehicle.orientation)
        thrust = self.ctrl_z.step(reference[2], self.vehicle.position[2])
        return np.asarray([thrust, *torques])
