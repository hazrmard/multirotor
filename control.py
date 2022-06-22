from dataclasses import dataclass

import numpy as np

from .simulation import Multirotor



@dataclass
class PIDController:

    k_p: np.ndarray
    k_i: np.ndarray
    k_d: np.ndarray
    max_err_i: np.ndarray

    def __post_init__(self):
        self.err_p = np.zeros_like(self.k_p)
        self.err_i = np.zeros_like(self.k_i)
        self.err_d = np.zeros_like(self.k_d)
        self.err = 0.
        if self.max_err_i is None:
            self.max_err_i = np.inf


    def reset(self):
        self.err = 0.
        self.err_p *= 0
        self.err_i *= 0
        self.err_d *= 0


    def step(self, reference: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        err = reference - measurement
        self.err_p = err
        self.err_i = np.clip(self.err_i + err, a_min=-self.max_err_i, a_max=self.max_err_i)
        self.err_d = err - self.err
        self.err = err
        return self.k_p * self.err_p + self.k_i * self.err_i + self.k_d * self.err_d



@dataclass
class PosController(PIDController):

    vehicle: Multirotor
    max_tilt: float = np.pi / 15
    max_velocity: float = 7.0


    def __post_init__(self):
        self.k_p = np.ones(2) * np.asarray(self.k_p)
        self.k_i = np.ones(2) * np.asarray(self.k_i)
        self.k_d = np.ones(2) * np.asarray(self.k_d)
        
        self.err_p = np.zeros_like(self.k_p)
        self.err_i = np.zeros_like(self.k_i)
        self.err_d = np.zeros_like(self.k_d)
        self.err = 0.
        if self.max_err_i is None:
            self.max_err_i = np.inf


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
        # ref_vel_xy = ref_delta_xy / self.vehicle.simulation.dt
        # abs_max_vel = np.abs((ref_vel_xy / (np.linalg.norm(ref_vel_xy) + 1e-6)) * self.max_velocity)
        # ref_vel_xy = np.clip(ref_vel_xy, a_min=-abs_max_vel, a_max=abs_max_vel)
        # actual/measured velocity
        mea_delta_xy = mea_vel_xy = self.vehicle.velocity[:2]
        # desired pitch, roll
        ctrl = super().step(reference=ref_delta_xy, measurement=mea_delta_xy)
        # ctrl[0] -> x dir -> pitch -> forward
        # ctrl[1] -> y dir -> roll -> lateral
        ctrl[0:2] = np.clip(ctrl[0:2], a_min=-self.max_tilt, a_max=self.max_tilt)
        ctrl[1] *= -1 # +y motion requires negative roll
        return ctrl # desired pitch, roll



@dataclass
class AttController(PIDController):

    vehicle: Multirotor


    def step(self, reference, measurement):
        # desired change in orientation i.e. angular velocity
        ref_delta = reference - measurement
        # actual change in orientation
        mea_delta = self.vehicle.angular_rate
        # prescribed change in velocity i.e. angular acceleration
        ctrl = super().step(reference=ref_delta, measurement=mea_delta)
        # torque = moment of inertia . angular_acceleration
        return self.vehicle.params.inertia_matrix.dot(ctrl)


@dataclass
class AltController(PIDController):

    vehicle: Multirotor

    def step(self, reference, measurement):
            roll, pitch, yaw = self.vehicle.orientation
            # desired change in z i.e. velocity
            ref_delta_z = reference - measurement
            # actual change in z
            mea_delta_z = self.vehicle.velocity[2]
            # change in delta_z i.e. change in velocity i.e. acceleration
            ctrl = super().step(reference=ref_delta_z, measurement=mea_delta_z)
            # change in z-velocity i.e. acceleration
            ctrl = self.vehicle.params.mass * (
                    ctrl / (np.cos(roll) * np.cos(pitch))
                ) + \
                self.vehicle.weight
            return ctrl # thrust force


class Controller:

    def __init__(self, ctrl_p: PosController, ctrl_a: AttController, ctrl_z: AltController):
        self.ctrl_p = ctrl_p
        self.ctrl_a = ctrl_a
        self.ctrl_z = ctrl_z
        self.vehicle = self.ctrl_a.vehicle
        assert self.ctrl_a.vehicle is self.ctrl_p.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_z.vehicle, "Vehicle instances different."

    def step(self, reference, measurement=None):
        # x,y,z,yaw
        pitch_roll = self.ctrl_p.step(reference[:2], self.vehicle.position[:2])
        ref_orientation = np.asarray([pitch_roll[1], pitch_roll[0], reference[3]])
        torques = self.ctrl_a.step(ref_orientation, self.vehicle.orientation)
        thrust = self.ctrl_z.step(reference[2], self.vehicle.position[2])
        return np.asarray([thrust, *torques])
