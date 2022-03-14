from dataclasses import dataclass

import numpy as np

from simulation import Multirotor



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
        if self.max_err_i is None:
            self.max_err_i = np.inf


    def reset(self):
        self.err_p *= 0
        self.err_i *= 0
        self.err_d *= 0


    def step(self, reference: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        self.err = err = reference - measurement
        self.err_p = err
        self.err_i = np.clip(self.err_i + err, a_min=-self.max_err_i, a_max=self.max_err_i)
        self.err_d = err - self.err_d
        return (self.k_p * self.err_p + self.k_i * self.err_i + self.k_d * self.err_d)



@dataclass
class PositionController(PIDController):

    vehicle: Multirotor


    def step(self, reference, measurement):
        ctrl = super().step(reference=reference, measurement=measurement)
        # ctrl[0] -> x dir -> pitch -> forward
        # ctrl[1] -> y dir -> roll -> lateral
        # ctrl[2] -> z dir -> thrust -> vertical
        roll, pitch, yaw = self.vehicle.orientation
        ctrl[2] = self.vehicle.params.mass * (
                ctrl[2] / (np.cos(roll) * np.cos(pitch))
            ) + \
            self.vehicle.weight
        return ctrl


AttitudeController = PIDController



class PosAttController:


    def __init__(self, ctrl_p: PositionController, ctrl_a: AttitudeController):
        self.ctrl_p = ctrl_p
        self.ctrl_a = ctrl_a


    def reset(self):
        self.ctrl_a.reset()
        self.ctrl_p.reset()


    def step(self, ref_pos, pos, ref_yaw, att):
        pitch, roll, thrust = self.ctrl_p.step(ref_pos, pos)
        pitch, roll = np.clip((pitch, roll), a_min=-np.pi/2, a_max=np.pi/2)
        ref_att = np.asarray([roll, pitch, ref_yaw])
        torques = self.ctrl_a.step(ref_att, att)
        return thrust, torques

