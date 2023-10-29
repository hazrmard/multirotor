import numpy as np
from numba import njit
from typing import Dict, Union

from .pid import Controller
from ..helpers import get_vehicle_ability
try:
    from pyscurve import ScurvePlanner
    from pyscurve.scurve import PlanningError
except ImportError:
    ScurvePlanner = False
    PlanningError = Exception



class SCurveController:

    def __init__(self, ctrl: Controller):
        if not ScurvePlanner:
            raise ImportError('Py-Scurve not installed')
        self.ctrl = ctrl
        self.vehicle = self.ctrl.vehicle
        self.ctrl_p = self.ctrl.ctrl_p
        self.ctrl_v = self.ctrl.ctrl_v
        self.ctrl_a = self.ctrl.ctrl_a
        self.ctrl_r = self.ctrl.ctrl_r
        self.ctrl_z = self.ctrl.ctrl_z
        self.ctrl_vz = self.ctrl.ctrl_vz
        self.reset()


    def get_params(self):
        p = dict(
            steps=self.steps,
            max_velocity=self.max_velocity, max_acceleration=self.max_acceleration,
            max_jerk=self.max_jerk,
        )
        p.update(ctrl=self.ctrl.get_params())
        return p


    def set_params(self,**params: Dict[str, Union[np.ndarray, bool, float, int]]):
        ctrl_params = params.get('ctrl', None)
        # if controller params are not nested under 'ctrl, make a dict
        # containing those params...
        if ctrl_params is None:
            # ...using keys from self.ctrl
            ctrls = self.ctrl.get_params().keys()
            ctrl_params = {name: params.get(name, {}) for name in ctrls}
            # delete those param names from the params dict. The remaning
            # params are for this (self) controller
            for name in ctrls:
                if name in params:
                    del params[name]
        else:
            del params['ctrl']
        self.ctrl.set_params(**ctrl_params)

        for name, value in params.items():
            if hasattr(self, name):
                setattr(self, name, value)
        # these parameters are dictated by the controller
        self.max_velocity = self.ctrl.ctrl_p.max_velocity
        self.steps = self.ctrl.steps_p


    def reset(self):
        self.ctrl.reset()
        # these parameters are dictated by the controller
        self.steps = self.ctrl.steps_p
        self.max_velocity = self.ctrl.ctrl_p.max_velocity

        self.max_acceleration = get_vehicle_ability(
            self.vehicle.params, self.vehicle.simulation,
            self.ctrl_v.max_tilt, self.ctrl_r.max_acceleration,
            max_rads=700.
        )['max_acc_xy']
        self.max_jerk = self.max_acceleration # TODO: arbitrary
        # max accelerateion is determined from the physical properties of the vehicle
        self.ctrl_p.max_acceleration = self.max_acceleration
        self.ctrl_p.max_jerk = self.max_jerk

        self.planner = ScurvePlanner()
        self.n = 0
        self.n_since_replan = 0
        self.ref_xy = np.empty(2, self.ctrl.vehicle.dtype)


    def step(self, reference: np.ndarray, ref_is_error=False):
        ref_xy = reference[:2]
        if self.n==0 or not np.array_equal(ref_xy, self.ref_xy):
            self.ref_xy = ref_xy
            try:
                self.traj = self.planner.plan_trajectory(
                    q0=self.ctrl.vehicle.position[:2],
                    q1=ref_xy,
                    v0=min(
                        np.linalg.norm(self.max_velocity),
                        np.linalg.norm(self.ctrl.vehicle.velocity[:2])
                    ) * (self.ctrl.vehicle.velocity[:2] / np.linalg.norm(self.ctrl.vehicle.velocity[:2])),
                    v1=(ref_xy-self.ctrl.vehicle.position[:2]) * self.max_velocity \
                        / np.linalg.norm(ref_xy-self.ctrl.vehicle.position[:2]),
                    v_max=self.max_velocity,
                    a_max=self.max_acceleration,
                    j_max=self.max_jerk
                )
                self.n_since_replan = 0
            except PlanningError:
                pass
        if self.n % self.steps == 0:
            target = self.traj((self.n_since_replan + self.steps) * self.ctrl.vehicle.simulation.dt)
            # point = target[:, 2]
            self._ref_vel = target[:, 1]
            # self._ref = np.concatenate((point, state[2:4])) # position, yaw
        self.ctrl.step(reference, ref_is_error=False,
                       feed_forward_velocity=self._ref_vel)
        self.n += 1
        self.n_since_replan += 1
        return self.ctrl.action


    @property
    def action(self):
        return self.ctrl.action
    @property
    def reference(self):
        return self.ctrl.reference
    @property
    def feedforward_weight(self):
        return self.ctrl.feedforward_weight




def acc_1d(amax: float, vmax: float, v0: float, v1: float, disp: float, dt: float=1e-2):
    """Solve 1d kinematics problem"""
    sign = np.sign(disp)
    dist = np.abs(disp)
    relvel = v1 - v0 # desired change in velocity
    # s = ut + 0.5 a(t) t^2 => 2 (s - ut) / t^2 = a(t)
    # 2 a(t) s = v^2 - u^2
    # distances to accelerate and decelerate from max vel to current
    # and target velocities
    dist_v0_vmax = np.abs((vmax**2 - v0**2 ) / (2 * amax))
    dist_vmax_v1 = np.abs((vmax**2 - v1**2 ) / (2 * amax))
    dist_0_v1 = (v1**2) / (2 * amax)
    dist_vmax_0 = (vmax**2) / (2 * amax)

    if dist > dist_vmax_v1:
        pass
    else:
        vmax = np.sqrt(((2 * amax * dist) + v0**2 + v1**2) / 2)

    if v0 < vmax:
        should_decelerate = False
    elif v0 > vmax:
        should_decelerate = True
    else:
        return 0
    amax = np.clip(np.abs((v0 - vmax)) / dt, 0, amax)

    if should_decelerate:
        return -sign * amax
    return sign * amax



# @njit
def _acc_1d(amax: float, vmax: float, v0: float, v1: float, disp: float):
    """Solve 1d kinematics problem"""
    sign = np.sign(disp)
    dist = np.abs(disp)
    relvel = v1 - v0 # desired change in velocity
    # s = ut + 0.5 a(t) t^2 => 2 (s - ut) / t^2 = a(t)
    # 2 a(t) s = v^2 - u^2
    # distances to accelerate and decelerate from max vel to current
    # and target velocities
    dist_v0_vmax = np.abs((vmax**2 - v0**2 ) / (2 * amax))
    dist_vmax_v1 = np.abs((vmax**2 - v1**2 ) / (2 * amax))
    dist_0_v1 = (v1**2) / (2 * amax)
    dist_vmax_0 = (vmax**2) / (2 * amax)
    # if at relative rest
    if relvel==0:
        # if self.log: print('Relative rest')
        if dist > 0:
            # if self.log: print('Accelerate. Distance > 0')
            should_decelerate = False
            amax = np.sqrt(v0**2 / (2 * dist))
        else:
            # if self.log: print('No change. Distance == 0')
            should_decelerate = False # doesn't matter, since amax=0
            amax = 0
    # if approaching. For e.g. v0=1ms, disp=10
    elif (np.sign(v0) == sign):
        # if self.log: print('Approaching')
        # moving in opposite directions to desired velocity
        # need to decelerate to 0 and reverse to match velocity at some point, except:
        # if any one velocity is 0, and they're approaching, it's like catch-up
        if (np.sign(v0) != np.sign(v1)) and (v1!=0 and v0!=0):
            # distance needed to overshoot enough such that can reverse to the 
            # currect velocity:
            dist_net = dist + dist_0_v1
            # distance to halt from max vel and accelerate to v1, changing direction
            # disp_net = dist_vmax_0 - dist_0_v1
            # If close enough that need to stop and reverse
            if dist_net > dist_v0_vmax + dist_vmax_0:
                should_decelerate = False
            else:
                should_decelerate = True
                # 2 a s = - v0^2
                amax = np.clip(np.sqrt(v0**2 / (2 * dist)), 0, amax)
        # moving in same direction (catching up)
        else:
            # if self.log: print('catch-up')
            # If far enough that can decelerate from vmax to match velocity, then
            # keep accelerating to catch up
            if dist >= dist_vmax_v1:
                # if self.log: print('Accelerate. Distance > vmax->v1 > v0->v1')
                should_decelerate = False
            else:
                # check if actually at vmax, then decelerate
                if abs(v0)==vmax:
                    # if self.log: print('Decelerate. Distance < vmax->v1')
                    should_decelerate = True
                # else if v0 < vmax and v0 > v1, but close enough not to decelerate from vmax,
                # recalculate acceleration
                else:
                    # calculate peak velocity to accelerate to such that can acceleate
                    # and decelerate at amax and cover the distance
                    v_m = np.sqrt((2 * amax * dist + v0**2 + v1**2) / 2)
                    should_decelerate = abs(v0) > v_m
                    # if self.log:
                    #     v1 = v_m
                    #     dec = 'Decelerating' if should_decelerate else 'Accelerating'
                    #     print(dec, 'from %.2f to %.2f at acc value %.2f' % (v0, v1, amax))
    # if receding
    else:
        # if self.log: print('Receding. Accelerate')
        should_decelerate = False

    if should_decelerate:
        return -sign * amax
    return sign * amax



# @njit
def acc_to_target_3d(amax: np.ndarray, vmax: np.ndarray, v0_vec: np.ndarray, v1_vec: np.ndarray, disp: float, dt: float=1e-2):
    """Solve 3d kinematics problem"""
    acc_x = acc_1d(amax[0], vmax[0], v0_vec[0], v1_vec[0], disp[0], dt)
    acc_y = acc_1d(amax[1], vmax[1], v0_vec[1], v1_vec[1], disp[1], dt)
    acc_z = acc_1d(amax[2], vmax[2], v0_vec[2], v1_vec[2], disp[2], dt)
    acc_xyz = np.asarray([acc_x, acc_y, acc_z])
    acc = acc_xyz * amax / (np.linalg.norm(acc_xyz) + 1e-6)
    return acc
# @njit
def acc_to_target_2d(amax: float, vmax: float, v0_vec: np.ndarray, v1_vec: np.ndarray, disp: float, dt: float=1e-2):
    """Solve 2d kinematics problem"""
    acc_x = acc_1d(amax, vmax, v0_vec[0], v1_vec[0], disp[0], dt)
    acc_y = acc_1d(amax, vmax, v0_vec[1], v1_vec[1], disp[1], dt)
    acc_xyz = np.asarray([acc_x, acc_y])
    acc = acc_xyz * amax / (np.linalg.norm(acc_xyz) + 1e-6)
    return acc



# @njit
def compute_accel_2d(
    amax: float,
    vmax: float,
    from_position : np.ndarray,
    from_velocity : np.ndarray,
    to_position : np.ndarray,
    to_velocity : np.ndarray,
    dt: float=1e-2
    ) -> np.ndarray:
    target_disp = to_position - from_position
    acc = acc_to_target_2d(amax, vmax, from_velocity, to_velocity, target_disp, dt)
    # if self.log:
    #     print('Target disp', target_disp)
    #     print('Target acc', acc)
    return acc