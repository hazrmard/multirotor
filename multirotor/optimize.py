import os
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, Union
from pprint import pprint as print
import pickle

import numpy as np
import optuna
from .simulation import Multirotor
from .trajectories import Trajectory
from .vehicle import VehicleParams, SimulationParams
from .helpers import DataLog
from .env import DynamicsMultirotorEnv
from .controller import (
    AltController, AltRateController,
    PosController, AttController,
    VelController, RateController,
    Controller,
    SCurveController
)


DEFAULTS = Namespace(
    bounding_box = 20,
    max_velocity = 5,
    max_acceleration = 2.5,
    max_tilt = np.pi/12,
    leashing = True,
    sqrt_scaling = True,
    use_yaw = True,
    num_sims = 10,
    scurve = False,
    max_err_i_attitude = 0.1,
    max_err_i_rate = 0.1,
)

def get_study(study_name: str=None, seed: int=0) -> optuna.Study:
    """
    Create an `optuna.Study` set to maximize reward.

    Parameters
    ----------
    study_name : str, optional
        If name is given, a study is created in `./studies/study_name.db`, by default None
    seed : int, optional
        The random seed to use for parameter sampling, by default 0

    Returns
    -------
    optuna.Study
        The study object. Best params can be accessed via `study.best_params`.
    """
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, direction='maximize',
        storage=storage_name if study_name is not None else None,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    return study



def run_sim(
    env, traj: Trajectory,
    ctrl: Controller
) -> DataLog:
    """
    Run a single episode, where the environment is controlled by the `Controller`
    object. The `env` object must be `reset()` before calling.

    Parameters
    ----------
    env : DynamicsMultirotorEnv
        The environment to run
    traj : Trajectory
        The waypoints to follow.
    ctrl : Controller
        The controller which outputs dynamics to be fed to the environment.

    Returns
    -------
    DataLog
        A log containing state measurements for each step.
    """
    log = DataLog(env.vehicle, ctrl,
                  other_vars=('reward',))
        
    for i, (pos, feed_forward_vel) in enumerate(traj):
        # Get prescribed normalized action for system as thrust and torques
        ref = np.asarray([*pos, 0], env.vehicle.dtype)
        action = ctrl.step(reference=ref, feed_forward_velocity=feed_forward_vel)
        # Send speeds to environment
        dynamics = np.asarray([0,0,*action], env.vehicle.dtype) # (Fx, Fy, Fz, Tx, Ty, Tz)
        state, r, done, *_ = env.step(dynamics)
        log.log(reward=r)
        if done:
            break

    log.done_logging()
    return log



def get_controller(m: Multirotor, scurve=False, args: Namespace=DEFAULTS) -> Controller:
    """
    Create a `multirotor.controller.Controller` object with default values.

    Note: `max_acceleration`, `max_velocity`, `max_error_i` etc are important
    parameters that are set as defaults.

    Parameters
    ----------
    m : Multirotor
        The vehicle to control.
    scurve : bool, optional
        Whether to use an `SCurve` trajectory planning approach, by default False
    args : Namespace, optional
        The default settings, by default DEFAULTS

    Returns
    -------
    Controller
    """
    assert m.simulation.dt <= 0.1, 'Simulation time step too large.'
    pos = PosController( # PD
        0.8, 0., 3.75,
        max_err_i=args.max_velocity, vehicle=m,
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        square_root_scaling=args.sqrt_scaling,
        leashing=args.leashing
    )
    vel = VelController( # P
        1, 0., 0,
        max_err_i=args.max_acceleration,
        vehicle=m,
        max_tilt=args.max_tilt)
    att = AttController( # P
        [1., 1., 0.], 0, 0., # yaw param is set to 0, in case use_yaw=False
        max_err_i=args.max_err_i_attitude, vehicle=m)
    rat = RateController( # PD
        [4, 4, 0], 0, [40, 40, 0], # yaw param is set to 0, in case use_yaw=False
        max_err_i=args.max_err_i_rate,
        vehicle=m)

    alt = AltController(
        1, 0, 0,
        max_err_i=args.max_velocity, vehicle=m,
        max_velocity=args.max_velocity)
    alt_rate = AltRateController(
        10, 0, 0,
        max_err_i=args.max_acceleration, vehicle=m)

    ctrl = Controller(
        pos, vel, att, rat, alt, alt_rate,
        period_p=0.1, period_a=0.01, period_z=0.1,
    )
    if scurve:
        return SCurveController(ctrl)
    return ctrl



def make_controller_from_trial(trial: optuna.Trial, args: Namespace=DEFAULTS, prefix='') -> dict:
    """
    Makes a dictionary of parameters which can be used with the `Controller.set_params()`

    Parameters
    ----------
    trial : optuna.Trial
        The trail with the suggest parameters.
    args : Namespace, optional
        Namespace of optimization arguments, by default DEFAULTS
    prefix : str, optional
        _description_, by default ''

    Returns
    -------
    Dict
        A dictionary of parameters to be used by `Controller.set_params(**dict)`
    """
    r_pitch_roll_p = trial.suggest_float(prefix + 'r_pitch_roll.k_p', 0.1, 50)
    r_pitch_roll_i = trial.suggest_float(prefix + 'r_pitch_roll.k_i', 0.1, 10)
    r_pitch_roll_d = trial.suggest_float(prefix + 'r_pitch_roll.k_d', 1, 250)
    r_pitch_roll_max_acc = trial.suggest_float(prefix + 'r_pitch_roll.max_acceleration', 0.1, 25)
    if args.use_yaw:
        r_yaw_p = trial.suggest_float(prefix + 'r_yaw.k_p', 0.1, 5)
        r_yaw_i = trial.suggest_float(prefix + 'r_yaw.k_i', 0.1, 1)
        r_yaw_d = trial.suggest_float(prefix + 'r_yaw.k_d', 1, 25)
        r_yaw_max_acc = trial.suggest_float(prefix + 'r_yaw.max_acceleration', 0.1, 5)
    else:
        r_yaw_p, r_yaw_i, r_yaw_d, r_yaw_max_acc = 0, 0, 0, 0

    a_pitch_roll_p = trial.suggest_float(prefix + 'a_pitch_roll.k_p', 0.1, 50)
    a_pitch_roll_i = trial.suggest_float(prefix + 'a_pitch_roll.k_i', 0.1, 10)
    a_pitch_roll_d = trial.suggest_float(prefix + 'a_pitch_roll.k_d', 1, 250)
    if args.use_yaw:
        a_yaw_p = trial.suggest_float(prefix + 'a_yaw.k_p', 0.1, 5)
        a_yaw_i = trial.suggest_float(prefix + 'a_yaw.k_i', 0.1, 1)
        a_yaw_d = trial.suggest_float(prefix + 'a_yaw.k_d', 1, 25)
    else:
        a_yaw_p, a_yaw_i, a_yaw_d = 0, 0, 0

    params = dict(
        ctrl_p = dict(
            k_p = trial.suggest_float(prefix + 'p.k_p', 0.1, 50),
            k_i = trial.suggest_float(prefix + 'p.k_i', 0.1, 10),
            k_d = trial.suggest_float(prefix + 'p.k_d', 1, 250),
        ),
        ctrl_v = dict(
            k_p = trial.suggest_float(prefix + 'v.k_p', 0.1, 50),
            k_i = trial.suggest_float(prefix + 'v.k_i', 0.1, 10),
            k_d = trial.suggest_float(prefix + 'v.k_d', 1, 250),
        ),
        ctrl_a = dict(
            k_p = np.asarray((a_pitch_roll_p, a_pitch_roll_p, a_yaw_p)),
            k_i = np.asarray((a_pitch_roll_i, a_pitch_roll_i, a_yaw_i)),
            k_d = np.asarray((a_pitch_roll_d, a_pitch_roll_d, a_yaw_d)),
        ),
        ctrl_r = dict(
            k_p = np.asarray((r_pitch_roll_p, r_pitch_roll_p, r_yaw_p)),
            k_i = np.asarray((r_pitch_roll_i, r_pitch_roll_i, r_yaw_i)),
            k_d = np.asarray((r_pitch_roll_d, r_pitch_roll_d, r_yaw_d)),
            max_acceleration = np.asarray((r_pitch_roll_max_acc, r_pitch_roll_max_acc, r_yaw_max_acc)),
            max_err_i = np.asarray((r_pitch_roll_max_acc, r_pitch_roll_max_acc, r_yaw_max_acc)),
        ),
        ctrl_z = dict(
            k_p = trial.suggest_float('z.k_p', 0.1, 50),
            k_i = trial.suggest_float('z.k_i', 0.1, 10),
            k_d = trial.suggest_float('z.k_d', 1, 250),
        ),
        ctrl_vz = dict(
            k_p = trial.suggest_float('vz.k_p', 0.1, 50),
            k_i = trial.suggest_float('vz.k_i', 0.1, 10),
            k_d = trial.suggest_float('vz.k_d', 1, 250),
        ),
    )
    if args.scurve:
        params.update({prefix + 'feedforward_weight': trial.suggest_float(prefix + 'feedforward_weight', 0.1, 1.0, step=0.1)})
    return params



def make_env(vp: VehicleParams, sp: SimulationParams, args: Namespace=DEFAULTS) -> DynamicsMultirotorEnv:
    """
    Make the environment instance to be used by `objective()`, and set default
    env parameters.

    Parameters
    ----------
    vp : VehicleParams
    sp : SimulationParams
    args : Namespace, optional
        Optimization params, by default DEFAULTS

    Returns
    -------
    DynamicsMultirotorEnv
    """
    env = DynamicsMultirotorEnv(Multirotor(vp, sp), allocate=True)
    env.bounding_box = args.bounding_box
    return env



def make_objective(vp: VehicleParams, sp: SimulationParams, args: Namespace=DEFAULTS) -> Callable:
    """
    Make the function that `optuna` will use to optimize.

    Parameters
    ----------
    vp : VehicleParams
    sp : SimulationParams
    args : Namespace, optional
        Optimization params, by default DEFAULTS
    """
    def objective(trial: optuna.Trial):
        # objective is to navigate to origin from an intial position
        ctrl_params = make_controller_from_trial(trial=trial, args=args)
        env = make_env(vp, sp, args)
        ctrl = get_controller(env.vehicle, args.scurve, args)
        ctrl.set_params(**ctrl_params)
        errs = []
        for i in range(args.num_sims):
            env.reset()
            ctrl.reset()
            waypoints = np.asarray([[0,0,0]])
            traj = Trajectory(env.vehicle, waypoints, proximity=0.1)
            log = run_sim(env, traj, ctrl)
            errs.append(log.reward.sum())
        return np.mean(errs)
    return objective



def optimize(
    vp: VehicleParams, sp: SimulationParams,
    ntrials: int=1000,
    args: Namespace=DEFAULTS, seed: int=0,
    study_name: str=None,
    verbosity=optuna.logging.WARNING
) -> optuna.Study:
    """
    Search over parameter space to optimize controller parameters in `make_controller_from_trial()`.

    Parameters
    ----------
    vp : VehicleParams
    sp : SimulationParams
    ntrials : int, optional
        Number of trials, by default 1000
    args : Namespace, optional
        Optimization params, by default DEFAULTS
    seed : int, optional
       Optimization seed, by default 0
    study_name : str, optional
        Name of study to be saved in `studies/study_name.db`, by default None
    verbosity : _type_, optional
        Logging level of `optuna.optimize()`, by default optuna.logging.WARNING

    Returns
    -------
    optuna.Study
        The completed `Study` object with `study.best_params` which can be used
        by `apply_params(Controller, **best_params)` to get the optimized
        controller.
    """
    optuna.logging.set_verbosity(verbosity)
    study = get_study(study_name, seed=seed)
    study.optimize(make_objective(vp, sp, args), n_trials=ntrials, show_progress_bar=True)
    return study



def apply_params(ctrl: Controller, params: dict) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply parameters from an optuna.Study to a `Controller`. Converts the dictionary
    of `best_params` in the format accepted by `controller.set_params()`

    Parameters
    ----------
    ctrl : Controller
        The controller to which to apply the parameters. If None, just return the
        dictionary of parameters which `Controller.set_params()` can use.
    params : dict[str, np.ndarray]
        The dictionary of parameters returned by `optuna.Study`

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A nested dictionary for `Controller.set_params()`
    """
    p = dict(ctrl_p={}, ctrl_v={}, ctrl_a={}, ctrl_r={}, ctrl_z={}, ctrl_vz={})
    for name, param in params.items():
        if '.' in name:
            pre, post = name.split('.')
            # skip parameters such as r_pitch_roll.*/r_yaw.*, which need to be combined together
            if pre.startswith('r_p') or pre.startswith('a_p') or pre.startswith('r_y') or pre.startswith('a_y'):
                continue
            p['ctrl_' + pre][post] = param
        else:
            p[name] = param
    # special case for rate controller with differenr pitch/roll, and yaw params
    # if r_yaw or a_yaw are not present, assume that yaw was not being controlled, and
    # set yaw params to 0.
    p['ctrl_r']['k_p'] = np.asarray([params['r_pitch_roll.k_p'], params['r_pitch_roll.k_p'], params.get('r_yaw.k_p', 0)])
    p['ctrl_r']['k_i'] = np.asarray([params['r_pitch_roll.k_i'], params['r_pitch_roll.k_i'], params.get('r_yaw.k_i', 0)])
    p['ctrl_r']['k_d'] = np.asarray([params['r_pitch_roll.k_d'], params['r_pitch_roll.k_d'], params.get('r_yaw.k_d', 0)])
    p['ctrl_r']['max_acceleration'] = np.asarray([params['r_pitch_roll.max_acceleration'],
                                                  params['r_pitch_roll.max_acceleration'],
                                                  params.get('r_yaw.max_acceleration', 0)])
    p['ctrl_a']['k_p'] = np.asarray([params['a_pitch_roll.k_p'], params['a_pitch_roll.k_p'], params.get('a_yaw.k_p', 0)])
    p['ctrl_a']['k_i'] = np.asarray([params['a_pitch_roll.k_i'], params['a_pitch_roll.k_i'], params.get('a_yaw.k_i', 0)])
    p['ctrl_a']['k_d'] = np.asarray([params['a_pitch_roll.k_d'], params['a_pitch_roll.k_d'], params.get('a_yaw.k_d', 0)])
    if ctrl is not None:
        ctrl.set_params(**p)
    return p



if __name__=='__main__':
    raise NotImplementedError('Use the optimize() function directly.')
    parser = ArgumentParser()
    parser.add_argument('study_name', help='Name of study', default=DEFAULTS.study_name, type=str, nargs='?')
    # parser.add_argument('--nprocs', help='Number of processes.', default=DEFAULTS.nprocs, type=int)
    parser.add_argument('--ntrials', help='Number of trials.', default=DEFAULTS.ntrials, type=int)
    parser.add_argument('--max_velocity', default=DEFAULTS.max_velocity, type=float)
    parser.add_argument('--max_acceleration', default=DEFAULTS.max_acceleration, type=float)
    parser.add_argument('--max_tilt', default=DEFAULTS.max_tilt, type=float)
    parser.add_argument('--leashing', action='store_true', default=DEFAULTS.leashing)
    parser.add_argument('--sqrt_scaling', action='store_true', default=DEFAULTS.sqrt_scaling)
    parser.add_argument('--use_yaw', action='store_true', default=DEFAULTS.use_yaw)
    parser.add_argument('--bounding_box', default=DEFAULTS.bounding_box, type=float)
    parser.add_argument('--num_sims', default=DEFAULTS.num_sims, type=int)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--pid_params', help='File to save pid params to.', type=str, default='')
    parser.add_argument('--comment', help='Comments to attach to studdy.', type=str, default='')
    args = parser.parse_args()

    if not args.append:
        try:
            os.remove(('studies/' + args.study_name + '.db'))
        except OSError:
            pass
    
    # create study if it doesn't exist. The study will be reused with a new seed
    # by each process
    study = get_study(args.study_name)

    for key in vars(args):
        study.set_user_attr(key, getattr(args, key))

    with mp.Pool(args.nprocs) as pool:
        pool.starmap(optimize, [(args, i) for i in range(args.nprocs)])
    print(study.best_trial.number)
    print(study.best_params)
    if len(args.pid_params) > 0:
        with open(args.pid_params + ('.pickle' if not args.pid_params.endswith('pickle') else ''), 'wb') as f:
            pickle.dump(apply_params(None, **study.best_params), f)
