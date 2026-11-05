import copy
from argparse import Namespace
from typing import Callable, Dict, Tuple

import numpy as np
import optuna
from .fleet import Fleet
from .simulation import Multirotor
from .trajectories import Trajectory
from .vehicle import VehicleParams, SimulationParams
from .helpers import DataLog
from .env import DynamicsMultirotorEnv
from .controller import (
    AltController, AltRateController,
    PosController, AttController,
    VelController, RateController,
    Controller, FleetController,
    SCurveController
)


DEFAULTS = Namespace(
    bounding_box = 20,
    max_velocity = 5,
    max_acceleration = 2.5,
    # Below `BaseMultirotorEnv.max_angle`, the tilt at which the environments end
    # an episode, so a controller that saturates does not tip the vehicle.
    max_tilt = np.pi/18,
    max_angular_acceleration = 20.,
    leashing = True,
    sqrt_scaling = True,
    use_yaw = True,
    num_sims = 60,          # scenarios flown by each trial
    num_heldout_sims = 150, # scenarios that score the best parameters at the end
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
        
    for pos, feed_forward_vel in traj:
        # Get prescribed normalized action for system as thrust and torques
        ref = np.asarray([*pos, 0], env.vehicle.dtype)
        action = ctrl.step(reference=ref, feed_forward_velocity=feed_forward_vel)
        # Send speeds to environment
        dynamics = np.asarray([0,0,*action], env.vehicle.dtype) # (Fx, Fy, Fz, Tx, Ty, Tz)
        _, r, terminated, truncated, _ = env.step(dynamics)
        log.log(reward=r)
        if terminated or truncated:
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
    # The rate loop is stable while |1 - k_p * dt| < 1, i.e. dt < 2 / 32 = 0.0625 s.
    assert m.simulation.dt <= 0.05, 'Simulation time step too large for the default gains.'
    pos = PosController( # PD
        0.8, 0., 3.75,
        max_err_i=args.max_velocity,
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        square_root_scaling=args.sqrt_scaling,
        leashing=args.leashing
    )
    # Each loop must be slower than the loop inside it. Approximate bandwidths:
    # velocity g * k_p = 5 rad/s, attitude 8 rad/s, rate 32 rad/s.
    vel = VelController( # P
        0.5, 0., 0,
        max_err_i=args.max_acceleration,
        max_tilt=args.max_tilt)
    att = AttController( # P
        [8., 8., 0.], 0, 0., # yaw param is set to 0, in case use_yaw=False
        max_err_i=args.max_err_i_attitude)
    # The rate controller outputs angular acceleration, and k_d acts on the change
    # of the rate error, which that acceleration causes. A k_d of 1 or more makes
    # the torque flip sign at every step, so there is no D term.
    rat = RateController( # P
        [32, 32, 0], 0, 0, # yaw param is set to 0, in case use_yaw=False
        max_err_i=args.max_err_i_rate,
        max_acceleration=args.max_angular_acceleration)

    alt = AltController(
        1, 0, 0,
        max_err_i=args.max_velocity,
        max_velocity=args.max_velocity)
    alt_rate = AltRateController(
        10, 0, 0,
        max_err_i=args.max_acceleration)

    ctrl = Controller(
        pos, vel, att, rat, alt, alt_rate, vehicle=m,
        period_p=0.1, period_a=0.01, period_z=0.1,
    )
    if scurve:
        return SCurveController(ctrl)
    return ctrl



def get_fleet_controller(fleet: Fleet, args: Namespace=DEFAULTS) -> FleetController:
    """
    Create a `FleetController` whose vehicles all have the default gains of
    `get_controller`. Set the gains of a vehicle with `FleetController.set_params`.

    Parameters
    ----------
    fleet : Fleet
        The vehicles to control.
    args : Namespace, optional
        The default settings, by default DEFAULTS

    Returns
    -------
    FleetController
    """
    template = get_controller(Multirotor(fleet.params[0], fleet.simulation), args=args)
    return FleetController(fleet, template)



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
    # The ranges surround the gains of `get_controller()`. Each loop must be slower
    # than the loop inside it, and a D term acts on the change of the loop's output,
    # so large gains make the vehicle unstable within a few steps.
    r_pitch_roll_p = trial.suggest_float(prefix + 'r_pitch_roll.k_p', 5, 64)
    r_pitch_roll_i = trial.suggest_float(prefix + 'r_pitch_roll.k_i', 0, 1)
    # The rate controller outputs acceleration, so a k_d of 1 or more is unstable
    r_pitch_roll_d = trial.suggest_float(prefix + 'r_pitch_roll.k_d', 0, 0.5)
    r_pitch_roll_max_acc = trial.suggest_float(prefix + 'r_pitch_roll.max_acceleration', 0.1, 25)
    if args.use_yaw:
        r_yaw_p = trial.suggest_float(prefix + 'r_yaw.k_p', 0.1, 5)
        r_yaw_i = trial.suggest_float(prefix + 'r_yaw.k_i', 0.1, 1)
        r_yaw_d = trial.suggest_float(prefix + 'r_yaw.k_d', 0, 0.5)
        r_yaw_max_acc = trial.suggest_float(prefix + 'r_yaw.max_acceleration', 0.1, 5)
    else:
        r_yaw_p, r_yaw_i, r_yaw_d, r_yaw_max_acc = 0, 0, 0, 0

    a_pitch_roll_p = trial.suggest_float(prefix + 'a_pitch_roll.k_p', 1, 16)
    a_pitch_roll_i = trial.suggest_float(prefix + 'a_pitch_roll.k_i', 0, 1)
    a_pitch_roll_d = trial.suggest_float(prefix + 'a_pitch_roll.k_d', 0, 3)
    if args.use_yaw:
        a_yaw_p = trial.suggest_float(prefix + 'a_yaw.k_p', 0.1, 5)
        a_yaw_i = trial.suggest_float(prefix + 'a_yaw.k_i', 0.1, 1)
        a_yaw_d = trial.suggest_float(prefix + 'a_yaw.k_d', 0, 3)
    else:
        a_yaw_p, a_yaw_i, a_yaw_d = 0, 0, 0

    params = dict(
        ctrl_p = dict(
            k_p = trial.suggest_float(prefix + 'p.k_p', 1, 50),
            k_i = trial.suggest_float(prefix + 'p.k_i', 0, 1),
            k_d = trial.suggest_float(prefix + 'p.k_d', 0, 10),
        ),
        ctrl_v = dict(
            k_p = trial.suggest_float(prefix + 'v.k_p', 0.05, 1),
            k_i = trial.suggest_float(prefix + 'v.k_i', 0, 0.5),
            k_d = trial.suggest_float(prefix + 'v.k_d', 0, 1),
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
            k_p = trial.suggest_float(prefix + 'z.k_p', 0.2, 4),
            k_i = trial.suggest_float(prefix + 'z.k_i', 0, 0.5),
            k_d = trial.suggest_float(prefix + 'z.k_d', 0, 0.1),
        ),
        ctrl_vz = dict(
            k_p = trial.suggest_float(prefix + 'vz.k_p', 1, 20),
            k_i = trial.suggest_float(prefix + 'vz.k_i', 0, 1),
            k_d = trial.suggest_float(prefix + 'vz.k_d', 0, 1),
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



def make_scenarios(
    num: int, bounding_box: float, proximity: float, rng: np.random.Generator,
    dtype: type=np.float32
) -> np.ndarray:
    """
    Make start states for flights to the origin. The distances of the starts from
    the origin are spread evenly from near to far, and their directions are random.

    Parameters
    ----------
    num : int
        The number of scenarios.
    bounding_box : float
        The side of the cube, centered at the origin, that the vehicle flies in.
        Starts are inside 0.75 of it, as in `BaseMultirotorEnv.reset()`.
    proximity : float
        Distance from the origin at which the vehicle has reached it. Starts are
        at least this far from the origin in every coordinate.
    rng : np.random.Generator
        The source of randomness.
    dtype : type, optional
        The data type of the states, by default np.float32

    Returns
    -------
    np.ndarray
        The vehicle states, shape (num, 12). The vehicle is at rest.
    """
    half = bounding_box * 0.75 / 2
    # One distance in each of `num` equal ranges between near and the cube's corner
    near, far = 2 * proximity, half * np.sqrt(3)
    distances = near + (far - near) * (np.arange(num) + rng.random(num)) / num
    directions = rng.normal(size=(num, 3))
    positions = directions / np.linalg.norm(directions, axis=1, keepdims=True) * distances[:, None]
    # Shrink the starts that fall outside the cube towards the origin
    positions *= np.minimum(1., half / np.abs(positions).max(axis=1, keepdims=True))
    positions = np.where(np.abs(positions) < proximity, np.sign(positions) * proximity, positions)
    states = np.zeros((num, 12), dtype)
    states[:, :3] = positions
    return states



def make_scenario_suites(
    args: Namespace=DEFAULTS, seed: int=0, dtype: type=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make the scenarios that a study flies, and the scenarios that score its result.
    The two are drawn from different random streams, and only `seed` decides them.

    Parameters
    ----------
    args : Namespace, optional
        Optimization params, by default DEFAULTS
    seed : int, optional
        The seed of the study, by default 0
    dtype : type, optional
        The data type of the states, by default np.float32

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The training and held-out start states, each of shape (num, 12).
    """
    train_stream, heldout_stream = np.random.SeedSequence(seed).spawn(2)
    proximity = DynamicsMultirotorEnv.proximity
    return (
        make_scenarios(args.num_sims, args.bounding_box, proximity,
                       np.random.default_rng(train_stream), dtype),
        make_scenarios(args.num_heldout_sims, args.bounding_box, proximity,
                       np.random.default_rng(heldout_stream), dtype),
    )



def run_scenarios(env: DynamicsMultirotorEnv, ctrl: Controller, starts: np.ndarray) -> float:
    """
    Fly the controller from each start state to the origin.

    Parameters
    ----------
    env : DynamicsMultirotorEnv
        The environment to fly in.
    ctrl : Controller
        The controller, made for the vehicle of `env`.
    starts : np.ndarray
        The start states, shape (num, 12).

    Returns
    -------
    float
        The mean over the scenarios of the total reward of the flight.
    """
    rewards = []
    for start in starts:
        env.reset(options={'state': start})
        ctrl.reset()
        traj = Trajectory(env.vehicle, np.asarray([[0, 0, 0]]), proximity=0.1)
        rewards.append(run_sim(env, traj, ctrl).reward.sum())
    return float(np.mean(rewards))



def make_trial_controller(
    env: DynamicsMultirotorEnv, ctrl: Controller=None, args: Namespace=DEFAULTS
) -> Controller:
    """
    Make the controller that a trial or the final scoring flies: a copy of `ctrl`,
    or a `get_controller()` controller if `ctrl` is None, made for the vehicle of
    `env`. The caller's `ctrl` keeps its vehicle and parameters.
    """
    if ctrl is None:
        return get_controller(env.vehicle, args.scurve, args)
    # The copy controls the vehicle of `env`, not a copy of the caller's vehicle
    return copy.deepcopy(ctrl, {id(ctrl.vehicle): env.vehicle})



def make_objective(
    vp: VehicleParams, sp: SimulationParams, ctrl: Controller=None,
    args: Namespace=DEFAULTS, seed: int=0, starts: np.ndarray=None
) -> Callable:
    """
    Make the function that `optuna` will use to optimize. Each trial flies the
    same scenarios, so the scores of two trials differ only because their
    parameters differ.

    Parameters
    ----------
    vp : VehicleParams
    sp : SimulationParams
    ctrl: Controller, optional
        The controller class to use. If not provided, `get_controller` is used with `DEFAULTS`
        to make a controller instance.
    args : Namespace, optional
        Optimization params, by default DEFAULTS
    seed : int, optional
        The seed of the scenarios, by default 0. Not used if `starts` is given.
    starts : np.ndarray, optional
        The start states of the scenarios, shape (num, 12). By default, the
        training suite of `make_scenario_suites(args, seed)`.
    """
    if starts is None:
        starts, _ = make_scenario_suites(args, seed, sp.dtype or np.float32)

    def objective(trial: optuna.Trial, ctrl=ctrl):
        # objective is to navigate to origin from the initial positions
        ctrl_params = make_controller_from_trial(trial=trial, args=args)
        env = make_env(vp, sp, args)
        trial_ctrl = make_trial_controller(env, ctrl, args)
        trial_ctrl.set_params(**ctrl_params)
        return run_scenarios(env, trial_ctrl, starts)
    return objective



def optimize(
    vp: VehicleParams, sp: SimulationParams, ctrl: Controller=None,
    ntrials: int=1000,
    args: Namespace=DEFAULTS, seed: int=0,
    study_name: str=None,
    n_jobs: int=1,
    timeout: float=None,
    verbosity=optuna.logging.WARNING
) -> optuna.Study:
    """
    Search over parameter space to optimize controller parameters in `make_controller_from_trial()`.

    Parameters
    ----------
    vp : VehicleParams
    sp : SimulationParams
    ctrl: Controller, optional
        The controller class to use. If not provided, `get_controller` is used with `DEFAULTS`
        to make a controller instance.
    ntrials : int, optional
        Number of trials, by default 1000
    args : Namespace, optional
        Optimization params, by default DEFAULTS
    seed : int, optional
       Optimization seed, by default 0
    study_name : str, optional
        Name of study to be saved in `studies/study_name.db`, by default None
    n_jobs: int, optional
        Number of parallel optimization jobs to run. If -1, sets to number of processors. By default 1
    timeout: float, optional
        Number of seconds to run the study for. By default, run until n_trials reached
    verbosity : _type_, optional
        Logging level of `optuna.optimize()`, by default optuna.logging.WARNING

    Returns
    -------
    optuna.Study
        The completed `Study` object with `study.best_params` which can be used
        by `apply_params(controller, study.best_params)` to get the optimized
        controller. `study.user_attrs['heldout_score']` is the mean total reward
        of the best parameters over `args.num_heldout_sims` scenarios that the
        study did not fly. `study.best_value` is the mean over the scenarios
        that it did fly.
    """
    optuna.logging.set_verbosity(verbosity)
    study = get_study(study_name, seed=seed)
    # A resumed study must use the scenarios that its earlier trials flew.
    scenarios = dict(
        seed=seed, num_sims=args.num_sims, num_heldout_sims=args.num_heldout_sims,
        bounding_box=args.bounding_box)
    if 'scenarios' in study.user_attrs and study.user_attrs['scenarios'] != scenarios:
        raise ValueError(
            'The study was made with scenarios %s, not %s.' % (study.user_attrs['scenarios'], scenarios))
    study.set_user_attr('scenarios', scenarios)
    train, heldout = make_scenario_suites(args, seed, sp.dtype or np.float32)
    study.optimize(
        make_objective(vp, sp, ctrl=ctrl, args=args, starts=train),
        n_trials=ntrials,
        show_progress_bar=True,
        n_jobs=n_jobs,
        timeout=timeout
    )
    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        env = make_env(vp, sp, args)
        best_ctrl = make_trial_controller(env, ctrl, args)
        apply_params(best_ctrl, study.best_params)
        study.set_user_attr('heldout_score', run_scenarios(env, best_ctrl, heldout))
    return study



def apply_params(ctrl: Controller, params: dict) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply parameters from an optuna.Study to a `Controller`. Converts the dictionary
    of `best_params` in the format accepted by `controller.set_params()`

    Parameters
    ----------
    ctrl : Controller
        The controller to which to apply the parameters. If None, return the
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
    p['ctrl_r']['max_err_i'] = p['ctrl_r']['max_acceleration']
    p['ctrl_a']['k_p'] = np.asarray([params['a_pitch_roll.k_p'], params['a_pitch_roll.k_p'], params.get('a_yaw.k_p', 0)])
    p['ctrl_a']['k_i'] = np.asarray([params['a_pitch_roll.k_i'], params['a_pitch_roll.k_i'], params.get('a_yaw.k_i', 0)])
    p['ctrl_a']['k_d'] = np.asarray([params['a_pitch_roll.k_d'], params['a_pitch_roll.k_d'], params.get('a_yaw.k_d', 0)])
    if ctrl is not None:
        ctrl.set_params(**p)
    return p
