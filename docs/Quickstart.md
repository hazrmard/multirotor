---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# multirotor - Quickstart


This tutorial shows how to use `multirotor` to simulate, control, and tune a drone. It has four parts:

1. **A first flight.** Create a vehicle, tune a controller, and fly waypoints with logging. This is the example in the README, with explanations.
2. **Motors and a battery.** Simulate the electrical side of the same flight.
3. **Trajectories.** Fly a longer path and add a wind disturbance.
4. **Environments.** Use the vehicle as a [Gymnasium](https://gymnasium.farama.org) environment for reinforcement learning.

This document can be run as a Jupyter notebook using the [jupytext](https://jupytext.readthedocs.io/en/latest/) extension. Link to raw markdown file is [here](https://raw.githubusercontent.com/hazrmard/multirotor/master/docs/Quickstart.md).

```{code-cell} ipython3
%matplotlib inline
from pprint import pprint
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
```

## 1. A first flight


### Create a vehicle

A vehicle is represented by a `multirotor.simulation.Multirotor` class, which is responsible for the physics calculations. The `Multirotor` is parametrized by `VehicleParams` and `SimulationParams`.

`VehicleParams` itself is parametrized by `PropellerParams` for each propeller. A propeller may optionally have `MotorParams` characterizing the electrical properties. If not, any changes to propeller speeds are instant. Part 2 adds motors. Let's create a simple vehicle without them:

```{code-cell} ipython3
from multirotor.vehicle import PropellerParams, SimulationParams
pp = PropellerParams(
    k_thrust=0.01,
    k_drag=1e-2,
    moment_of_inertia=1e-3
)
sp = SimulationParams(
    dt=1e-2 # simulation time-step in seconds
)
```

The vertical thrust in Newtons $F_z$ is governed by the thrust coefficient. For propeller velocity (radians per second) $\omega$,

$$
F_z = k_{thrust} \cdot \omega^2
$$

Similarly, the torque produced about the vertical axis of the multirotor due to propeller drag, $T_z$ is governed by the drag coefficient,

$$
T_z = k_{drag} \cdot \omega^2
$$


`multirotor` has a convenience function for creating a vehicle from simple geometries:

```{code-cell} ipython3
from multirotor.helpers import vehicle_params_factory
from multirotor.simulation import Multirotor
vp = vehicle_params_factory(
    n=4,          # number of propellers, equally spaced
    m_prop=1e-2,  # mass of propeller / kg
    d_prop=0.3,   # distance of propeller from center of mass /m
    params=pp,    # each propeller's parameters
    m_body=2      # mass of body
)
vehicle = Multirotor(params=vp, simulation=sp)
```

For other layouts, create `VehicleParams` directly. It takes the propellers, their angles and distances, the mass and the inertia matrix. `multirotor.models` has ready-made parameters for two vehicles.

The `Multirotor` provides a host of methods to manipulate the vehicle. The two main methods are:

1. `step_dynamics`. It takes a 6D vector of forces and torques in the body frame of the vehicle, and returns the state.
2. `step_speeds`. It takes a `n`D vector of propeller speeds, and returns the state.

**Note**: The data type of the vectors should match `vehicle.dtype`.

The vehicle is described by its `.state` attribute. It is a 12-dimensional array containing the position, velocity, orientation, and angular velocity. Individual state components can be accessed by their relevant attributes:

```{code-cell} ipython3
# Step using forces/torques
state = vehicle.step_dynamics(np.asarray([1,2,vehicle.weight,0,0,1], vehicle.dtype))
# Step using speeds (rad/s)
state = vehicle.step_speeds(np.asarray([100, 100, 100, 100], dtype=vehicle.dtype))

print('State:\n', state)
print('\nConsists of:')
print(f'{"Position": >32}', vehicle.position)
print(f'{"Velocity (body-frame)": >32}', vehicle.velocity)
print(f'{"Orientation": >32}', vehicle.orientation)
print(f'{"Angular Rate": >32}', vehicle.angular_rate)

print('\nAdditional properties:')
print(f'{"Velocity (inertial-frame)": >32}', vehicle.inertial_velocity)
print(f'{"Acceleration": >32}', vehicle.acceleration)
print(f'{"Euler rate": >32}', vehicle.euler_rate)
```

### Visualize

`multirotor.visualize.VehicleDrawing` class is a wrapper around matplotlib. It can interactively visualize the vehicle in 3D. In a notebook, use the `%matplotlib widget` backend (from `ipympl`) for an interactive figure. Here, the figure shows the path after 10 seconds:

```{code-cell} ipython3
from multirotor.visualize import VehicleDrawing
vehicle.reset() # resets time/position back to 0
drawing = VehicleDrawing(vehicle, trace=True)

for j in range(1000):
    vehicle.step_dynamics(np.asarray(
        [0.4, 2 * np.sin(j*np.pi/100), vehicle.params.mass*sp.g +  np.cos(j*2*np.pi/1000),
         0,0,0],
        dtype=vehicle.dtype))
    if j % 10 == 0:
        drawing.update()
drawing.axis.set_title(f'pos:{vehicle.position}')
```

### Control the vehicle

Controlling a vehicle is another challenge. `multirotor` provides a `multirotor.controller.Controller` class, which uses cascaded PID control to navigate. `multirotor.optimize.get_controller` makes one with default gains:

```{code-cell} ipython3
from multirotor.optimize import get_controller

vehicle.reset()
ctrl = get_controller(vehicle)
```

A controller has a `step()` method, which takes the reference position and yaw, and outputs the dynamics needed to achieve that.

$$
F_z,\tau_x,\tau_y,\tau_z = \texttt{step(}x,y,z,\psi\texttt{)}
$$

Let's say we want the vehicle to go up to $z=10$ and $x=20$

```{code-cell} ipython3
dynamics = ctrl.step(reference=[20,0,10,0], persist=False)
print('F_z=%.3f, T_x==%.3f, T_y=%.3f, T_z=%.3f' % (dynamics[0], dynamics[1], dynamics[2], dynamics[3]))
```

Now that we have the prescribed dynamics, we must convert them into prescribed speeds (radians / s) for the vehicle propellers. That's where control allocation comes in:

```{code-cell} ipython3
speeds = vehicle.allocate_control(dynamics[0], dynamics[1:4])
print(speeds)
```

And finally, the speeds can be applied to the vehicle:

```{code-cell} ipython3
state = vehicle.step_speeds(speeds)
print(state)
```

This can then be looped over and over again. `DataLog` records the vehicle and controller at each step, and `plot_datalog` plots the recording:

```{code-cell} ipython3
from multirotor.helpers import DataLog
from multirotor.visualize import plot_datalog

vehicle.reset()
ctrl.reset()
log = DataLog(vehicle, ctrl)
for i in range(500):
    dynamics = ctrl.step(reference=[20,0,1,0], persist=True)
    speeds = vehicle.allocate_control(dynamics[0], dynamics[1:4])
    state = vehicle.step_speeds(speeds)
    log.log()
log.done_logging() # converts to numpy arrays
plot_datalog(log, figsize=(6,4));
```

### Tune the controller

The PID controller has many tunable parameters. Searching for an adequate parametrization for a vehicle is a complex search problem. `multirotor.optimize` provides a convenience function called `optimize()` to search for the best parameters. Parameter search is done using the [optuna](https://optuna.org/) library, which is installed as a dependency.

Each trial of the search sets the gains of the controller, then flies 10 episodes from random start positions to the origin. The score is the mean total reward of the episodes. The search takes about 15 seconds for 100 trials:

```{code-cell} ipython3
from multirotor.optimize import optimize, apply_params
study = optimize(vp, sp, ntrials=100)
print('Best score:', study.best_value)
print('Best parameters:')
pprint(study.best_params, compact=True)
```

Then, the best parameters can be applied to a controller:

```{code-cell} ipython3
tuned = get_controller(vehicle)
apply_params(tuned, study.best_params);
```

To compare the tuned gains with the default gains, fly 20 seeded episodes to the origin with each controller. The seeds are not the ones the search used. An episode succeeds if the vehicle ends within `env.proximity` of the origin:

```{code-cell} ipython3
from multirotor.optimize import make_env, run_sim
from multirotor.trajectories import Trajectory

def evaluate(params=None, episodes=20):
    env = make_env(vp, sp)
    controller = get_controller(env.vehicle)
    if params is not None:
        apply_params(controller, params)
    successes, rewards, steps = 0, [], []
    for seed in range(100, 100 + episodes):
        env.reset(seed=seed)
        controller.reset()
        log = run_sim(env, Trajectory(env.vehicle, [[0, 0, 0]], proximity=0.1), controller)
        successes += np.linalg.norm(env.vehicle.position) <= env.proximity
        rewards.append(log.reward.sum())
        steps.append(len(log.reward))
    print(f'{successes} of {episodes} reached the origin, '
          f'mean reward {np.mean(rewards):.0f}, mean length {np.mean(steps):.0f} steps')

print('Default gains:'); evaluate()
print('Tuned gains:  '); evaluate(study.best_params)
```

The default gains are a good starting point, and for this vehicle the tuned gains perform about the same. Use the search when the default gains fly your vehicle poorly.

### Fly waypoints

`multirotor.trajectories.Trajectory` takes a list of waypoints and breaks them into smaller segments for the controller. It moves to the next segment when the vehicle is within `proximity` of the current one. `resolution` is the length of a segment:

```{code-cell} ipython3
vehicle.reset()
tuned.reset()
log = DataLog(vehicle, tuned)
traj = Trajectory(vehicle, points=[[5,0,5], [5,5,5], [0,0,2]], proximity=0.5, resolution=1)
for i, (ref, feed_forward_velocity) in enumerate(traj):
    dynamics = tuned.step(reference=[*ref, 0], feed_forward_velocity=feed_forward_velocity) # yaw=0 in this case
    speeds = vehicle.allocate_control(dynamics[0], dynamics[1:4])
    vehicle.step_speeds(speeds)
    log.log()
    if i == 6000: # stop if a waypoint is not reached
        break
log.done_logging()
print(f'Reached the end after {vehicle.t:.1f} s at {vehicle.position}')
plot_datalog(log, figsize=(6,4));
```

## 2. Motors and a battery


So far, propeller speeds change instantly. A real propeller is driven by a motor, and the motor by a battery. Add a `MotorParams` to each propeller to simulate the motor's electrical and rotational dynamics. Add a `BatteryParams` to the vehicle to simulate the battery.

This example uses the parameters of an octorotor with 18-inch propellers from `multirotor.models.tarott18`, and gives it a 6-cell lithium-polymer battery. The open-circuit voltage of the battery falls with its state of charge, and its terminal voltage is lower by `internal_resistance` times the current. If the terminal voltage falls below `min_voltage`, the battery cuts off the motors.

```{code-cell} ipython3
from copy import deepcopy
from dataclasses import replace
from multirotor.models.tarott18 import MP, PP, VP, SP
from multirotor.vehicle import BatteryParams

cells = 6
cell_soc = [0., 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.]
cell_ocv = [3.3, 3.55, 3.65, 3.72, 3.78, 3.85, 3.97, 4.06, 4.2]
battery = BatteryParams(
    capacity=16.,              # Ah
    internal_resistance=0.01,  # ohm
    ocv_soc=cell_soc,
    ocv_voltage=cells * np.asarray(cell_ocv),
    min_voltage=cells * 3.5
)
propeller = replace(PP, motor=MP)  # the propeller is driven by this motor
vp_motors = replace(VP, propellers=[deepcopy(propeller) for _ in range(8)], battery=battery)
octo = Multirotor(vp_motors, SP)
print(f'Mass: {vp_motors.mass} kg, weight: {octo.weight:.1f} N')
```

The actions of `step_speeds` stay the same: speeds in rad/s. Each motor converts its speed into a voltage using `MotorParams.speed_voltage_scaling`, and the motor accelerates the propeller. The controller and the allocation do not change. One extra step is needed: spinning up the motors from rest draws a large current, which can sag the battery below `min_voltage`. So the motors start at hover speed:

```{code-cell} ipython3
octo_ctrl = get_controller(octo)
octo.reset()
octo_ctrl.reset()

hover_speeds = octo.allocate_control(octo.weight, np.zeros(3))
for propeller, speed in zip(octo.propellers, hover_speeds):
    propeller.motor.speed = speed

log = DataLog(octo, octo_ctrl, other_vars=('soc', 'voltage', 'current'))
traj = Trajectory(octo, points=[[0,0,3], [5,0,3], [5,5,3]], proximity=0.5, resolution=1)
for i, (ref, feed_forward_velocity) in enumerate(traj):
    dynamics = octo_ctrl.step(reference=[*ref, 0], feed_forward_velocity=feed_forward_velocity)
    octo.step_speeds(octo.allocate_control(dynamics[0], dynamics[1:4]))
    log.log(soc=octo.battery.soc, voltage=octo.battery.voltage, current=octo.battery_current)
    if i == 6000:
        break
log.done_logging()
print(f'Reached the end after {octo.t:.1f} s at {octo.position}')
print(f'Battery: {100 * octo.battery.soc:.1f} % charge, {float(octo.battery.voltage):.1f} V')
```

The log has the battery measurements next to the vehicle's:

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(7, 4.5), sharex=True)
axes[0, 0].plot(log.t, 100 * log.soc)
axes[0, 0].set_ylabel('State of charge /%')
axes[0, 1].plot(log.t, log.voltage)
axes[0, 1].set_ylabel('Battery voltage /V')
axes[1, 0].plot(log.t, log.current)
axes[1, 0].set_ylabel('Battery current /A')
axes[1, 1].plot(log.t, log.z)
axes[1, 1].set_ylabel('Altitude /m')
for ax in axes[1]:
    ax.set_xlabel('Time /s')
fig.tight_layout()
```

The [Detailed Demo](./Detailed%20Demo.ipynb) flies this vehicle until the battery cuts off.

## 3. Trajectories


A `Trajectory` is a sequence of waypoints. Here, the vehicle flies a figure-eight at a height of 3 m. `multirotor.trajectories.eight_curve` generates the waypoints. A wind disturbance pushes the vehicle: `step_speeds` takes disturbance forces and torques in the body frame of the vehicle.

```{code-cell} ipython3
from multirotor.coords import direction_cosine_matrix, inertial_to_body
from multirotor.trajectories import eight_curve

def wind(t, vehicle):
    "A gust along x that grows and fades over 20 s. Returns the force in the body frame."
    gust_inertial = np.asarray([2 * np.sin(np.pi * t / 20), 0, 0])
    dcm = direction_cosine_matrix(*vehicle.orientation)
    return inertial_to_body(gust_inertial, dcm)

waypoints = eight_curve(a=8, N=20)
waypoints[:, 2] = 3.
vehicle.reset()
tuned.reset()
log = DataLog(vehicle, tuned)
traj = Trajectory(vehicle, points=waypoints, proximity=1., resolution=2.)
for i, (ref, feed_forward_velocity) in enumerate(traj):
    dynamics = tuned.step(reference=[*ref, 0], feed_forward_velocity=feed_forward_velocity)
    speeds = vehicle.allocate_control(dynamics[0], dynamics[1:4])
    vehicle.step_speeds(speeds, disturb_forces=wind(vehicle.t, vehicle))
    log.log()
    if i == 8000:
        break
log.done_logging()

fig, ax = plt.subplots(figsize=(4, 3.5))
ax.plot(log.x, log.y, label='flown')
ax.plot(waypoints[:, 0], waypoints[:, 1], 'o', ms=3, label='waypoints')
ax.set_xlabel('x /m'); ax.set_ylabel('y /m'); ax.set_aspect('equal'); ax.legend();
```

The `Trajectory` class gives the controller one waypoint at a time. `multirotor.trajectories.GuidedTrajectory` plans smooth S-curve motion between waypoints. It needs the optional [py-scurve](https://github.com/hazrmard/py-scurve) package. See the Detailed Demo for the other controller and trajectory options.

## 4. Environments


`multirotor.env` defines [Gymnasium](https://gymnasium.farama.org) environments for reinforcement learning. The environments, called `DynamicsMultirotorEnv` and `SpeedsMultirotorEnv`, take either the dynamics or propeller speeds as actions. The observation is the 12-dimensional state of the vehicle. By default, the reward is for navigating to the origin.

- `reset(seed=..., options=...)` starts an episode at a random position inside a bounding box, and returns `(observation, info)`. Use `options={'state': x}` to start from the state `x`.
- `step(action)` returns `(observation, reward, terminated, truncated, info)`. An episode terminates when the vehicle reaches the origin, leaves the bounding box or tips over. It is truncated when it runs out of time. `info` says which of `reached`, `out_of_bounds`, `tipped` and `out_of_time` happened.

```{code-cell} ipython3
from multirotor.env import DynamicsMultirotorEnv

env = DynamicsMultirotorEnv(Multirotor(vp, sp), allocate=True)
observation, info = env.reset(seed=0)
print('Initial position:', env.vehicle.position)
print('Observation space:', env.observation_space)
print('Action space:     ', env.action_space)
```

With `allocate=True`, an action is the 6D vector of forces and torques (the force along $z$ and the three torques matter). The environment allocates propeller speeds for it. Here, the PID controller is the policy. It gets the observation from the vehicle that it was made for:

```{code-cell} ipython3
policy = get_controller(env.vehicle)
total_reward = 0.
for step in range(2000):
    thrust_torques = policy.step(reference=[0, 0, 0, 0]) # go to the origin
    action = np.asarray([0, 0, *thrust_torques], env.vehicle.dtype)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print(f'Episode ended after {step + 1} steps with total reward {total_reward:.1f}')
print('Why:', [name for name, happened in info.items() if happened])
```

A reinforcement learning library calls the same `reset()` and `step()` methods with the actions of its own policy. To change the objective, subclass `BaseMultirotorEnv` or one of its children and override `reward()`. `reward()` also sets the flags that end an episode, so call the parent's method and adjust its result:

```python
class SmoothEnv(DynamicsMultirotorEnv):
    "Penalizes the angular rate of the vehicle."

    def reward(self, state, action, nstate):
        reward = super().reward(state, action, nstate)
        return reward - 0.01 * np.linalg.norm(nstate[9:12])
```
