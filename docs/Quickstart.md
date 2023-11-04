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


First, import the dependencies:

```{code-cell} ipython3
%matplotlib widget
%reload_ext autoreload
%autoreload 2
from pprint import pprint
import numpy as np

np.set_printoptions(precision=3)
```

## Creating a vehicle


A vehicle is represented by a `multirotor.simulation.Multirotor` class, which is responsible for the physics calculations. The `Multirotor` is parametrized by `VehicleParams` and `SimulationParams`.

`VehicleParams` itself is parametrized by `PropellerParams` for each propeller. A propeller may optionally have `MotorParams` characterizing the electrical properties. If not, any changes to propeller speeds are instant. Let's create a simple vehicle, ignoring motor dynamics:

```{code-cell} ipython3
from multirotor.vehicle import VehicleParams, PropellerParams, SimulationParams
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
vp = vp = vehicle_params_factory(
    n=4,          # number of propellers, equally spaced
    m_prop=1e-2,  # mass of propeller / kg
    d_prop=0.3,   # distance of propeller from center of mass /m
    params=pp,    # each propeller's parameters
    m_body=2      # mass of body
)
vehicle = Multirotor(params=vp, simulation=sp)
```

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

## Visualizing


`multirotor.visualize.VehicleDrawing` class is a wrapper around matplotlib. It can interactively visualize the vehicle in 3D.

```{code-cell} ipython3
from multirotor.visualize import VehicleDrawing
vehicle.reset() # resets time/position back to 0
drawing = VehicleDrawing(vehicle, trace=True)

for j in range(1000):
    vehicle.step_dynamics(np.asarray(
        [0.4, 2 * np.sin(j*np.pi/100), vehicle.params.mass*sp.g +  np.cos(j*2*np.pi/1000),
         0,0,0],
        dtype=vehicle.dtype))
    drawing.axis.set_title(f'pos:{vehicle.position}')
    drawing.update()
```

Additionally, the `multirotor.visualize.plot_datalog` function can visualize timeseries measurements from the vehicle:

```{code-cell} ipython3
from multirotor.helpers import DataLog
from multirotor.visualize import plot_datalog
log = DataLog(vehicle)
vehicle.reset()
for j in range(1000):
    vehicle.step_dynamics(np.asarray(
        [0.4, 2 * np.sin(j*np.pi/100), vehicle.params.mass*sp.g +  np.cos(j*2*np.pi/1000),
         0,0,0],
        dtype=vehicle.dtype))
    log.log()
log.done_logging() # converts to numpy arrays
plot_datalog(log, figsize=(8,4));
```

## Controlling a vehicle


So far, so good. However, controlling a vehicle is another challenge.`multirotor` provides a `multirotor.controller.Controller` class, which uses PID control to navigate.

```{code-cell} ipython3
from multirotor.controller import Controller

ctrl = Controller.make_for(vehicle)
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

This can then be looped over and over again:

```{code-cell} ipython3
vehicle.reset()
ctrl.reset()
drawing = VehicleDrawing(
    vehicle, trace=True,
    make_fig_kwargs={'xlim':(-1,10), 'ylim':(-3,3), 'zlim':(-1,2)}
)
log = DataLog(vehicle, ctrl)
for i in range(500):
    dynamics = ctrl.step(reference=[20,0,1,0], persist=True)
    speeds = vehicle.allocate_control(dynamics[0], dynamics[1:4])
    state = vehicle.step_speeds(speeds)
    drawing.update()
    log.log()
log.done_logging()
drawing.axis.view_init(elev=30, azim=-100)
```

```{code-cell} ipython3
plot_datalog(log, figsize=(8,6));
```

## Optimizing control


The PID controller has many tunable parameters. Searching for an adequate parametrization for a vehicle is a complex search problem.

```{code-cell} ipython3
pprint(ctrl.get_params())
```

`multirotor.optimize` provides a convenience function called `optimize()` to search for the best parameters. Parameter search is done using the [optuna](https://optuna.org/) library, which is installed as a dependency.

```{code-cell} ipython3
from multirotor.optimize import optimize, run_sim, apply_params
study = optimize(vp, sp, ctrl, ntrials=100)

from optuna.visualization.matplotlib import plot_parallel_coordinate
plot_parallel_coordinate(study)
```

Then, the best parameters can be applied to the controller:

```{code-cell} ipython3
apply_params(ctrl, study.best_params);
```

## Trajectories


`multirotor.trajectories` defines the `Trajectory` class. It can take a list of waypoints and break them into smaller segments for the controller.

```{code-cell} ipython3
from multirotor.trajectories import Trajectory
from multirotor.env import DynamicsMultirotorEnv
vehicle.reset()
ctrl = Controller.make_for(vehicle)
drawing = VehicleDrawing(vehicle, trace=True)
traj = Trajectory(vehicle, points=[[5,0,0], [5,0,5], [5,5,2]],  proximity=0.1, resolution=0.3)
for i, (ref, _) in enumerate(traj):
    dynamics = ctrl.step(reference=[*ref, 0]) # yaw=0 in this case
    speeds = vehicle.allocate_control(dynamics[0], dynamics[1:4])
    vehicle.step_speeds(speeds)
    drawing.update()
    if i==1000:
        break
```

## Environments


`multirotor.env` defines [gym-compatible][1] environments for reinforcement learning experiments. The environments, called `DynamicsMultirotorEnv` and `SpeedsMultirotorEnv` take either the dynamics or propeller speeds as inputs. By default, rewards are for navigating to the origin. `env.reset()` method initializes to a random position inside some bounding box. The `BaseMultirotorEnv` class can be extended for a variety of objectives.

[1]: https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.StepAPICompatibility

```{code-cell} ipython3
from multirotor.env import DynamicsMultirotorEnv

env = DynamicsMultirotorEnv(vehicle, allocate=True)
state = env.reset()
print('Initial position')
print(env.vehicle.position)

forces_torques = np.asarray([1,1,10,0,0,0.1])
state, reward, done, info = env.step(forces_torques)
```
