# Quickstart


First, import the dependencies:

```python
%matplotlib widget
%reload_ext autoreload
%autoreload 2
import numpy as np
from multirotor.simulation import Multirotor, Propeller
from multirotor.vehicle import VehicleParams, PropellerParams, SimulationParams
from multirotor.helpers import vehicle_params_factory
from multirotor.optimize import optimize, run_sim, apply_params
from multirotor.visualize import VehicleDrawing
```

## Creating a vehicle


A vehicle is represented by a `multirotor.simulation.Multirotor` class, which is responsible for the physics calculations. The `Multirotor` is parametrized by `VehicleParams` and `SimulationParams`.

`VehicleParams` itself is parametrized by `PropellerParams` for each propeller. A propeller may optionally have `MotorParams` characterizing the electrical properties. If not, any changes to propeller speeds are instant. Let's create a simple vehicle, ignoring motor dynamics:

```python
pp = PropellerParams(
    k_thrust=0.01,
    k_drag=1e-2,
    moment_of_inertia=1e-3
)
vp = vehicle_params_factory(
    n=4,
    m_prop=1e-2,
    d_prop=0.3,
    params=pp,
    m_body=2
)
sp = SimulationParams(dt=1e-2)
```

<!-- #region -->
The vertical thrust in Newtons $F_z$ is governed by the thrust coefficient. For propeller velocity (radians per second) $\omega$,

$$
F_z = k_{thrust} \cdot \omega^2
$$

Similarly, the torque produced about the vertical axis of the multirotor due to propeller drag, $T_z$ is governed by the drag coefficient,

$$
T_z = k_{drag} \cdot \omega^2
$$


`multirotor` has a convenience function for creating a vehicle from simple geometries:
<!-- #endregion -->

```python
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


## Visualizing

```python
vehicle.reset()
drawing = VehicleDrawing(vehicle, trace=True, max_frames_per_second=5)
drawing.connect()
i = 0
```

```python
for j in range(1000):
    vehicle.step_dynamics(np.asarray(
        [0.1, np.sin(i*np.pi/100), vehicle.params.mass*sp.g +  np.cos(i*2*np.pi/1000),
         0,0,0],
        dtype=vehicle.dtype))
    drawing.axis.set_title(f'pos:{vehicle.position}')
    # drawing.update()
    i += 1
```

```python
drawing.disconnect()
```

```python
len(drawing.trajectory[0])
```
