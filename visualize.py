from typing import Tuple
import threading as th
from dataclasses import dataclass
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

from coords import body_to_inertial, direction_cosine_matrix
from vehicle import VehicleParams
from simulation import Multirotor



@dataclass
class VehicleDrawing:
    vehicle: Multirotor
    axis: Axes3D = None
    max_frames_per_second: float = 30.
    trace: bool = False
    body_axes: bool = False

    def __post_init__(self):
        self.t = self.vehicle.t
        self.params = self.vehicle.params
        self.interval = 1 / self.max_frames_per_second
        self.is_ipython = is_ipython()
        self.arm_lines, self.arm_lines_points, \
        self.trajectory_line, \
        self.axis_lines, self.axis_lines_points = \
            make_drawing(self, self.body_axes)
        self.trajectory = [[], [], []] # [[X,..], [Y,...], [Z,...]]


    def connect(self):
        self.ev_cancel = th.Event()
        self.thread = th.Thread(target=self._animator, daemon=False)
        self.thread.start()


    def disconnect(self, force=False):
        self.ev_cancel.set()
        self.thread.join(timeout=1.5 * self.interval)
        del self.anim


    def _worker(self):
        if self.axis is None:
            self.figure, self.axis = make_fig((-10,10), (-10,10), (0,20))
        else:
            self.figure = self.axis.figure

        for l in self.arm_lines:
            self.axis.add_line(l)
        self.axis.add_line(self.trajectory_line)

        while not self.ev_cancel.is_set():
            start = time.time()
            position = self.vehicle.position
            orientation = self.vehicle.orientation
            update_drawing(self, position, orientation)
            end = time.time()
            self.ev_cancel.wait(self.interval - (end - start))


    def _animator(self):
        if self.axis is None:
            self.figure, self.axis = make_fig((-10,10), (-10,10), (-10,10))
        else:
            self.figure = self.axis.figure
        
        def init_func():
            for l in self.arm_lines:
                self.axis.add_line(l)
            self.axis.add_line(self.trajectory_line)
            for l in self.axis_lines:
                self.axis.add_line(l)
            return (*self.arm_lines, self.trajectory_line, self.axis_lines)
        
        def func(f):
            if self.ev_cancel.is_set():
                raise th.ThreadError('Canceling animation update.')
            if self.vehicle.t != self.t:
                if self.vehicle.t < self.t:
                    self.trajectory = [[], [], []] # likely vehicle is reset, so reset trajectory
                self.t = self.vehicle.t
                return update_drawing(self, self.vehicle.position, self.vehicle.orientation)
            else:
                return []

        self.anim = FuncAnimation(
            self.figure,
            func=func,
            init_func=init_func,
            blit=True,
            repeat=False,
            interval=self.interval * 1000)
        return self.anim
        


def make_drawing(drawing: VehicleDrawing, body_axes: bool=False):
    params = drawing.params
    arm_lines_points = np.zeros((len(params.propellers) * 2, 3)) # [2 points/ propeller, axis]
    x = params.distances * np.cos(params.angles)
    y = params.distances * np.sin(params.angles)
    arm_lines_points[1::2,0] = x
    arm_lines_points[1::2,1] = y
    arm_lines = []
    for i in range(len(params.propellers)):
        arm_lines.append(
            Line3D(
                arm_lines_points[2*i:2*i+2,0],
                arm_lines_points[2*i:2*i+2,1],
                arm_lines_points[2*i:2*i+2,2],
                antialiased=False))

    trajectory_line = Line3D([], [], [], linewidth=0.5, color='black', linestyle=':')

    axis_lines_points = np.zeros((6, 3)) # [2 points/ axis, axis]
    axis_lines_points[1::2] = np.eye(3)
    axis_lines = []
    for i, c in enumerate(['r', 'g', 'b']):
        if body_axes:
            axis_lines.append(Line3D(
                axis_lines_points[2*i:2*i+2,0],
                axis_lines_points[2*i:2*i+2,1],
                axis_lines_points[2*i:2*i+2,2],
                antialiased=False,
                linewidth=0.5,
                color=c
            )) 
        else:
            axis_lines.append(Line3D([], [], []))

    return arm_lines, arm_lines_points, trajectory_line, axis_lines, axis_lines_points



def update_drawing(drawing: VehicleDrawing, position: np.ndarray, orientation: np.ndarray):
    dcm = direction_cosine_matrix(orientation[0], orientation[1], orientation[2])
    arms = np.copy(drawing.arm_lines_points)
    arms = body_to_inertial(arms.T, dcm).T
    arms += position
    for i, l in enumerate(drawing.arm_lines):
        j = 2*i
        l.set_data(arms[j:j+2, 0], arms[j:j+2, 1])
        l.set_3d_properties(arms[j:j+2, 2])
    
    if drawing.trace:
        drawing.trajectory[0].append(position[0])
        drawing.trajectory[1].append(position[1])
        drawing.trajectory[2].append(position[2])
        drawing.trajectory_line.set_data(drawing.trajectory[0], drawing.trajectory[1])
        drawing.trajectory_line.set_3d_properties(drawing.trajectory[2])
    
    if drawing.body_axes:
        axes = np.copy(drawing.axis_lines_points)
        axes[1::2] = position
        axes[0::2] = dcm.T + position
        for i, l in enumerate(drawing.axis_lines):
            l.set_data(axes[i*2:i*2+2,0], axes[i*2:i*2+2,1])
            l.set_3d_properties(axes[i*2:i*2+2,2])
    # drawing.figure.canvas.draw_idle()
    # drawing.figure.canvas.flush_events()
    # plt.pause(1e-7)

    return (*drawing.arm_lines, drawing.trajectory_line, drawing.axis_lines)



def make_fig(xlim, ylim, zlim) -> Tuple[plt.Figure, Axes3D]:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return fig, ax



def is_ipython() -> bool:
    # https://stackoverflow.com/q/15411967/4591810
    # https://stackoverflow.com/a/39662359/4591810
    try:
        import IPython
        shell = IPython.get_ipython().__class__.__name__
        if ('Terminal' in shell) or ('terminal' in shell) or (shell == 'NoneType'):
            return False
        else:
            return True
    except NameError:
        return False
    except ImportError:
        return False
