from typing import Tuple, Union
import threading as th
import multiprocessing as mp
import queue
from dataclasses import dataclass
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

from .coords import body_to_inertial, direction_cosine_matrix
from .simulation import Multirotor
from .trajectories import Trajectory, GuidedTrajectory
from .controller import Controller
from .helpers import DataLog



def plot_datalog(log: DataLog, figsize=(21,10.5),
    plots=('pos', 'vel', 'ctrl', 'traj'),
    nrows=2, ncols=None):
    """
    Plot recorded values from a Multirotor's flight. Including:

    1. Position and orientation,
    2. Motor speeds,
    3. Velocity in world frame,
    4. Control allocation,
    5. Allocation errors,
    6. 2D x-y position plot

    Parameters
    ----------
    log : DataLog
        The datalog, where `datalog.done_logging()` has been called.
    figsize : tuple, optional
        The x/y dimensions of the figure, by default (21,10.5)
    """
    nplots = len(plots)
    if nrows is None and ncols is not None:
        nrows = nplots // ncols + (nplots % ncols !=0)
    elif nrows is not None and ncols is None:
        ncols = nplots // nrows + (nplots % nrows !=0)

    plt.figure(figsize=figsize)
    plot_grid = (nrows, ncols)

    n = len(log)
    hasctrl = log.controller is not None
    plot_number = 1

    # Positions
    if 'pos' in plots:
        plt.subplot(*plot_grid,plot_number)
        plt.plot(log.t, log.x, label='x', c='r')
        plt.plot(log.t, log.y, label='y', c='g')
        plt.plot(log.t, log.z, label='z', c='b')
        if hasctrl:
            plt.plot(log.t, log.target.position[:, 0], c='r', ls=':')
            plt.plot(log.t, log.target.position[:, 1], c='g', ls=':')
        lines = plt.gca().lines[:3]
        plt.ylabel('Position /m')
        plt.twinx()
        plt.plot(log.t, log.roll * (180 / np.pi), label='roll', c='c')
        plt.plot(log.t, log.pitch * (180 / np.pi), label='pitch', c='m')
        plt.plot(log.t, log.yaw * (180 / np.pi), label='yaw', c='y')
        if hasctrl:
            plt.plot(log.t, log.target.orientation[:,0] * (180 / np.pi), c='c', ls=':')
            plt.plot(log.t, log.target.orientation[:,1] * (180 / np.pi), c='m', ls=':')
            plt.plot(log.t, log.target.orientation[:,2] * (180 / np.pi), c='y', ls=':')
        plt.ylabel('Orientation /deg')
        plt.legend(handles=plt.gca().lines[:3] + lines, ncol=2)
        plt.title('Position and Orientation')
        plot_number += 1

    if 'vel' in plots:
        plt.subplot(*plot_grid, plot_number)
        v_world, v_ref = np.zeros_like(log.velocity), np.zeros_like(log.velocity)
        for i, (v, o) in enumerate(zip(log.velocity, log.orientation)):
            dcm = direction_cosine_matrix(*o)
            v_world[i] = body_to_inertial(v, dcm)
            if hasctrl:
                v_ref[i] = body_to_inertial(log.target.velocity[i], dcm)
            # v_ref[i] = body_to_inertial(np.concatenate((log.ctrl_p.action, log.ctrl_z.action)), dcm)
        for i, c, a in zip(range(3), 'rgb', 'xyz'):
            l, = plt.plot(log.t, v_world[:,i], label='Velocity %s' % a, c=c)
            if hasctrl:
                plt.plot(log.t, v_ref[:,i], ls=':', c=l.get_c())
        #     plt.plot(velocities[:,i], label='Velocity %s' % a, c=c)
        plt.legend()
        plt.title('Velocities')
        plot_number += 1

    if 'ctrl' in plots:
        plt.subplot(*plot_grid, plot_number)
        plt.title('Controller allocated dynamics')
        l = plt.plot(log.t, log.actions[:,0], label='Ctrl Thrust')
        plt.ylabel('Force /N')
        plt.twinx()
        for i, c, a in zip(range(3), 'rgb', 'xyz'):
            plt.plot(log.t, log.actions[:,1+i], label='Ctrl Torque %s' % a, c=c)
        plt.ylabel('Torque /Nm')
        plt.legend(handles=plt.gca().lines + l, ncol=2)
        plot_number += 1

    if 'traj' in plots:
        plt.subplot(*plot_grid, plot_number)
        plt.plot(log.target.position[:,0], log.target.position[:,1], label='Prescribed traj', ls=':')
        plt.plot(log.x, log.y, label='Actual traj', ls='-')
        plt.gca().set_aspect('equal', 'box')
        plt.title('XY positions /m')
        plt.xlabel('X /m')
        plt.ylabel('Y /m')
        plt.legend()
        plot_number += 1

    plt.tight_layout()



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
        self.is_terminal = is_terminal()
        self.arm_lines, self.arm_lines_points, \
        self.trajectory_line, \
        self.axis_lines, self.axis_lines_points = \
            make_drawing(self, self.body_axes)
        self.trajectory = [[], [], []] # [[X,..], [Y,...], [Z,...]]


    def connect(self, via: str ='animation') -> Union[FuncAnimation, th.Thread]:
        self.connection_via = via
        self.ev_cancel = mp.Event()
        self.queue = mp.Queue(maxsize=1)
        self.update_thread = th.Thread(target=self._update_worker)
        self.update_thread.start()
        if via=='animation':
            return self._animator()
        elif via=='thread':
            # self.thread = th.Thread(target=self._worker, daemon=True)
            # self.thread.start()
            # return self.thread
            return self._worker()
        elif via=='process':
            pr = mp.Process(target=self._worker)
            raise NotImplementedError()


    def disconnect(self, force=False):
        self.ev_cancel.set()
        if self.connection_via == 'thread':
            # self.thread.join(timeout=1.5 * self.interval)
            self.update_thread.join(timeout=1.5 * self.interval)
            del self.queue
            del self.update_thread
        elif self.connection_via=='animation':
            try:
                self.anim.pause()
            except AttributeError:
                # If the interactive jupyter figure is closed, this error is
                # raised if disconnect is called after.
                pass
            del self.anim
        elif self.connection_via=='process':
            raise NotImplementedError()


    def _init_func(self):
        for l in self.arm_lines:
                self.axis.add_line(l)
        self.axis.add_line(self.trajectory_line)
        for l in self.axis_lines:
            self.axis.add_line(l)
        return (*self.arm_lines, self.trajectory_line, *self.axis_lines)


    def _update_func(self, frame: int=None):
        vehicle_t, position, orientation = self.queue.get()
        if self.ev_cancel.is_set():
            # This function is called repeatedly so it needs to quit the calling
            # function (FuncAnimation(), _worker, when it gets the signal.
            raise th.ThreadError('Canceling animation update.')
        if vehicle_t != self.t:
            if vehicle_t < self.t:
                self.trajectory = [[], [], []] # likely vehicle is reset, so reset trajectory
            self.t = vehicle_t
            return update_drawing(self, position, orientation)
        else:
            return (*self.arm_lines, self.trajectory_line, *self.axis_lines)


    def _worker(self):
        # To be run in main thread. The vehicle control should be in a separate
        # thread that controls self.vehicle
        if self.axis is None:
            self.figure, self.axis = make_fig((-10,10), (-10,10), (0,20))
        else:
            self.figure = self.axis.figure

        self._init_func()

        i = 0
        while not self.ev_cancel.is_set():
            start = time.time()
            self._update_func(i)
            # self.figure.canvas.draw_idle()
            # self.figure.canvas.flush_events()
            i += 1
            end = time.time()
            pause = max(1e-3, self.interval - (end - start))
            plt.pause(pause)
            # self.ev_cancel.wait(self.interval - (end - start))


    def _animator(self):
        if self.axis is None:
            self.figure, self.axis = make_fig((-10,10), (-10,10), (-10,10))
        else:
            self.figure = self.axis.figure

        self.anim = FuncAnimation(
            self.figure,
            func=self._update_func,
            init_func=self._init_func,
            blit=True,
            repeat=False,
            interval=self.interval * 1000,
            cache_frame_data=False)
        return self.anim


    def _update_worker(self):
        # Thread that puts time, position, orientation into thread-/process-safe
        # queue.
        while not self.ev_cancel.is_set():
            start = time.time()
            try:
                self.queue.put(
                    (self.vehicle.t, self.vehicle.position, self.vehicle.orientation),
                    timeout=self.interval)
            except queue.Full:
                pass
            end = time.time()
            self.ev_cancel.wait(max(0, self.interval - (end - start)))
        self.queue.close()
        


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
        axes = body_to_inertial(axes.T, dcm).T
        axes += position
        # axes[1::2] = position
        # axes[0::2] = dcm.T + position
        for i, l in enumerate(drawing.axis_lines):
            l.set_data(axes[i*2:i*2+2,0], axes[i*2:i*2+2,1])
            l.set_3d_properties(axes[i*2:i*2+2,2])

    return (*drawing.arm_lines, drawing.trajectory_line, *drawing.axis_lines)



def make_fig(xlim, ylim, zlim) -> Tuple[plt.Figure, Axes3D]:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return fig, ax



def is_terminal() -> bool:
    # https://stackoverflow.com/q/15411967/4591810
    # https://stackoverflow.com/a/39662359/4591810
    try:
        import IPython
        shell = IPython.get_ipython().__class__.__name__
        if ('Terminal' in shell) or ('terminal' in shell) or (shell == 'NoneType'):
            return True
        else:
            return False
    except NameError:
        return True
    except ImportError:
        return True
