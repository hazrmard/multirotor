from typing import Tuple, Union, Dict
import threading as th
import multiprocessing as mp
import queue
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.lines import Line2D

from .coords import body_to_inertial, direction_cosine_matrix
from .simulation import Multirotor
from .vehicle import VehicleParams
from .helpers import DataLog



def plot_datalog(log: DataLog, figsize=(21,10.5),
    plots=('pos', 'vel', 'ctrl', 'traj'),
    nrows=2, ncols=None) -> Dict[str, plt.Axes]:
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
    
    Returns
    -------
    Dict : Dict[str, plt.Axes]
        A dictionary of plot names mapping to Axes
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
    axes = {}

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
        axes['pos'] = plt.gca()

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
        axes['vel'] = plt.gca()

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
        axes['ctrl'] = plt.gca()

    if 'traj' in plots:
        plt.subplot(*plot_grid, plot_number)
        if len(log.target.position) > 0:
            plt.plot(log.target.position[:,0], log.target.position[:,1], label='Prescribed traj', ls=':')
        plt.plot(log.x, log.y, label='Actual traj', ls='-')
        plt.gca().set_aspect('equal', 'box')
        plt.title('XY positions /m')
        plt.xlabel('X /m')
        plt.ylabel('Y /m')
        plt.legend()
        plot_number += 1
        axes['traj'] = plt.gca()

    plt.tight_layout()
    return axes



def get_wind_quiver(heading: str, ax: plt.Axes, n=5, dim=2):
    """
    Create arrays of x,y,z coordinates for a quiver plot of wind.

    Parameters
    ----------
    heading : str
        The heading of the wind, e.g. '5@45' for 5N wind from 45 degrees.
    ax : plt.Axes
        The axes on which to plot the quiver.
    n : int, optional
        Size of arrays, by default 5
    dim : int, optional
        Dimension of quiver (2 or 3), by default 2

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        x,y,[z],dx,dy,[dz] coordinates for quiver plot.
    """
    magnitude, angle = heading.split('@')
    magnitude = float(magnitude)
    if magnitude==0:
        return (0,0,0,0) if dim==2 else (0,0,0,0,0,0)
    angle = float(angle) * np.pi / 180
    dx, dy = -np.cos(angle), -np.sin(angle)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if dim==3:
        zlim = ax.get_zlim()
        dz = 0
        x,y,z = np.meshgrid(np.linspace(*xlim, num=n), np.linspace(*ylim, num=n), np.linspace(*zlim, num=n),
                      indexing='xy')
        return x,y,z,dx,dy,dz
    else:
        x,y = np.meshgrid(np.linspace(*xlim, num=n), np.linspace(*ylim, num=n),
                      indexing='xy')
        return x,y,dx,dy



class VehicleDrawing:
    """
    A 3D representation of the vehicle.
    """

    def __init__(self,
        vehicle: Multirotor,
        axis: Axes3D = None,
        updates_per_second: float = 30.,
        trace: bool = False,
        body_axes: bool = False,
        connection_via: str = 'animation'
    ):
        """

        Parameters
        ----------
        vehicle : Multirotor
            The vehicle instance to draw
        axis : Axes3D, optional
            The axes on which to draw, by default None
        updates_per_second : float, optional
            Maximum update interval of plot, by default 30.
        trace : bool, optional
            Whether to draw a dotted line showing past positions, by default False
        body_axes : bool, optional
            Whether to draw the body-frame axes, by default False
        connection_via : str, optional
            The drawing update method, either 'animation' (using matplotlib FuncAnimation), or
            'thread' using python threading, by default 'animation'
        """
        self.vehicle = vehicle
        self.axis = axis
        self.updates_per_second = updates_per_second
        self.trace = trace
        self.body_axes = body_axes
        self.connection_via = connection_via

        self.t = self.vehicle.t
        self.interval = 1 / self.updates_per_second
        self.is_terminal = is_terminal()
        self.arm_lines, self.arm_lines_points, self.trajectory_line, \
        self.axis_lines, self.axis_lines_points = \
            make_drawing(self.vehicle.params, self.body_axes)
        if self.axis is None:
            self.figure, self.axis = make_fig((-10,10), (-10,10), (-10,10))
        else:
            self.figure = self.axis.figure, self.axis = axis
        self.trajectory = [[], [], []] # [[X,..], [Y,...], [Z,...]]
        self.ev_cancel = th.Event()
        self.ev_new = th.Event()
        self.queue = queue.Queue(maxsize=1)
        self.sync = False
        self.update_thread = th.Thread(target=self._update_worker, daemon=True, name='update_thread')
        self.update_thread.start()
        # self.animate_thread = th.Thread(target=self._animator, daemon=True, name='animate_thread')
        # self.animate_thread.start()
        self._animator()


    def connect(self) -> Union[FuncAnimation, th.Thread]:
        """Create a drawing that updates with the `self.vehicle` instance.

        Returns
        -------
        Union[FuncAnimation, th.Thread]
            The `FuncAnimation` or `Thread` class being used to update the plot.
        """
        # self.update_thread.start()
        self.sync = True
        self.ev_new.set()
        return
        if self.connection_via=='animation':
            return self._animator()
        else:
            raise NotImplementedError()


    def update(self):
        self.ev_new.set()


    def disconnect(self, force=False):
        """Disconnect the plot from the update function. Should be called after no interaction
        is needed.

        Parameters
        ----------
        force : bool, optional
            Whether to force disconnection. Not implemented, by default False
        """
        self.ev_cancel.set()
        if self.connection_via == 'thread':
            # self.thread.join(timeout=1.5 * self.interval)
            self.update_thread.join(timeout=1.5 * self.interval)
            try:
                del self.queue
                del self.update_thread
            except AttributeError:
                pass # Means queue and update_thread are already deleted
        elif self.connection_via=='animation':
            try:
                self.anim.pause()
                del self.anim
            except AttributeError:
                # If the interactive jupyter figure is closed, this error is
                # raised if disconnect is called after.
                pass
        elif self.connection_via=='process':
            raise NotImplementedError()


    def _init_func(self) -> Tuple[Line3D]:
        """Add vehicle and trajectory lines to plot by initializing the Artist objects
        in `self.axis`.

        Returns
        -------
        Tuple[Line3d]
        """
        for l in self.arm_lines:
            self.axis.add_line(l)
        self.axis.add_line(self.trajectory_line)
        self.trajectory = [[self.vehicle.position[0]],[self.vehicle.position[1]], [self.vehicle.position[0]]]
        self.trajectory_line.set_data([self.vehicle.position[0]], [self.vehicle.position[1]])
        self.trajectory_line.set_3d_properties([self.vehicle.position[2]])
        for l in self.axis_lines:
            self.axis.add_line(l)
        return (*self.arm_lines, self.trajectory_line, *self.axis_lines)


    def _update_drawing_from_queue(self, frame: int=None):
        try:
            vehicle_t, position, orientation = self.queue.get(block=False)
            if vehicle_t != self.t:
                if vehicle_t < self.t:
                    self.trajectory = [[position[0]], [position[1]], [position[2]]] # likely vehicle is reset, so reset trajectory
                self.t = vehicle_t
                res = update_drawing(self, position, orientation)
            else:
                # if there is no change in time, then return the old lines
                res = (*self.arm_lines, self.trajectory_line, *self.axis_lines)
            self.queue.task_done()
        except queue.Empty:
            res = (*self.arm_lines, self.trajectory_line, *self.axis_lines)
        return res


    def _generator(self):
        while not self.ev_cancel.is_set():
            yield self._update_drawing_from_queue()

        i = 0
        while not self.ev_cancel.is_set():
            start = time.time()
            self._update_drawing_from_queue(i)
            # self.figure.canvas.draw_idle()
            # self.figure.canvas.flush_events()
            i += 1
            end = time.time()
            pause = max(1e-3, self.interval - (end - start))
            plt.pause(pause)
            # self.ev_cancel.wait(self.interval - (end - start))


    def _animator(self):
        print('Called _animator')
        self.anim = FuncAnimation(
            self.figure,
            func= lambda x: x,
            frames = self._generator,
            init_func=self._init_func,
            blit=True,
            repeat=False,
            interval = self.interval * 1000,
            save_count=0,
            cache_frame_data=False)
        return self.anim


    def _update_worker(self):
        # Thread that puts time, position, orientation into thread-/process-safe
        # queue.
        while not self.ev_cancel.is_set():
            start = time.time()
            if not self.sync:
                self.ev_new.wait()
            try:
                self.queue.put(
                    (self.vehicle.t, self.vehicle.position, self.vehicle.orientation),
                    timeout=self.interval if self.sync else None)
            except queue.Full:
                pass
            if not self.sync:
                self.ev_new.clear()
            else:
                time.sleep(max(0, self.interval - (start-time.time())))
            # self.ev_cancel.wait(max(0, self.interval - (end - start)))
        


def make_drawing(params: VehicleParams, body_axes: bool=False, make_2d: bool=False, scale_arms=1.):
    Line = Line3D if not make_2d else Line2D
    arm_lines_points = np.zeros((len(params.propellers) * 2, 2 if make_2d else 3)) # [2 points/ propeller, axis]
    x = params.distances * np.cos(params.angles)
    y = params.distances * np.sin(params.angles)
    arm_lines_points[1::2,0] = x
    arm_lines_points[1::2,1] = y
    arm_lines_points *=  scale_arms
    arm_lines = []
    for i in range(len(params.propellers)):
        arm_lines.append(
            Line(
                arm_lines_points[2*i:2*i+2,0],
                arm_lines_points[2*i:2*i+2,1],
                antialiased=False,
                **({'zs':arm_lines_points[2*i:2*i+2,2]} if not make_2d else {}),
            )
        )

    trajectory_line = Line([], [], linewidth=0.5, color='black', linestyle=':',
                           **({'zs':[]} if not make_2d else {}))

    axis_lines_points = np.zeros((6, 3)) # [2 points/ axis, axis]
    axis_lines_points[1::2] = np.eye(3)
    axis_lines = []
    for i, c in enumerate(['r', 'g', 'b']):
        if body_axes:
            axis_lines.append(Line(
                axis_lines_points[2*i:2*i+2,0],
                axis_lines_points[2*i:2*i+2,1],
                antialiased=False,
                linewidth=0.5,
                color=c,
                **({'zs':axis_lines_points[2*i:2*i+2,2]} if not make_2d else {}),
            )) 
        else:
            axis_lines.append(Line([], [], **({'zs':[]} if not make_2d else {})))

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



def make_fig(
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float]
    ) -> Tuple[plt.Figure, Axes3D]:
    """Convenience function for creating a 3D axis.

    Parameters
    ----------
    xlim/ylim/zlim : Tuple[float, float]
        Min/max coordinates in plot

    Returns
    -------
    Tuple[plt.Figure, Axes3D]
        The figure and Axes3D instance
    """
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
