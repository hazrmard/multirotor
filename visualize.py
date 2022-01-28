from typing import Tuple
import threading as th
from dataclasses import dataclass
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

from coords import direction_cosine_matrix, inertial_to_body
from vehicle import VehicleParams
from simulation import Multirotor



@dataclass
class VehicleDrawing:
    vehicle: Multirotor
    axis: Axes3D = None
    max_frames_per_second: float = 30.
    trace: bool = False

    def __post_init__(self):
        self.params = self.vehicle.params
        self.lines, self.body_frame_segments, self.trajectory_line = make_drawing(self.params)
        self.interval = 1 / self.max_frames_per_second
        self.is_ipython = is_ipython()
        if self.trace:
            self.trajectory = [[], [], []] # [[X,..], [Y,...], [Z,...]]
        else:
            self.ipython_display = None


    def connect(self):
        self.called = 0
        self.ev_cancel = th.Event()
        self.thread = th.Thread(target=self._worker, daemon=False)
        self.thread.start()


    def disconnect(self, force=False):
        self.ev_cancel.set()
        self.thread.join(timeout=1.5 * self.interval)


    def _worker(self):
        if self.axis is None:
            self.figure, self.axis = make_fig((-10,10), (-10,10), (0,20))
        else:
            self.figure = self.axis.figure

        for l in self.lines:
            self.axis.add_line(l)
        self.axis.add_line(self.trajectory_line)
        
        # if self.is_ipython:
        #     from IPython.display import display
        #     self.ipython_display = display(self.figure, display_id=True)
        # else:
        self.ipython_display = None
        self.figure.show()

        while not self.ev_cancel.is_set():
            start = time.time()
            position = self.vehicle.position
            orientation = self.vehicle.orientation
            update_drawing(self, position, orientation)
            end = time.time()
            self.ev_cancel.wait(self.interval - (end - start))
        


def make_drawing(params: VehicleParams):
    arms = np.zeros((len(params.propellers) * 2, 3)) # [2 points/ propeller, axis]
    x = params.distances * np.cos(params.angles)
    y = params.distances * np.sin(params.angles)
    arms[1::2,0] = x
    arms[1::2,1] = y
    lines = []
    for i in range(len(params.propellers)):
        lines.append(
            Line3D(
                arms[2*i:2*i+2,0],
                arms[2*i:2*i+2,1],
                arms[2*i:2*i+2,2],
                antialiased=False))
    trajectory = Line3D([], [], [])
    return lines, arms, trajectory



def update_drawing(drawing: VehicleDrawing, position: np.ndarray, orientation: np.ndarray):
    dcm = direction_cosine_matrix(orientation[0], orientation[1], orientation[2])
    arms = np.copy(drawing.body_frame_segments)
    arms = inertial_to_body(arms.T, dcm).T
    arms += position
    for i, l in enumerate(drawing.lines):
        j = 2*i
        l.set_data(arms[j:j+2, 0], arms[j:j+2, 1])
        l.set_3d_properties(arms[j:j+2, 2])
    
    if drawing.trace:
        drawing.trajectory[0].append(position[0])
        drawing.trajectory[1].append(position[1])
        drawing.trajectory[2].append(position[2])
        drawing.trajectory_line.set_data(drawing.trajectory[0], drawing.trajectory[1])
        drawing.trajectory_line.set_3d_properties(drawing.trajectory[2])

    drawing.figure.canvas.draw_idle()
    drawing.figure.canvas.flush_events()
    plt.pause(1e-7)

    if drawing.ipython_display is not None:
        drawing.ipython_display.update(drawing.figure)



def make_fig(xlim, ylim, zlim) -> Tuple[plt.Figure, Axes3D]:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', xlim=xlim, ylim=ylim, zlim=zlim)
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
