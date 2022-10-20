"""
Guided mode control. Generates x,y,z,yaw references for PID controllers.
"""

import numpy as np



class GuidedTrajectory:


    def __init__(self, vehicle, controller) -> None:
        self.vehicle = vehicle
        self.controller = controller
        self.waypoints = []


    def add_waypoint(self, waypoint: np.ndarray):
        self.waypoints.append(np.asarray(waypoint))


    def __iter__(self):
        pass