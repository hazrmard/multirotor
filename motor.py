import scipy.integrate
import numpy as np

from vehicle import MotorParams, SimulationParams

"""
This Script simulates a motor object class that can receive a voltage or current amount and
convert voltage into an angular speed based on the specifications of the 
Tarot T18 Octocopter motor model.
"""

class Motor:

	def __init__(self, params: MotorParams, simulation: SimulationParams):
		self.params = params
		self.simulation = simulation
		self.reset()
		
	
	def step(self, voltage: float) -> float:
		
		# ----------------------------
		# 		UPDATE
		# Returns a motor's angular velocity moving one step in time 
		# with a given voltage. Takes in as parameters, voltage and sample rate
		# ----------------------------
		
		self.t += self.simulation.dt
		self.v = voltage
		self.ode.set_initial_value(self.omega, 0)
		self.speed = self.ode.integrate(self.ode.t + self.simulation.dt)
		return self.speed
		



	def omega_dot_i(self, time, state):
		
		# ----------------------------
		#		OMEGA_DOT_I
		# Helper Method to calculate omega_dot for our ode integrator.
		# Can be written as a lambda function inside update for other shorter motors
		# ----------------------------
		
		rpm = self.speed * 30 / np.pi
		
		t1 = self.motor_constant / self.params.r * self.v
		t2 = -((2.138e-08)*rpm**2 + (-1.279e-05)*rpm)
		t3 = -(self.params.k_m * self.params.k_q / self.params.r * self.speed)
		
		dspeed = (t1 + t2 + t3) / self.params.moment_of_inertia
		
		return dspeed


	def reset(self):
		self.ode = scipy.integrate.ode(self.omega_dot_i).set_integrator('vode', method='bdf')
		self.t = 0.
		self.speed = 0.
		return self.speed
