import json 
from glob import glob
import numpy as np
import pandas as pd


class Calibrator:
	"""
	[recommanded format]
	simulation:
		sim_1.csv
		sim_2.csv
	experience(phantom):
		phantom_1.csv
		phantom_2.csv
	experience(live):
		live_1.csv
		live_2.csv


	only use these function:
		run(plot=True, cross_validation=True)
			1. fit the parameters a and b for y = a*x + b
			2. calibrate the live spectrum
		calibrate(live)
			1. once the parameters is done, calibrate the live spectrum
		reload(config_path)
			1. reload the data for calibration

	"""
	def __init__(self, config_path="config.json"):

		with open(config_path, 'rb') as f:
			config = json.load(f)

		# parameters
		self.wavelength = pd.read_csv(config["wavelength"]).values
		self.num_fiber = config["num_fiber"]
		self.num_phantom = config["num_phantom"]

		# data
		# [num_experience/simulation, wavelength]
		self.exp_phantom_data = self._load(config["exp_phantom_path"])
		self.exp_live_data = self._load(config["exp_live_path"])
		self.sim_data = self._load(config["sim_phantom_path"])

		assert len(self.exp_phantom_data) == len(self.sim_data),\
		"The simulated data and experience data has to matched!"

		# result
		# y = ax + b
		self.a = None
		self.b = None
		self.r_square = None
		self.calibrated = None

	def run(self, plot=True, cross_validate=True):

		phantom, sim, live = self._interpolate_all()
		coefficients, r_square = self._fit(phantom, sim)
		print('R_square: %.4f' % r_square)

		if cross_validate:
			self._cross_validate(phantom, sim)

		self.calibrate(live)

		self._save_result()

	def reload(self, config_path):

		self.__init__(config_path)

	def calibrate(self, live):
		assert self.a is not None, "Must fit the phantom first, try run()"
		assert self.b is not None, "Must fit the phantom first, try run()"

		self.calibrated = live * self.a + self.b

		return self.calibrated

	def _load(self, path):
		data_list = glob(path + '*.csv')
		data_list.sort()
		data = []
		for d in data_list:
			data.append(pd.read_csv(d, header=None).values)

		# [num_experience/simulation, [num_wavelength, 2]]
		# list contain 2D-ndarray
		return data 

	def _interpolate_all(self):
		phantom = []
		sim = []
		live = []

		for data in self.exp_phantom_data:
			p = np.interp(self.wavelength, data.T[0], data.T[1])
			phantom.append(p)

		for data in self.sim_data:
			s = np.interp(self.wavelength, data.T[0], data.T[1])
			sim.append(s)

		for data in self.exp_live_data:
			l = np.interp(self.wavelength, data.T[0], data.T[1])
			live.append(l)

		phantom = np.asarray(phantom)
		sim = np.asarray(sim)
		live = np.asarray(live)

		# [num_experience/simulation, num_wavelength]
		return phantom, sim, live

	def _fit(self, phantom, sim, save=True):
		coefficients = []
		for p, s in zip(phantom.T, sim.T):
			c = np.polyfit(p, s, 1)	# linear fit
			coefficients.append(c)

		coefficients = np.asarray(coefficients)

		a = coefficients.T[0]
		b = coefficients.T[1]			

		r_square = []
		for x, y in zip(phantom, sim):
			y_fit = x * a + b
			residual = ((y-y_fit)**2).sum()
			SS_total = ((y.mean()-y_fit)**2).sum()

			r_square.append(1 - residual/SS_total)

		r_square = np.mean(r_square)

		# save the fitting result
		if save:
			self.a = a
			self.b = b
			self.r_square = r_square

		# [num_wavelength, [2(a,b)]], scalar
		return coefficients, r_square	

	def _cross_validate(self, phantom, sim):
		num = phantom.shape[0]

		test = []
		for idx in range(num):
			filter_list = [i for i in range(num) if i != idx]
			phantom_one_out = phantom[filter_list]
			sim_one_out = sim[filter_list]

			c, r = self._fit(phantom_out_out, sim_one_out, save=False)
			print('removing #%d.. R square: %.4f' % (idx, r))
			test.append((c, r))

		test.sort(key=lambda x: x[1], reverse=True)
		if test[0][1] > self.r_square:
			print('change the result..')
			self.a = test[0][0].T[0]
			self.b = test[0][0].T[1]
			self.r_square = test[0][1]

	def _save_result(self):
		df = pd.DataFrame(self.calibrated)
		df.to_csv("calibrated.csv")


if __name__ == "__main__":
	pass
