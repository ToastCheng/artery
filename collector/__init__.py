import os
import json
import pickle 
from glob import glob

import numpy as np 
import pandas as pd 




class Collector:
	"""
	A class designed for collect the mcx output

	"""
	def __init__(self, in_path, out_path):
		"""
		in_path: 

		out_path:

		"""
		self.in_path = in_path
		self.out_path = out_path

		# parameters include: 
		# absorb: b, s, w
		# scatter: fray, bmie, muspx 
		# tissue type: IJV, CCA, muscle, skin

	def collect(self, num=50):
		
		scv = []
		spectrum = []
		geometry = []
		parameter = []

		for idx in range(num):

			# load result
			path = os.path.join(self.in_path, "train_%d" % idx, "result", "result.pkl")

			# [101, sds, wavelength]
			result = np.load(path)

			# [101, wavelength] (select the last sds)
			result = result[:, -1, :]

			# append..
			scv += [i for i in range(101)]
			for r in result:
				spectrum.append(r)

			# load geometry and other optical parameters
			path = "output_first_train/generator/parameter/parameter_%d.json" % idx
			g, p = self._get_data(path)

			for i in range(101):
				geometry.append(g)
				parameter.append(p)

		scv = np.asarray(scv)
		scv = np.expand_dims(scv, axis=1)
		spectrum = np.asarray(spectrum)
		geometry = np.asarray(geometry)
		parameter = np.asarray(parameter)

		with open(os.path.join(self.out_path, "scv.pkl"), "wb+") as f:
			pickle.dump(scv, f)

		with open(os.path.join(self.out_path, "spectrum.pkl"), "wb+") as f:
			pickle.dump(spectrum, f)

		with open(os.path.join(self.out_path, "geometry.pkl"), "wb+") as f:
			pickle.dump(geometry, f)

		with open(os.path.join(self.out_path, "parameter.pkl"), "wb+") as f:
			pickle.dump(parameter, f)


	def _get_data(self, path):
		with open(path) as f:
			d = json.load(f)

		parameter = [
			d["skin"]["blood_volume_fraction"], d["skin"]["ScvO2"], d["skin"]["water_volume"], d["skin"]["fray"], d["skin"]["bmie"], d["skin"]["muspx"], \
			d["muscle"]["blood_volume_fraction"], d["muscle"]["ScvO2"], d["muscle"]["water_volume"], d["muscle"]["fray"], d["muscle"]["bmie"], d["muscle"]["muspx"], \
			d["IJV"]["blood_volume_fraction"], d["IJV"]["water_volume"], d["IJV"]["fray"], d["IJV"]["bmie"], d["IJV"]["muspx"], \
			d["CCA"]["blood_volume_fraction"], d["CCA"]["ScvO2"], d["CCA"]["water_volume"], d["CCA"]["fray"], d["CCA"]["bmie"], d["CCA"]["muspx"]
		]

		geometry = [
			d["geometry"]["skin_thickness"], d["geometry"]["ijv_radius"], \
			d["geometry"]["ijv_depth"], d["geometry"]["cca_radius"], \
			d["geometry"]["cca_depth"], d["geometry"]["ijv_cca_distance"]
		]

		return geometry, parameter




