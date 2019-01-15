import os 
import json 
from glob import glob 
from random import uniform
from collections import namedtuple
from collections import defaultdict

import numpy as np 
import matplotlib.pyplot as plt



class Generator:
	"""
	Author: Shih Cheng Tu

	A helper class that generate random input for MCX
	"""
	def __init__(self):
		
		# a light-weight structure (using collections.namedtuple)
		self.Parameter = namedtuple('parameters', ['min', 'max'])

		# a dict that contains parameters
		self.parameters = defaultdict(dict)


	def run(self, config_file=os.path.join("generator", "config_generate.json"), idx=0):
		if config_file is not None:
			self._load(config_file)

		assert len(self.parameters) > 0, "no parameter is set in the Generator"

		with open(os.path.join("generator", "parameters_gen.json")) as f:
			parameters = json.load(f)


		for tissue_name, parameter_dict in self.parameters.items():
			for parameter_name, value in parameter_dict.items():
				# random sample
				parameters[tissue_name][parameter_name] = uniform(value.min, value.max)

		# step by step setting: skin --> ijv radius, depth --> cca radius, depth --> distance 

		parameters["geometry"]["skin_thickness"] = uniform(
			self.parameters["geometry"]["skin_thickness"].min, 
			self.parameters["geometry"]["skin_thickness"].max
			)

		parameters["geometry"]["ijv_radius"] = uniform(
			self.parameters["geometry"]["ijv_radius"].min, 
			self.parameters["geometry"]["ijv_radius"].max
			)

		if self.parameters["geometry"]["ijv_depth"].min > parameters["geometry"]["skin_thickness"] + parameters["geometry"]["ijv_radius"]:
			check = self.parameters["geometry"]["ijv_depth"].min
		else:
			check = parameters["geometry"]["skin_thickness"] + parameters["geometry"]["ijv_radius"]
		parameters["geometry"]["ijv_depth"] = uniform(
			check, 
			check + self.parameters["geometry"]["ijv_depth"].max - self.parameters["geometry"]["ijv_depth"].min
			)

		parameters["geometry"]["cca_radius"] = uniform(
			self.parameters["geometry"]["cca_radius"].min, 
			self.parameters["geometry"]["cca_radius"].max
			)
		
		# if self.parameters["geometry"]["cca_depth"].min > parameters["geometry"]["skin_thickness"] + parameters["geometry"]["cca_radius"]:
		# 	check = self.parameters["geometry"]["cca_depth"].min
		# else:
		# 	check = parameters["geometry"]["skin_thickness"] + parameters["geometry"]["cca_radius"]
		# parameters["geometry"]["cca_depth"] = uniform(
		# 	check,
		# 	check + self.parameters["geometry"]["cca_depth"].max - self.parameters["geometry"]["cca_depth"].min
		# 	)

		parameters["geometry"]["cca_depth"] = uniform(
			parameters["geometry"]["ijv_depth"],
			self.parameters["geometry"]["cca_depth"].max
			)


		if (parameters["geometry"]["cca_radius"] + parameters["geometry"]["ijv_radius"])**2 -\
		(parameters["geometry"]["cca_depth"] - parameters["geometry"]["ijv_depth"])**2 >\
		(self.parameters["geometry"]["ijv_cca_distance"].min)**2:
			check = np.sqrt((parameters["geometry"]["cca_radius"] + parameters["geometry"]["ijv_radius"])**2 -\
		(parameters["geometry"]["cca_depth"] - parameters["geometry"]["ijv_depth"])**2)
		else:
			check = self.parameters["geometry"]["ijv_cca_distance"].min
		parameters["geometry"]["ijv_cca_distance"] = uniform(
			check,
			check + self.parameters["geometry"]["ijv_cca_distance"].max - self.parameters["geometry"]["ijv_cca_distance"].min

			)

		file_count = len(glob(os.path.join("generator", "parameter", "*")))
		with open(os.path.join("generator", "parameter", "parameter_%d.json" % file_count), 'w+') as f:
			json.dump(parameters, f, indent=4)

		# save the geometry vector to geometry/
		with open(os.path.join("generator", "geometry", "geometry.csv"), 'a+') as f:
			f.write("%d,%f,%f,%f,%f,%f,%f\n" % \
				(file_count,\
					parameters["geometry"]["skin_thickness"],\
					parameters["geometry"]["ijv_radius"],\
					parameters["geometry"]["ijv_depth"],\
					parameters["geometry"]["cca_radius"],\
					parameters["geometry"]["cca_depth"],\
					parameters["geometry"]["ijv_cca_distance"]))

		# plot figure
		skin = plt.Rectangle((0, 0), 100, self._convert_unit(parameters["geometry"]["skin_thickness"]), fc="#d1a16e")
		muscle = plt.Rectangle((0, self._convert_unit(parameters["geometry"]["skin_thickness"])), 100, 300-self._convert_unit(parameters["geometry"]["skin_thickness"]), fc="#ea6935")
		ijv = plt.Circle((50, self._convert_unit(parameters["geometry"]["ijv_depth"])), radius=self._convert_unit(parameters["geometry"]["ijv_radius"]), fc="#437ddb")
		cca = plt.Circle((50-self._convert_unit(parameters["geometry"]["ijv_cca_distance"]), self._convert_unit(parameters["geometry"]["cca_depth"])), radius=self._convert_unit(parameters["geometry"]["cca_radius"]), fc="#c61f28")
		
		plt.axis([0, 100, 100, 0])
		plt.gca().add_patch(skin)
		plt.gca().add_patch(muscle)
		plt.gca().add_patch(ijv)
		plt.gca().add_patch(cca)
		# plt.show()
		plt.savefig(os.path.join("generator", "plot", "geometry_%d.png" % idx))

	def _convert_unit(self, length_mm):
		# convert mm to number of grid
		num_grid = length_mm//0.25
		return num_grid
		
	def add_parameter(self, tissue_name, parameter_name, min, max):
		self.parameters[tissue_name][parameter_name] = self.Parameter(min, max)

	def _load(self, config_file):
		with open(config_file) as f:
			config = json.load(f)

		for tissue_name, parameter_dict in config.items():
			for parameter_name, value in parameter_dict.items():
				self.add_parameter(tissue_name, parameter_name, value[0], value[1])


def test():
	g = Generator()
	g.run()


if __name__ == "__main__":
	from pprint import PrettyPrinter
	pp = PrettyPrinter()

	g = Generator()
	for i in range(100):
		g.run(idx=i)

	# pp.pprint(g.parameters)
# exec(open('__init__.py').read())
