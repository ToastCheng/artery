import os
import json
import pickle
from glob import glob 
from random import randint
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

from utils import load_mc2
from mchhandler import MCHHandler



class MCX:
	# a class that use as an interface
	def __init__(self, config_file="config.json"):
		with open(config_file) as f:
			self.config = json.load(f)
		with open(self.config["input_file"]) as f:
			self.parameters = json.load(f)
			for key in self.parameters["geometry"].keys():
				self.parameters["geometry"][key] = self._convert_unit(self.parameters["geometry"][key])
				print(key, self.parameters["geometry"][key])
		with open(self.config["geometry_file"]) as f:
			self.mcx_input = json.load(f)

		self.wavelength = pd.read_csv(self.config["wavelength_file"])["wavelength"]
		self.mua = pd.read_csv(self.config["absorb_file"])
		self.fiber = pd.read_csv(self.config["fiber_file"])

		self.handler = MCHHandler(config=config_file)

		# handle the situation that batch > total
		if self.config["photon_batch"] > self.config["num_photon"]:
			self.config["photon_batch"] = self.config["num_photon"] 

		# mua
		oxy = self.mua['oxy'].values
		deoxy = self.mua['deoxy'].values
		water = self.mua['water'].values
		collagen = self.mua['collagen'].values
		fat = self.mua['fat'].values
		melanin = self.mua['mel'].values
		wl = self.mua['wavelength'].values

		# interpolation
		oxy = np.interp(self.wavelength, wl, oxy)
		deoxy = np.interp(self.wavelength, wl, deoxy)
		collagen = np.interp(self.wavelength, wl, collagen)
		water = np.interp(self.wavelength, wl, water)
		fat = np.interp(self.wavelength, wl, fat)
		melanin = np.interp(self.wavelength, wl, melanin)


		# turn the unit 1/cm --> 1/mm
		self.oxy = oxy * 0.1
		self.deoxy = deoxy * 0.1
		self.water = water * 0.1
		self.collagen = collagen * 0.1
		self.fat = fat * 0.1
		self.melanin = melanin * 0.1


		self.session = self.session = os.path.join("output", self.config["session_id"])
		self.plot = os.path.join(self.session, 'plot')
		self.plot_mc2 = os.path.join(self.session, 'plot_mc2')
		self.result = os.path.join(self.session, 'result')
		self.mcx_output = os.path.join(self.session, 'mcx_output')

		self.reflectance = None


	# public functions

	def run(self, white=True):
		# run the MCX simulation
		
		# session name
		if not os.path.isdir(self.session):
			os.mkdir(self.session)

		# directory for saving plot
		if not os.path.isdir(self.plot):
			os.mkdir(self.plot)

		# directory for saving mc2 plot
		if not os.path.isdir(self.plot_mc2):
			os.mkdir(self.plot_mc2)


		# directory for saving result
		if not os.path.isdir(self.result):
			os.mkdir(self.result)

		if not os.path.isdir(self.mcx_output):
			os.mkdir(self.mcx_output)

		# plot figure
		skin_th = self.parameters["geometry"]["skin_thickness"]
		fat_th = self.parameters["geometry"]["fat_thickness"]
		artery_r = self.parameters["geometry"]["artery_radius"]

		skin = plt.Rectangle((0, 0), 50, skin_th, fc="#d1a16e")
		fat = plt.Rectangle((0, skin_th), 50, fat_th, fc="#b1814e")
		muscle = plt.Rectangle((0, skin_th+fat_th), 50, 20-skin_th-fat_th, fc="#ea6935")
		artery = plt.Circle((25, skin_th+fat_th+artery_r), radius=artery_r, fc="#437ddb")

		plt.axis([0, 50, 20, 0])
		plt.gca().add_patch(skin)
		plt.gca().add_patch(fat)
		plt.gca().add_patch(muscle)
		plt.gca().add_patch(artery)
		plt.gca().set_aspect(1)
		plt.savefig(os.path.join(self.plot, self.config["session_id"] + "_geometry.png"))
		plt.close()


		for idx, wl in enumerate(self.wavelength):

			for sds_idx in range(len(self.fiber)):

				if white:
					self._make_mcx_input_white(idx, sds_idx)
				else:
					self._make_mcx_input(idx)

				command = self._get_command(wl, self.fiber.values[sds_idx][0])

				print("wavelength: ", wl)
				print("sds: ", self.fiber.values[sds_idx][0])
				print(command)
				os.chdir("mcx/bin")
				os.system(command)
				os.chdir("../..")

		mc2_list = glob(os.path.join(self.mcx_output, "*.mc2"))

		for mc2 in mc2_list:
			fig = plt.figure(figsize=(10,16))
			d = load_mc2(mc2, [100, 100, 300])
			plt.imshow(d[50,:,:100].T)
			name = mc2.split('/')[-1].split('.')[0]
			plt.title(name)
			plt.xlabel('y axis')
			plt.ylabel('z axis')
			plt.savefig(os.path.join(self.plot_mc2, name + ".png"))
			plt.close()

	def calculate_reflectance(self, white=True, plot=True, verbose=True, save=True):
		# This function should called after 
		# mcx.run(), or the .mch files are outputed by mcx
		results = []
		portions = defaultdict(list)
		if white:	
			for wl in self.wavelength:
				resultss = []
				for sds, r in self.fiber.values:
					# [1, ScvO2]
					result, portion = self.handler.compute_reflectance_white(
						wl=wl, 
						mch_file=os.path.join(self.mcx_output, "%s_%d_%d.mch" % (self.config["session_id"], wl, sds))
						)

					if result is None:
						continue

					portions["wavelength"].append(wl)
					portions["sds"].append(sds)
					portions["skin"].append(portion[0])
					portions["fat"].append(portion[1])
					portions["muscle"].append(portion[2])
					portions["artery"].append(portion[3])

					# [SDS, 1, ScvO2]
					resultss.append(result)
				# [wl, SDS, 1, ScvO2] 
				results.append(resultss)




			# [wl, SDS, ScvO2]
			# -> [ScvO2, SDS, wl]
			results = np.asarray(results).transpose(2, 1, 0)
			self.reflectance = results
			# print(results.shape)

		else:
			for wl in self.wavelength:
				result, portion = self.handler.compute_reflectance(
					wl=wl,
					mch_file=os.path.join(self.mcx_output, "%s_%d.mch" % (self.config["session_id"], wl))
					)
				if result is None:
					continue
				results.append(result)

				portions["wavelength"].append(wl)
				portions["skin"].append(portion[0])
				portions["fat"].append(portion[1])
				portions["muscle"].append(portion[2])
				portions["artery"].append(portion[3])

			# [wl, SDS]
			# -> [SDS, wl]
			results = np.asarray(results).transpose(1, 0)
			self.reflectance = results
			# print(results.shape)


		# if plot:
		# 	if white:
		# 		for r_idx, r in enumerate(results):
		# 			fig = plt.figure()
		# 			for d_idx, d in enumerate(r):
		# 				plt.plot(self.wavelength, np.log(d), label="detector %d" % d_idx)
					
		# 			path = os.path.join(self.plot, "Scv_%d.png" % r_idx)
		# 			plt.title('Scv_%d' % r_idx)
		# 			plt.legend()
		# 			plt.savefig(path)
		# 			plt.close()
		# 	else:
		# 		pass

		# 	plt.clf()

		# 	path = os.path.join(self.plot, "portion.png")
		# 	plt.plot(portions["wavelength"], portions["skin"], label="skin")
		# 	plt.plot(portions["wavelength"], portions["fat"], label="fat")
		# 	plt.plot(portions["wavelength"], portions["muscle"], label="muscle")
		# 	plt.plot(portions["wavelength"], portions["artery"], label="artery")
		# 	plt.legend()
		# 	plt.xlabel('wavelength [nm]')
		# 	plt.ylabel('pathlength portion')
		# 	plt.savefig(path)
		# 	plt.close()

		if save:
			result_path = os.path.join(self.result, "result.pkl")
			with open(result_path, 'wb') as f:
				pickle.dump(results, f)

			portion_path = os.path.join(self.result, "portion.csv")
			df = pd.DataFrame(portions)
			df.to_csv(portion_path, index=False)


	def reload_result(self, path):
		with open(path, 'rb') as f:
			results = pickle.load(f)
		return results

	# private functions

	def _convert_unit(self, length_mm):
		# convert mm to number of grid
		num_grid = length_mm//self.config["voxel_size"]
		return int(num_grid)

	def _make_mcx_input_white(self, idx, sds_idx):
		mcx_input = self.mcx_input

		mcx_input["Session"]["ID"] = self.config["session_id"] + "_%d" % self.wavelength[idx]

		# optical parameter

		# 
		mcx_input["Domain"]["Media"][0]["mua"] = 0
		mcx_input["Domain"]["Media"][0]["mus"] = 0
		mcx_input["Domain"]["Media"][0]["g"] = 1
		mcx_input["Domain"]["Media"][0]["n"] = 1

		# skin
		mcx_input["Domain"]["Media"][1]["name"] = "skin"
		mcx_input["Domain"]["Media"][1]["mua"] = self._calculate_mua(
			idx, 
			self.parameters["skin"]["blood_volume_fraction"], 
			self.parameters["skin"]["ScvO2"], 
			self.parameters["skin"]["water_volume"],
			self.parameters["skin"]["fat_volume"],
			self.parameters["skin"]["melanin_volume"]
			)
		mcx_input["Domain"]["Media"][1]["mus"] = self._calculate_mus(
			idx,
			self.parameters["skin"]["muspx"], 
			self.parameters["skin"]["fray"], 
			self.parameters["skin"]["bmie"],
			self.parameters["skin"]["g"]
			)
		mcx_input["Domain"]["Media"][1]["g"] = self.parameters["skin"]["g"]
		mcx_input["Domain"]["Media"][1]["n"] = self.parameters["skin"]["n"]

		# fat
		mcx_input["Domain"]["Media"][3]["name"] = "fat"
		mcx_input["Domain"]["Media"][3]["mua"] = self._calculate_mua(
			idx, 
			self.parameters["fat"]["blood_volume_fraction"], 
			self.parameters["fat"]["ScvO2"], 
			self.parameters["fat"]["water_volume"],
			self.parameters["fat"]["fat_volume"],
			self.parameters["fat"]["melanin_volume"]
			)
		mcx_input["Domain"]["Media"][3]["mus"] = self._calculate_mus(
			idx,
			self.parameters["fat"]["muspx"], 
			self.parameters["fat"]["fray"], 
			self.parameters["fat"]["bmie"],
			self.parameters["fat"]["g"]
			)
		mcx_input["Domain"]["Media"][3]["g"] = self.parameters["fat"]["g"]
		mcx_input["Domain"]["Media"][3]["n"] = self.parameters["fat"]["n"]

		# muscle
		mcx_input["Domain"]["Media"][3]["name"] = "muscle"
		mcx_input["Domain"]["Media"][3]["mua"] = self._calculate_muscle_mua(
			idx, 
			self.parameters["muscle"]["water_volume"]
			)
		mcx_input["Domain"]["Media"][3]["mus"] = self._calculate_mus(
			idx,
			self.parameters["muscle"]["muspx"], 
			self.parameters["muscle"]["fray"], 
			self.parameters["muscle"]["bmie"],
			self.parameters["muscle"]["g"]
			)
		mcx_input["Domain"]["Media"][3]["g"] = self.parameters["muscle"]["g"]
		mcx_input["Domain"]["Media"][3]["n"] = self.parameters["muscle"]["n"]
		
		
		# artery
		mcx_input["Domain"]["Media"][4]["name"] = "artery"
		mcx_input["Domain"]["Media"][4]["mua"] = 0
		mcx_input["Domain"]["Media"][4]["mus"] = self._calculate_mus(
			idx,
			self.parameters["artery"]["muspx"], 
			self.parameters["artery"]["fray"], 
			self.parameters["artery"]["bmie"],
			self.parameters["artery"]["g"]
			)
		mcx_input["Domain"]["Media"][4]["g"] = self.parameters["artery"]["g"]
		mcx_input["Domain"]["Media"][4]["n"] = self.parameters["artery"]["n"]


		###########
		# geometry
		###########
		skin_th = self.parameters["geometry"]["skin_thickness"]
		fat_th = self.parameters["geometry"]["fat_thickness"]
		artery_r = self.parameters["geometry"]["artery_radius"]

		# skin
		mcx_input["Shapes"][1]["Subgrid"]["O"] = [1, 1, 1]
		mcx_input["Shapes"][1]["Subgrid"]["Size"] = [100, 100, skin_th]

		# fat
		mcx_input["Shapes"][2]["Subgrid"]["O"] = [1, 1, 1+skin_th]
		mcx_input["Shapes"][2]["Subgrid"]["Size"] = [100, 100, 300-skin_th]		

		# muscle
		mcx_input["Shapes"][3]["Subgrid"]["O"] = [1, 1, 1+skin_th+fat_th]
		mcx_input["Shapes"][3]["Subgrid"]["Size"] = [100, 100, 300-skin_th-fat_th]

		# artery 
		mcx_input["Shapes"][4]["Cylinder"]["C0"] = [100, 50, skin_th+fat_th+artery_r]
		mcx_input["Shapes"][4]["Cylinder"]["C1"] = [0, 50, skin_th+fat_th+artery_r]
		mcx_input["Shapes"][4]["Cylinder"]["R"] = artery_r

		# load fiber
		sds, r = self.fiber.values[sds_idx]
		sds = self._convert_unit(sds)
		r = self._convert_unit(r)

		mcx_input["Optode"]["Source"]["Pos"][1] = 50.0 - sds//2

		det = {
			"R": r,
			"Pos": [50.0, 50.0 + sds//2, 0.0]
		}
		mcx_input["Optode"]["Detector"].append(det)


		# set seed
		mcx_input["Session"]["RNGSeed"] = randint(0, 1000000000)





		# save the .json file in the output folder
		with open(self.config["geometry_file"], 'w+') as f:
			json.dump(mcx_input, f, indent=4)

	# def _make_mcx_input(self, idx):
	# 	mcx_input = self.mcx_input

	# 	mcx_input["Session"]["ID"] = self.config["session_id"] + "_%d" % self.wavelength[idx]

	# 	# optical parameter

	# 	# 
	# 	mcx_input["Domain"]["Media"][0]["mua"] = 0
	# 	mcx_input["Domain"]["Media"][0]["mus"] = 0
	# 	mcx_input["Domain"]["Media"][0]["g"] = 1
	# 	mcx_input["Domain"]["Media"][0]["n"] = 1

	# 	# skin
	# 	mcx_input["Domain"]["Media"][1]["name"] = "skin"
	# 	mcx_input["Domain"]["Media"][1]["mua"] = self._calculate_mua(
	# 		idx, 
	# 		self.parameters["skin"]["blood_volume_fraction"], 
	# 		self.parameters["skin"]["ScvO2"], 
	# 		self.parameters["skin"]["water_volume"]
	# 		)
	# 	mcx_input["Domain"]["Media"][1]["mus"] = self._calculate_mus(
	# 		idx,
	# 		self.parameters["skin"]["muspx"], 
	# 		self.parameters["skin"]["fray"], 
	# 		self.parameters["skin"]["bmie"],
	# 		self.parameters["skin"]["g"]
	# 		)
	# 	mcx_input["Domain"]["Media"][1]["g"] = self.parameters["skin"]["g"]
	# 	mcx_input["Domain"]["Media"][1]["n"] = self.parameters["skin"]["n"]

	# 	# muscle
	# 	mcx_input["Domain"]["Media"][2]["name"] = "muscle"
	# 	mcx_input["Domain"]["Media"][2]["mua"] = self._calculate_muscle_mua(
	# 		idx, 
	# 		self.parameters["muscle"]["water_volume"]
	# 		)
	# 	mcx_input["Domain"]["Media"][2]["mus"] = self._calculate_mus(
	# 		idx,
	# 		self.parameters["muscle"]["muspx"], 
	# 		self.parameters["muscle"]["fray"], 
	# 		self.parameters["muscle"]["bmie"],
	# 		self.parameters["muscle"]["g"]
	# 		)
	# 	mcx_input["Domain"]["Media"][2]["g"] = self.parameters["muscle"]["g"]
	# 	mcx_input["Domain"]["Media"][2]["n"] = self.parameters["muscle"]["n"]
		
	# 	# IJV
	# 	mcx_input["Domain"]["Media"][3]["name"] = "IJV"
	# 	mcx_input["Domain"]["Media"][3]["mua"] = self._calculate_mua(
	# 		idx, 
	# 		self.parameters["IJV"]["blood_volume_fraction"], 
	# 		self.parameters["IJV"]["ScvO2"], 
	# 		self.parameters["IJV"]["water_volume"],

	# 		)
	# 	mcx_input["Domain"]["Media"][3]["mus"] = self._calculate_mus(
	# 		idx,
	# 		self.parameters["IJV"]["muspx"], 
	# 		self.parameters["IJV"]["fray"], 
	# 		self.parameters["IJV"]["bmie"],
	# 		self.parameters["IJV"]["g"]
	# 		)
	# 	mcx_input["Domain"]["Media"][3]["g"] = self.parameters["IJV"]["g"]
	# 	mcx_input["Domain"]["Media"][3]["n"] = self.parameters["IJV"]["n"]
		
	# 	# CCA
	# 	mcx_input["Domain"]["Media"][4]["name"] = "CCA"
	# 	mcx_input["Domain"]["Media"][4]["mua"] = self._calculate_mua(
	# 		idx,
	# 		self.parameters["CCA"]["blood_volume_fraction"], 
	# 		self.parameters["CCA"]["ScvO2"], 
	# 		self.parameters["CCA"]["water_volume"]
	# 		)
	# 	mcx_input["Domain"]["Media"][4]["mus"] = self._calculate_mus(
	# 		idx,
	# 		self.parameters["CCA"]["muspx"], 
	# 		self.parameters["CCA"]["fray"], 
	# 		self.parameters["CCA"]["bmie"],
	# 		self.parameters["CCA"]["g"]
	# 		)
	# 	mcx_input["Domain"]["Media"][4]["g"] = self.parameters["CCA"]["g"]
	# 	mcx_input["Domain"]["Media"][4]["n"] = self.parameters["CCA"]["n"]


	# 	# geometry
		


	# 	# skin
	# 	mcx_input["Shapes"][1]["Subgrid"]["O"] = [1, 1, 1]
	# 	mcx_input["Shapes"][1]["Subgrid"]["Size"] = [100, 100, self.parameters["geometry"]["skin_thickness"]]

	# 	# muscle
	# 	mcx_input["Shapes"][2]["Subgrid"]["O"] = [1, 1, 1+self.parameters["geometry"]["skin_thickness"]]
	# 	mcx_input["Shapes"][2]["Subgrid"]["Size"] = [100, 100, 300-self.parameters["geometry"]["skin_thickness"]]

	# 	# ijv 
	# 	mcx_input["Shapes"][3]["Cylinder"]["C0"] = [100, 50, self.parameters["geometry"]["ijv_depth"]]
	# 	mcx_input["Shapes"][3]["Cylinder"]["C1"] = [0, 50, self.parameters["geometry"]["ijv_depth"]]
	# 	mcx_input["Shapes"][3]["Cylinder"]["R"] = self.parameters["geometry"]["ijv_radius"]

	# 	# cca 
	# 	mcx_input["Shapes"][4]["Cylinder"]["C0"] = [100, 50-self.parameters["geometry"]["ijv_cca_distance"], self.parameters["geometry"]["cca_depth"]]
	# 	mcx_input["Shapes"][4]["Cylinder"]["C1"] = [0, 50-self.parameters["geometry"]["ijv_cca_distance"], self.parameters["geometry"]["cca_depth"]]
	# 	mcx_input["Shapes"][4]["Cylinder"]["R"] = self.parameters["geometry"]["cca_radius"]

	# 	# set seed
	# 	mcx_input["Session"]["RNGSeed"] = randint(0, 1000000000)

	# 	# save the .json file in the output folder
	# 	with open(self.config["geometry_file"], 'w+') as f:
	# 		json.dump(mcx_input, f, indent=4)	

	def _calculate_mua(self, idx, b, s, w, f, m):
		mua = b * (s * self.oxy[idx] + (1-s) * self.deoxy[idx]) + w * self.water[idx] + f * self.fat[idx] + m * self.melanin[idx]
		# print("==============================")
		# print("wl: ", self.wavelength[idx])
		# print("oxy: ", self.oxy[idx])
		# print("deoxy: ", self.deoxy[idx])
		# print("water: ", self.water[idx])
		# print("mua: ", mua)
		return mua

	def _calculate_muscle_mua(self, idx, w):
		mua = w * self.water[idx] + (1-w) * self.collagen[idx]
		return mua

	def _calculate_mus(self, idx, mus500, fray, bmie, g):
		wl = self.wavelength[idx]
		mus_p = mus500 * (fray * (wl/500)**(-4) + (1-fray) * (wl/500) ** (-bmie))
		mus = mus_p/g * 0.1
		return mus 

	def _get_command(self, wl, sds):
		# dummy function create the command for mcx
		session_name = "\"%s_%d_%d\" " % (self.config["session_id"], wl, sds)
		geometry_file = "\"%s\" " % os.path.abspath(self.config["geometry_file"])
		root = "\"%s\" " % os.path.join(os.path.abspath(self.session), "mcx_output")
		unitmm = "%f " % self.config["voxel_size"]
		photon = "%d " % self.config["photon_batch"]
		num_batch = "%d " % (self.config["num_photon"]//self.config["photon_batch"])
		maxdetphoton = "10000000"
		# maxdetphoton = "%d" % (self.config["num_photon"]//5)
		save_mc2 = "0 " if self.config["train"] else "1 "

		command = \
		"./mcx --session " + session_name +\
		"--input " + geometry_file +\
		"--root " + root +\
		"--gpu 1 " +\
		"--autopilot 1 " +\
		"--photon " + photon +\
		"--repeat " + num_batch +\
		"--normalize 1 " +\
		"--save2pt " + save_mc2 +\
		"--reflect 1 " +\
		"--savedet 1 " +\
		"--saveexit 1 " +\
		"--unitinmm " + unitmm +\
		"--saveseed 0 " +\
		"--skipradius -2 " +\
		"--array 0 " +\
		"--dumpmask 0 " +\
		"--maxdetphoton " + maxdetphoton
		print(command)
		return command

	

def test_medium():
	mcx = MCX()
	command = mcx._get_command(400)
	os.chdir("mcx/bin")
	os.system(command)


def test_pickle():
	mcx = MCX()
	path = "output/test/result/" 
	results = mcx.reload_result(path+"result.pkl")
	
	for r_idx, r in enumerate(results):
		fig = plt.figure()
		for d_idx, d in enumerate(r):
			plt.plot(mcx.wavelength, np.log10(d), label="detector %d" % d_idx)
		
		_path = os.path.join(path, "Scv_%d.png" % r_idx)
		plt.title('Scv_%d' % r_idx)
		plt.legend()
		plt.savefig(_path)
		plt.close()


if __name__ == "__main__":
	# test_medium()
	test_pickle()
