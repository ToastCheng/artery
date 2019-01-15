import json
from time import time
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_mch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MCHHandler:
	# This class is a .mch file handler.

	def __init__(
		self,
		mch_file=None, 
		config="config.json"
		):

		with open(config) as f:
			self.config = json.loads(f.read())
		if mch_file is not None:
			self.df, self.header, self.photon = self._load_mch(mch_file)
		else:
			self.df = None
			self.header = None
			self.photon = None
		self.mua = pd.read_csv(self.config["absorb_file"])
		self.wavelength = pd.read_csv(self.config["wavelength_file"])
		with open(self.config["input_file"]) as f: 
			self.input = json.loads(f.read())

		self.detector_na = self.config["detector_na"]
		self.detector_n = self.config["medium_n"]
		self.critical_angle = np.arcsin(self.detector_na/self.detector_n)
		# self.critical_angle = 12


	def compute_reflectance_white(self, wl, mch_file=None):
		# reload a .mch file
		if mch_file is not None:
			self.df, self.header, self.photon = self._load_mch(mch_file)

		df = self._parse_mch()
		# filter the photon that not been detected(due to critical angle)
		df = df[np.arccos(df.angle.abs()) <= self.critical_angle]
		if len(df) == 0:
			print('no photon detected at %d' % wl)
			return None, None
		df = df.reset_index(drop=True)


		# [start test speed]
		if False:
			num_photon = int(1e6)
			idx = np.random.randint(4, size=(num_photon, 1)) + 1
			path = np.random.rand(num_photon, 3)
			angle = np.random.rand(num_photon, 1)
			data = np.concatenate([idx, path, angle], 1)

			df = pd.DataFrame(data, columns=['detector_idx', 'media_0', 'media_1', 'media_2', 'angle'])
		# [end test speed]

		# [photon, medium]
		path_length = df.iloc[:, 1:-1].values

		# calculate percentage
		skin_portion = path_length.sum(0)[0]/path_length.sum()
		fat_portion = path_length.sum(0)[1]/path_length.sum()
		muscle_portion = path_length.sum(0)[2]/path_length.sum()
		artery_portion = path_length.sum(0)[3]/path_length.sum()

		path_length = torch.tensor(path_length).float().to(device)

		# [medium, ScvO2]
		mua = self._make_tissue_white(wl)
		mua = torch.tensor(mua).float().to(device)

		# [photon, ScvO2]
		weight = torch.exp(-torch.matmul(path_length, mua) *self.header["unitmm"]) 
		
		# [1, ScvO2]
		result = torch.zeros(1, weight.shape[1]).float().to(device)

		# seperate photon with different detector
		for idx in range(1, self.header["detnum"]+1):

			# get the index of specific index
			detector_list = df.index[df["detector_idx"] == idx].tolist()
			if len(detector_list) == 0:
				# print("detector #%d detected no photon" % idx)
				result = torch.cat((result, torch.zeros(1, weight.shape[1]).float().to(device)), 0)
				continue
			# pick the photon that detected by specific detector
			
			# [photon, ScvO2]
			_weight = weight[detector_list]
			# print(_weight.shape)
			_weight = _weight.sum(0)
			_weight = _weight.unsqueeze(0)
			# print(_weight.shape)
			# print(result.shape)
			result = torch.cat((result, _weight), 0)

		# [SDS, ScvO2]
		result = result[1:]

		return result.cpu().numpy()/self.header["total_photon"], (skin_portion, fat_portion, muscle_portion, artery_portion)
		
	def _make_tissue_white(self, wl):

		# the ScvO2
		ScvO2 = np.arange(0, 1.01, 0.01)
		
		# mua
		oxy = self.mua['oxy'].values
		deoxy = self.mua['deoxy'].values
		water = self.mua['water'].values
		collagen = self.mua['collagen'].values
		fat = self.mua['fat'].values
		melanin = self.mua['mel'].values
		wavelength = self.mua['wavelength'].values

		# interpolation
		oxy = np.interp(wl, wavelength, oxy)
		deoxy = np.interp(wl, wavelength, deoxy)
		water = np.interp(wl, wavelength, water)
		collagen = np.interp(wl, wavelength, collagen)
		fat = np.interp(wl, wavelength, fat)
		melanin = np.interp(wl, wavelength, melanin)

		# turn the unit 1/cm --> 1/mm
		oxy *= 0.1
		deoxy *= 0.1
		water *= 0.1
		collagen *= 0.1
		fat *= 0.1
		melanin *= 0.1

		# [medium, 1]
		mua = np.zeros((self.header["maxmedia"], 1))


		for s in ScvO2:

			skin = self._calculate_mua(
				self.input["skin"]["blood_volume_fraction"],
				self.input["skin"]["ScvO2"],
				self.input["skin"]["water_volume"],
				self.input["skin"]["fat_volume"],
				self.input["skin"]["melanin_volume"],
				oxy, 
				deoxy, 
				water,
				fat,
				melanin
				)
			fat = self._calculate_mua(
				self.input["fat"]["blood_volume_fraction"],
				self.input["fat"]["ScvO2"],
				self.input["fat"]["water_volume"],
				self.input["fat"]["fat_volume"],
				self.input["fat"]["melanin_volume"],
				oxy, 
				deoxy, 
				water,
				fat,
				melanin
				)

			muscle = self._calculate_muscle_mua(
				self.input["muscle"]["water_volume"],
				water,
				collagen
				)

			artery = self._calculate_mua(
				self.input["artery"]["blood_volume_fraction"],
				s,	# set ScvO2
				self.input["artery"]["water_volume"],
				self.input["artery"]["fat_volume"],
				self.input["artery"]["melanin_volume"],
				oxy, 
				deoxy, 
				water,
				fat,
				melanin
				)

			_mua = np.concatenate(
				[np.expand_dims(skin, 0),
				 np.expand_dims(fat, 0), 
				 np.expand_dims(muscle, 0), 
				 np.expand_dims(artery, 0)], 0
				 )
			# print(_mua.shape)
			# print(mua.shape)
			mua = np.concatenate(
				[mua, np.expand_dims(_mua, 1)], 1
				)

		# [medium, ScvO2]
		mua = mua[:, 1:]
		return mua

	# def compute_reflectance(self, wl, mch_file=None):
	# 	# reload a .mch file
	# 	if mch_file is not None:
	# 		self.df, self.header, self.photon = self._load_mch(mch_file)

	# 	df = self._parse_mch()
	# 	# filter the photon that not been detected(due to critical angle)
	# 	df = df[np.arccos(df.angle.abs()) <= self.critical_angle]
	# 	if len(df) == 0:
	# 		print('no photon detected at %d' % wl)
	# 		return None, None
	# 	df = df.reset_index(drop=True)


	# 	# [photon, medium]
	# 	path_length = df.iloc[:, 1:-1].values

	# 	# calculate percentage
	# 	skin_portion = path_length.sum(0)[0]/path_length.sum()
	# 	muscle_portion = path_length.sum(0)[1]/path_length.sum()
	# 	ijv_portion = path_length.sum(0)[2]/path_length.sum()
	# 	cca_portion = path_length.sum(0)[3]/path_length.sum()

	# 	path_length = torch.tensor(path_length).float().to(device)

	# 	# [medium]
	# 	mua = self._make_tissue(wl)
	# 	mua = torch.tensor(mua).float().to(device)

	# 	# [photon]
	# 	weight = torch.exp(-torch.matmul(path_length, mua) *self.header["unitmm"]) 
		
	# 	# [1]
	# 	result = torch.zeros(1).float().to(device)

	# 	# seperate photon with different detector
	# 	for idx in range(1, self.header["detnum"]+1):

	# 		# get the index of specific index
	# 		detector_list = df.index[df["detector_idx"] == idx].tolist()
	# 		if len(detector_list) == 0:
	# 			# print("detector #%d detected no photon" % idx)
	# 			result = torch.cat((result, torch.zeros(1).float().to(device)), 0)
	# 			continue
	# 		# pick the photon that detected by specific detector
			
	# 		# [photon]
	# 		_weight = weight[detector_list]
	# 		# print(_weight.shape)
	# 		_weight = _weight.sum(0)
	# 		_weight = _weight.unsqueeze(0)
	# 		# print(_weight.shape)
	# 		# print(result.shape)
	# 		result = torch.cat((result, _weight), 0)

	# 	# [SDS, ScvO2]
	# 	result = result[1:]

	# 	return result.cpu().numpy()/self.header["total_photon"], (skin_portion, muscle_portion, ijv_portion, cca_portion)
		
	# def _make_tissue(self, wl):

	# 	# the ScvO2
	# 	ScvO2 = np.arange(0, 1.01, 0.01)
		
	# 	# mua
	# 	oxy = self.mua['oxy'].values
	# 	deoxy = self.mua['deoxy'].values
	# 	water = self.mua['water'].values
	# 	collagen = self.mua['collagen'].values
	# 	wavelength = self.mua['wavelength'].values

	# 	# interpolation
	# 	oxy = np.interp(wl, wavelength, oxy)
	# 	deoxy = np.interp(wl, wavelength, deoxy)
	# 	water = np.interp(wl, wavelength, water)
	# 	collagen = np.interp(wl, wavelength, collagen)

	# 	# turn the unit 1/cm --> 1/mm
	# 	oxy *= 0.1
	# 	deoxy *= 0.1
	# 	water *= 0.1
	# 	collagen *= 0.1

	# 	# [medium, 1]
	# 	mua = np.zeros((self.header["maxmedia"], 1))




	# 	skin = self._calculate_mua(
	# 		self.input["skin"]["blood_volume_fraction"],
	# 		self.input["skin"]["ScvO2"],
	# 		self.input["skin"]["water_volume"],
	# 		oxy, 
	# 		deoxy, 
	# 		water
	# 		)

	# 	muscle = self._calculate_muscle_mua(
	# 		self.input["muscle"]["water_volume"],
	# 		water,
	# 		collagen
	# 		)

	# 	IJV = self._calculate_mua(
	# 		self.input["IJV"]["blood_volume_fraction"],
	# 		self.input["IJV"]["ScvO2"],
	# 		self.input["IJV"]["water_volume"],
	# 		oxy, 
	# 		deoxy, 
	# 		water
	# 		)
	# 	CCA = self._calculate_mua(
	# 		self.input["CCA"]["blood_volume_fraction"],
	# 		self.input["CCA"]["ScvO2"],
	# 		self.input["CCA"]["water_volume"],
	# 		oxy, 
	# 		deoxy, 
	# 		water
	# 		)

	# 	mua = np.asarray([skin, muscle, IJV, CCA])

	# 	# [medium]
	# 	return mua

	def _convert_unit(self, length_mm):
		# convert mm to number of grid
		num_grid = length_mm//self.header["unitmm"]
		return num_grid

	def _parse_mch(self):
		
		# selected the detector_idx, pathlength, angle of the
		# photon when it is leaving the tissue

		num_media = self.header["maxmedia"]	
		# selected_list = [0] + [i for i in range(num_media+1, 2*num_media+1)] + [-1]
		selected_list = [0] + [i for i in range(2, 2 + num_media)] + [-1]
		df = self.df[:, selected_list]
		label = ["detector_idx"]
		label += ["media_{}".format(i) for i in range(self.header["maxmedia"])]
		label += ["angle"]
		df = pd.DataFrame(df, columns=label)
		return df

	@staticmethod
	def _calculate_mua(b, s, w, f, m, oxy, deoxy, water, fat, melanin):
		mua = b * (s * oxy + (1-s) * deoxy) + w * water + f * fat + m * melanin
		return mua

	@staticmethod
	def _calculate_muscle_mua(w, water, collagen):
		mua = w * water + (1-w) * collagen
		return mua 

	@staticmethod
	def _load_mch(path):
		data = load_mch(path)
		# check if the mcx saved the photon seed
		if data[1]["seed_byte"] == 0:
			df, header = data
			photon_seed = None
		elif data[1]["seed_byte"] == 1:
			df, header, photon_seed = data

		return df, header, photon_seed

	def test(self):
		# JUST FOR DEVELOPMENT
		wl = 900

		# mua
		oxy = self.mua['oxy'].values
		deoxy = self.mua['deoxy'].values
		water = self.mua['water'].values
		wavelength = self.mua['wavelength'].values

		# interpolation
		oxy = np.interp(wl, wavelength, oxy)
		deoxy = np.interp(wl, wavelength, deoxy)
		water = np.interp(wl, wavelength, water)

		# turn the unit 1/cm --> 1/mm
		oxy *= 0.1
		deoxy *= 0.1
		water *= 0.1

		IJV = []
		for s in np.arange(0, 1, 0.01):
			ijv = self._calculate_mua(
					self.input["IJV"]["blood_volume_fraction"],
					s,	# set ScvO2
					self.input["IJV"]["water_volume"],
					oxy, 
					deoxy, 
					water
					)
			IJV.append(ijv)
		plt.plot(IJV)
		plt.show()

def test_compute_reflectance_white():
	
	df = pd.DataFrame(
		{"detector_idx":[0,0,1,1,0,1,2,3,2,3,1,1,0,2,3,3],
		 "media_0":     [1,2,3,4,5,6,7,8,9,0,9,8,7,6,5,4],
		 "media_1":     [1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2],
		 "angle":       [0.1,0.9,0.1,0.9,0.9,0.9,0.1,0,0.9,0,1,1,1,0,0,1]
		 }
		)
	# filter the photon that not been detected(due to critical angle)
	df = df[np.arccos(df.angle.abs()) <= 0.8]
	print(df)
	# reset the index! (cause some of them are filtered)
	df = df.reset_index(drop=True)
	print(df)
	print(df.shape)
	
	# [medium, wavelength, ScvO2]
	# -> [ScvO2, wavelength, medium]
	mua = torch.rand(100, 20, 2)
	mua = torch.tensor(mua)
	print(mua.shape)

	# [photon, medium]
	# -> [medium, photon]
	path_length = df.iloc[:, 1:-1].values
	path_length = torch.tensor(path_length).float().transpose(1, 0)
	print(path_length)
	print(path_length.shape)


	# [photon, wavelength, ScvO2]
	# -> [ScvO2, wavelength, photon]
	weight = torch.exp(-torch.matmul(mua, path_length))
	print(weight.shape)
	
	# [1, wavelength, ScvO2]
	# -> [ScvO2, wavelength, 1]
	result = torch.zeros(weight.shape[:-1]).unsqueeze(2)
	print(result.shape)

	# seperate photon with different detector
	for idx in range(4):
		# get the index of specific index
		detector_list = df.index[df["detector_idx"] == idx].tolist()

		# pick the photon that detected by specific detector
		# [1, wavelength, ScvO2]
		# -> [ScvO2, wavelength, 1]
		_weight = weight[:,:, detector_list].sum(2).unsqueeze(2)
		result = torch.cat((result, _weight), 2)
		print(result.shape)

	# [SDS, wavelength, ScvO2]
	# -> [ScvO2, wavelength, SDS]
	result = result[:, :, 1:]
	print(result.shape)

	return result


if __name__ == "__main__":
	m = MCHHandler("stuff/test.mch")

	# s = time()
	# result = m.compute_reflectance_white(test=True)
	# e = time() - s
	# print("%2d:%2d" % (e//60, e%60))
	# # print(result)
	# print(result.shape)
	# # t = test_compute_reflectance_white()

	# m.test()
