import numpy as np 
import pandas as pd 


class Calibrator:
	def __init__(self, num_phantom, num_sds):
		self.num_phantom = num_phantom
		self.num_sds = num_sds
		self.a = []
		self.b = []

	def fit(self, measured, simulated):
		"""
		[input]
		measured: 
			2D array of measured spectrum
			shape: [num_sds, num_wavelength]
		simulated:
			2D array of simulated spectrum
			shape: [num_sds, num_wavelength]
		[output]
		a, b:
			in each wavelength
			simulated = a * measured + b

		"""
		for m, s in zip(measured.T, simulated.T):
			aa, bb = np.polyfit(m, s, 1)
			self.a.append(aa)
			self.b.append(bb)

		return self.a, self.b

	def read_tiff(self, path):
		from PIL import Image		
		img = Image.open(path)
		# !! need to be process to spectrum form !!
		# havent implement
		return np.array(img)

	def read_txt(self, path):
		# special for IJV
		df = pd.read_csv(path, sep=' ', header=None)

		return df.values


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	# set example data
	x = np.array([
			[1,4,6,7,1,3,6,3,2,1,9,5,8],
			[2,4,3,6,3,9,9,3,5,6,3,2,7],
			[4,7,1,9,5,8,4,9,5,3,2,6,7],
			[1,2,3,4,5,6,7,6,5,4,5,6,7]])
	x = x * 5
	a = [2,3,2,1,1,2,1,1,1,2,3,2,1]
	b = [1,1,0,0,0,1,1,2,0,1,1,1,1]
	
	y = x * a + b

	# function
	fit_a = []
	fit_b = []

	for xx, yy in zip(x.T, y.T):
		f_a, f_b = np.polyfit(xx, yy, 1)
		fit_a.append(np.round(f_a))
		fit_b.append(np.round(f_b))
	
	print(fit_a)
	print(fit_b)

	# for xx, yy in zip(x, y):
	# 	plt.plot(xx)
	# 	plt.plot(yy)
	# 	plt.show()