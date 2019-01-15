import os
import json 
from glob import glob

import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 

import utils
from utils.line import LineBot
from mcx import MCX
from generator import Generator


line = LineBot()


def main():
	#test_whiteMC()
	#line.print("test white done")
	test_smooth()
	line.print("test smooth done")

def test_whiteMC():
	"""
	This function is writen to test whether white Monte Carlo is 
	adequate for the IJV tissue model
	"""

	# white
	mcx = MCX("test/test_white.json")
	mcx.run()
	mcx.calculate_reflectance(plot=True)

	reflec_white = mcx.reflectance[70]

	# normal
	mcx2 = MCX("test/test_normal.json")
	mcx2.run(white=False)
	mcx2.calculate_reflectance(white=False, plot=False)

	reflec = mcx2.reflectance


	for idx, (r_w, r) in enumerate(zip(reflec_white, reflec)):
		plt.plot(r, label='normal')
		plt.plot(r_w, label='white')
		plt.title('on sds %d' % idx)
		plt.legend()
		plt.savefig("test/whiteMC_sds_%d.png" % idx)
		plt.close()

	print(rmse(reflec_white, reflec))

	df = pd.DataFrame({"white": reflec_white[-1], "normal": reflec[-1]})
	df.to_csv("test/result_whiteMC.csv", index=False)


def test_smooth():
	"""
	This function is writen to test whether we can use smooth technique
	to lower the number of photon and maintain the spectrum shape,
	in that, we can speed up the simulation
	"""
	mcx = MCX('test/test_smooth_6e8.json')
	mcx.run()
	mcx.calculate_reflectance()

	reflec_2e8 = mcx.reflectance[70][-1]

	mcx2 = MCX('test/test_smooth_2e9.json')
	mcx2.run()
	mcx2.calculate_reflectance()

	reflec_2e9 = mcx2.reflectance[70][-1]

	df = pd.DataFrame({"2e8": reflec_2e8, "2e9": reflec_2e9})
	df.to_csv("test/result_smooth.csv", index=False)



def rmse(x, y):
	e = np.sqrt(((x - y)**2).mean())
	return e/y.mean()



if __name__ == "__main__":
	main()
	os.system("sudo shutdown")
	# df = pd.read_csv("test/result_whiteMC.csv")
	# print (rmse(df["normal"], df["white"]))
