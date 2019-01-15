
import numpy as np 
from torch.utils.data import Dataset



class SimData(Dataset):

	def __init__(self):
		self.scv = np.load("training_data/scv.pkl")
		self.spectrum = np.load("training_data/spectrum.pkl")
		self.geometry = np.load("training_data/geometry.pkl")
		self.parameter = np.load("training_data/parameter.pkl")


		assert len(self.scv) == len(self.spectrum), "size of scv and spectrum are not matched!"
		assert len(self.spectrum) == len(self.geometry), "size of geometry and spectrum are not matched!"
		assert len(self.scv) == len(self.parameter), "size of scv and parameter are not matched!"

	def __getitem__(self, idx):
		return (self.spectrum[idx], self.geometry[idx], self.parameter[idx]), self.scv[idx] 

	def __len__(self):
		return len(self.scv)

