
import os 
from argparse import ArgumentParser

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from inverse.model import SCVNet
from inverse.dataset import SimData




def get_args():
	parser = ArgumentParser()

	parser.add_argument("-l", "--learning_rate", type=float, default=1e-4)
	parser.add_argument("-e", "--epoch", type=int, default=1000)
	parser.add_argument("-p", "--pretrain", action="store_true")
	parser.add_argument("-m", "--model", type=str, default="")
	parser.add_argument("-b", "--batch_size", type=int, default=32)

	parser.add_argument("--log", type=str, default="log/log2.csv")

	return parser.parse_args()


def get_log(args):

	if os.path.isfile(args.log):
		log = pd.read_csv(args.log)
		return log
	else:
		log = pd.DataFrame({
			"train_error": [],
			"valid_error": []
		})
		return log



def train():
	args = get_args()
	log = get_log(args)

	net = SCVNet()

	if args.pretrain:
		net.load_state_dict(torch.load("saved_model/model.pth"))

	dataset = SimData()

	# train test split
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(0.1 * dataset_size))

	# shuffle
	# np.random.seed(random_seed)
	np.random.shuffle(indices)
	
	train_indices, val_indices = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	# if sampler is used, do not set "shuffle"
	# drop last since there might be only 1 data in the last batch
	train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
	valid_loader = DataLoader(dataset, batch_size=split, sampler=valid_sampler)



	loss_func = nn.MSELoss()
	optimizer = optim.Adam(params=net.parameters(), lr=args.learning_rate)

	train_losses = []
	valid_losses = []

	for e in range(args.epoch):
		for idx, data in enumerate(train_loader):

			spectrum, geometry, parameter = data[0]
			scv = data[1]

			# default is double, but float is required!
			spectrum = spectrum.float()
			geometry = geometry.float()
			parameter = parameter.float()
			scv = scv.float()


			net.zero_grad()

			pred = net(spectrum, geometry, parameter)
			
			
			loss = loss_func(pred, scv)

			loss.backward()
			optimizer.step()

		for data in valid_loader:
			s, g, p = data[0]
			_scv = data[1]

			s = s.float()
			g = g.float()
			p = p.float()
			_scv = _scv.float()

			_pred = net(s, g, p)

			_loss = loss_func(_pred, _scv)
		
		log.loc[len(log)] = {"train_error": float(loss), "valid_error": float(_loss)}
		log.to_csv(args.log, index=False)

		train_losses.append(loss)
		valid_losses.append(_loss)

		if e%5 == 0:
			print("================="*2)
			print("epoch: %d | train loss: %.4f | valid loss: %.4f" %(e, loss, _loss))
			print("predict: ", pred[:5])
			print("target: ", scv[:5])


	plt.plot(train_losses, label="train")
	plt.plot(valid_losses, label="valid")
	plt.xlabel('epoch')
	plt.ylabel('rms error')
	plt.legend()
	plt.savefig('log.png')

	torch.save(net.state_dict(), "saved_model/model.pth")



if __name__ == "__main__":
	train()



