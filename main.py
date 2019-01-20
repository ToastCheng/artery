import os
import json 
from glob import glob
from argparse import ArgumentParser

from utils.line import LineBot
from mcx import MCX
from generator import Generator


line = LineBot()


def get_args():
	parser = ArgumentParser()
	parser.add_argument("-c", "--config", type=str, default="config.json")
	parser.add_argument("-f", "--forward", action="store_true")
	parser.add_argument("-i", "--inverse", action="store_true")
	parser.add_argument("-t", "--train", action="store_true")

	parser.add_argument("-g", "--generate", type=int, default=0)



	return parser.parse_args()


def main():
	args = get_args()
	if args.forward:

		config_file = os.path.join('configs', args.config)
		print(config_file)
		forward(config_file)
	
	if args.inverse:

		inverse()

	if args.train and args.forward:

		generate_new_input = args.generate
		train(generate_new_input)


def forward(config):

	# run mcx!
	mcx = MCX(config)
	mcx.run()
	mcx.calculate_reflectance()


def inverse():
	pass


def train(generate_new_input):

	train_list = glob(os.path.join('generator', 'parameter', '*'))
	train_list.sort()

	config_train = "config_train.json"

	if os.path.isfile("train_log.txt"):
		with open('train_log.txt', 'r') as f:
			check_point = int(f.read())
	else:
		check_point = 0

	gen = Generator()

	if generate_new_input != 0:
		for idx in range(generate_new_input):
			gen.run(idx=idx)

	for idx, parameter in enumerate(train_list[check_point:]):
		with open('config_train.json') as f:
			config = json.load(f)
			config["session_id"] = "train_%d" % idx
			config["input_file"] = parameter
		with open('config_train.json', 'w') as f:
			json.dump(config, f, indent=4)

		line.print('training %d/%d' % (idx, len(train_list)))
		with open('train_log.txt', 'w+') as f:
			f.write("%d" % idx)

		mcx = MCX("config_train.json")
		mcx.run()
		mcx.calculate_reflectance(plot=True)
		

	# from pprint import PrettyPrinter
	# pp = PrettyPrinter()
	# pp.pprint(train_list)




if __name__ == "__main__":
	l = LineBot()
	main()
	l.print("finish!")
	# os.system("sudo shutdown")
