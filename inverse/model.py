import torch 
import torch.nn as nn


class SCVNet(nn.Module):
	def __init__(self, spectrum_size=36, geometry_size=6, parameter_size=23):
		super(SCVNet, self).__init__()

		self.encoder_geometry = nn.Sequential(
			nn.Linear(geometry_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.encoder_spectrum = nn.Sequential(
			nn.Linear(spectrum_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.encoder_parameter = nn.Sequential(
			nn.Linear(parameter_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.fc = nn.Sequential(
			nn.Linear(150, 500),
			nn.BatchNorm1d(500),
			nn.ReLU(),
			nn.Linear(500, 500),
			nn.BatchNorm1d(500),
			nn.ReLU(), 
			nn.Linear(500, 1)
		)



	def forward(self, spectrum, geometry, parameter):
		# spectrum: [batch, spectrum_size]
		# geometry: [batch, geometry_size]

		# encode the geometry

		# [batch, 50]
		geometry = self.encoder_geometry(geometry)

		# [batch, 50]
		parameter = self.encoder_parameter(parameter)

		# spectrum flatten
		spectrum = spectrum.view(spectrum.shape[0], -1)

		# [batch, 50]
		spectrum = self.encoder_spectrum(spectrum)

		# [batch, 100]
		processed_input = torch.cat([spectrum, geometry, parameter], 1)

		out = self.fc(processed_input)

		return out



class SCVNetV1(nn.Module):
	def __init__(self, spectrum_size=36, geometry_size=6, parameter_size=23):
		super(SCVNet, self).__init__()

		self.encoder_geometry = nn.Sequential(
			nn.Linear(geometry_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.encoder_spectrum = nn.Sequential(
			nn.Linear(spectrum_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.encoder_parameter = nn.Sequential(
			nn.Linear(parameter_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.fc = nn.Sequential(
			nn.Linear(150, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(), 
			nn.Linear(100, 1)
		)



	def forward(self, spectrum, geometry, parameter):
		# spectrum: [batch, spectrum_size]
		# geometry: [batch, geometry_size]

		# encode the geometry

		# [batch, 50]
		geometry = self.encoder_geometry(geometry)

		# [batch, 50]
		parameter = self.encoder_parameter(parameter)

		# spectrum flatten
		spectrum = spectrum.view(spectrum.shape[0], -1)

		# [batch, 50]
		spectrum = self.encoder_spectrum(spectrum)

		# [batch, 100]
		processed_input = torch.cat([spectrum, geometry, parameter], 1)

		out = self.fc(processed_input)

		return out


class Encoder(nn.Module):
	def __init__(self, input_size=36, output_size=8):
		super(Encoder, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(input_size, output_size),
			nn.BatchNorm1d(output_size),
			nn.ReLU()
		)


	def forward(self, x):
		x = self.fc(x)
		return x 


class Decoder(nn.Module):
	def __init__(self, input_size=8, output_size=36):
		super(Decoder, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(input_size, output_size),
			nn.BatchNorm1d(output_size),
			nn.ReLU()
		)
		
	def forward(self, x):
		x = self.fc(x)
		return x 

if __name__ == "__main__":
	import torch.optim as optim
	import matplotlib.pyplot as plt 

	x = torch.tensor([
		[1.90628418e-07, 2.23723511e-07, 3.04706360e-07, 3.68610474e-07,
       4.40840485e-07, 4.82916199e-07, 5.55245972e-07, 5.19816895e-07,
       5.70398560e-07, 5.34469421e-07, 3.93041016e-07, 3.35176270e-07,
       3.67108521e-07, 3.81839417e-07, 3.62761139e-07, 3.80533752e-07,
       3.59872925e-07, 3.17645996e-07, 2.94018585e-07, 2.73521484e-07,
       2.45239624e-07, 2.40928513e-07, 2.42998779e-07, 2.22284027e-07,
       2.04363037e-07, 1.93216400e-07, 1.92067444e-07, 1.74558319e-07,
       1.63647888e-07, 1.55452988e-07, 1.29467667e-07, 1.43866302e-07,
       1.41914429e-07, 1.50341949e-07, 1.66262604e-07, 2.00582062e-07],
       [2.38439606e-07, 2.75679993e-07, 3.63077087e-07, 4.35120789e-07,
       5.07633209e-07, 5.45940979e-07, 6.14647339e-07, 5.64942749e-07,
       6.06417603e-07, 5.66201416e-07, 4.22743683e-07, 3.61968201e-07,
       3.87382080e-07, 3.93021057e-07, 3.66837891e-07, 3.78828491e-07,
       3.55677490e-07, 3.11917603e-07, 2.87385254e-07, 2.66320557e-07,
       2.38193039e-07, 2.33601593e-07, 2.35226166e-07, 2.15278976e-07,
       1.97750427e-07, 1.86946091e-07, 1.85695450e-07, 1.68869186e-07,
       1.58123428e-07, 1.49764755e-07, 1.23852707e-07, 1.37133209e-07,
       1.34130768e-07, 1.41510406e-07, 1.56228851e-07, 1.87888428e-07]
	])

	x[0] = (x[0] - x.mean(1)[0])/x.std(1)[0]
	x[1] = (x[1] - x.mean(1)[1])/x.std(1)[1]

	en = Encoder()
	de = Decoder()

	loss_func = nn.MSELoss()
	en_optimizer = optim.Adam(en.parameters(), lr=1e-5)
	de_optimizer = optim.Adam(de.parameters(), lr=1e-5)
	for epoch in range(1000):
		en_optimizer.zero_grad()
		de_optimizer.zero_grad()

		en_out = en(x)
		de_out = de(en_out)

		loss = loss_func(de_out, x)
		print(loss)
		loss.backward()
		en_optimizer.step()
		de_optimizer.step()

		if epoch%100 == 0:
			wl = [i for i in range(650, 1001, 10)]
			plt.plot(wl, x[1].detach().numpy(), label="input")
			plt.plot(wl, de_out[1].detach().numpy(), label="output")
			plt.xlabel("wavelength")
			plt.ylabel("reflectance")
			plt.legend()
			plt.show()

