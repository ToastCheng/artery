from inverse.model import SCVNet
import torch


x1, x2 = torch.rand(10, 31), torch.rand(10, 6)
net = SCVNet()

o = net(x1, x2)
print(o)