import matplotlib.pyplot as plt 
import pickle
import pandas as pd 
from collections import defaultdict
import sys

name = 'test_new_fiber'
name = sys.argv[1]
path = 'output/' + name + '/result/result.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

# (101, 5, 36)

wl = [i for i in range(650, 1001, 10)]
for idx, d in enumerate(data):
    for idxx, dd in enumerate(d):
        plt.plot(wl, dd, label='sds %d' % idxx)
    plt.legend()
    plt.xlabel('wavelength[nm]')
    plt.ylabel('reflectance')
    plt.title('ScvO2: %d' % idx)
    plt.savefig('output/' + name + '/plot/Scv_%d.png' % idx)
    plt.clf()


# sensitivity
record = defaultdict(list)
# record["sds"] = [, , , , ]
for i in range(0, 100, 10):
    for s in range(data.shape[1]):
        sens = (data[i, s] - data[i+10, s]/10/data[i, s].mean()).sum()
        record["%d-%d" % (i,i+10)].append(sens)

df = pd.DataFrame(record)
df.to_csv('output/' + name + '_result.csv')


