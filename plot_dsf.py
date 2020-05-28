import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt

data_all = pd.DataFrame(np.zeros((500, 501)))

for dir_name in os.listdir(os.getcwd()):
    if not dir_name.startswith('trial'):
        continue
    print(dir_name)
    for file_name in os.listdir(dir_name):
        if not file_name.endswith('.csv'):
            continue
        print(file_name)
        data = pd.read_csv(os.path.join(dir_name, file_name), index_col=0, header=None)
        assert(data.shape == data_all.shape)
        data_all += data

plt.figure()
plt.pcolor(data_all.transpose(), cmap='Blues')
plt.colorbar()
plt.savefig('plot_DSF.png')
