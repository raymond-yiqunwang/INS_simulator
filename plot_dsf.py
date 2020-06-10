import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt

data_all = pd.DataFrame(np.zeros((500, 501)))

for dir_name in os.listdir(os.getcwd()):
    if not dir_name.startswith('trial'):
        continue
    print("Processing:", dir_name)
    for file_name in os.listdir(dir_name):
        if not file_name.endswith('.csv'):
            continue
        print(" >>filename:", file_name)
        data = pd.read_csv(os.path.join(dir_name, file_name), index_col=0, header=None)
        assert(data.shape == data_all.shape)
        data_all += data

# plot 
fig, ax1 = plt.subplots()
mappable = ax1.pcolor(data_all.transpose(), cmap='jet')
# main axis
ax1.set_xticks(range(0, 501, 50))
ax1.set_xticklabels(range(0, 11))
ax1.set_xlabel(r'Momemtum transfer ($\AA^{-1}$)')
ax1.set_yticks(range(75, 451, 75))
ax1.set_yticklabels(range(50, 301, 50))
ax1.set_ylabel(r'Vibrational frequency ($cm^{-1}$)')
# secondary axis
ax2 = ax1.twinx()
ax2.set_yticks([0.121, 0.241, 0.363, 0.484, 0.6, 0.725, 0.846, 0.967])
ax2.set_yticklabels(range(5, 41, 5))
ax2.set_ylabel(r'Energy transfer (meV)')
# color bar
cbar = fig.colorbar(mappable, pad=0.13)
cbar.set_ticks([])
cbar.set_label('Scattering intensity')
fig.tight_layout()
plt.savefig('plot_DSF.png', dpi=200)
