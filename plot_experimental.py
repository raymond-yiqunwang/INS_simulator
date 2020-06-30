import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_full_spectrum(data, columns):
    # filter out strong elastic scattering
    data.loc[data['I'] > 0.002, 'I'] = 0.002
    # remove unobserved data
    data = data[data['I'] != -10000]
    # add one line so that (Qmax, Emax) == (9AA^{-1}, 50meV)
    Qmax, Emax = data['Q'].max(), 50.0
    data_view = pd.concat([data, 
                           pd.DataFrame([[np.nan, 0, Qmax, Emax]], 
                                          columns=data.columns.values)], 
                           axis=0, ignore_index=True)
    # pivot
    data_pivot = data_view.pivot(index='E', columns='Q', values='I')

    # make figure
    fig, ax = plt.subplots()
    mappable = ax.pcolormesh(data_pivot)
    # setup x-axis
    mdim = data_pivot.shape[1]
    ax.set_xlabel(r'Momemtum transfer ($\AA^{-1}$)')
    ax.set_xticks(np.linspace(0, mdim, 10))
    ax.set_xticklabels(np.arange(0, 11, 1))
    # setup y-axis
    ndim = data_pivot.shape[0]
    ax.set_ylabel(r'Energy transfer (meV)')
    ax.set_yticks(np.linspace(0, ndim, 6))
    ax.set_yticklabels(np.arange(0, 51, 10))
    ax.tick_params(axis=u'both', which=u'both',length=0)
    # secondary y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel(r'Vibrational frequency (THz)')
    ax2.set_yticks(np.linspace(0, ndim, 7))
    ax2.set_yticklabels(np.arange(0, 13, 2))
    ax2.tick_params(length=0)
    # color bar
    cbar = fig.colorbar(mappable, pad=0.13)
    cbar.set_ticks([])
    cbar.set_label('Scattering intensity')
    # save fig
    fig.tight_layout()
    plt.savefig('DSF_spectrum.png', dpi=200)
#    plt.show()


def plot_Qcut(data, columns, Q_start, Q_end):
    # filter out strong elastic scattering
    data.loc[data['I'] > 0.002, 'I'] = 0.002
    # remove unobserved data
    data = data[data['I'] != -10000]
    data_view = data[data['Q'].between(Q_start, Q_end)]
    data_view = data_view[data_view['E'] <= 35.]
    data_group = data_view.groupby('E')
    x_values = list(data_group.groups.keys())
    y_values = [ig['I'].mean(skipna=True) for _,ig in data_group]
    # chop at low E
    chop = 7
    x_values, y_values = x_values[chop:], y_values[chop:]

    # make figure
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values, s=10, marker='o', c='none', edgecolor='C1')
    ax.set_xlim(0, 35)
    edge = 0.1 * (max(y_values)-min(y_values))
    ax.set_ylim(min(y_values)-edge, max(y_values)+edge)
    # save fig
    fig.tight_layout()
    plt.savefig('DSF_Qcut_'+str(Q_start)+'-'+str(Q_end)+'.png', dpi=200)
#    plt.show()


if __name__ == "__main__":
    # read experimental data
    data_source = './data/GaTaSe_5K_50meV_slice.csv'
    columns = ['I', 'error', 'Q', 'E']
    data = pd.read_csv(data_source, names=columns)

    # plot full spectrum
    plot_full_spectrum(data, columns)

    # plot Q cut
    Q_pairs = [(3, 5), (5, 7), (7, 9)]
    for Q_start, Q_end in Q_pairs:
        plot_Qcut(data, columns, Q_start, Q_end)


