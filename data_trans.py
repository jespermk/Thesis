import matplotlib.pyplot as plt
import numpy as np

def ma(x, l):
    """
    Moving average
    """
    return np.convolve(x, np.ones(l)/l, mode='valid')


def rstd(x, l):
    """
    moving std
    """
    return np.stack(
        [np.std(x[0+i:l+i]) for i in range(len(x)-l+1)]
    )
stat = 'correct_rate'
data_loc = ['generated_data/correct_rate_normal_FM_less.npy',
            'generated_data/correct_rate_radial_FM_less.npy',
            'generated_data/correct_rate_ER_FM_less.npy',
            'generated_data/correct_rate_WER_FM_less.npy',
            'generated_data/correct_rate_t_radial_FM_less.npy',
            'generated_data/correct_rate_exp_radial_FM_less.npy']

models = ['Normal', 'Radial', 'ER', 'WER', 'Radial w. t', 'Radial w. exp']


data = [np.load(data) for data in data_loc]
"""
epoc_n = 150
w = 200

nll_per_epoc = np.stack([np.mean(nll[:, w*i: w*(i+1)], -1) for i in range(150)])


nll_per_epoc_mean = np.stack([1 - ma(nll_per_epoc[:,i], 5) for i in range(len(data_loc))])
nll_per_epoc_std =  np.stack([rstd(nll_per_epoc[:,i], 5) for i in range(len(data_loc))])
"""

per_epoc_mean = [ma(data, 2) for data in data]
per_epoc_std =  [rstd(data, 2) for data in data]

fig, ax = plt.subplots()

[ax.plot(per_epoc_mean[i], label=models[i]) for i in range(len(data_loc))]

[ax.fill_between(np.arange(np.size(per_epoc_mean[i])),
                                (per_epoc_mean[i] - per_epoc_std[i]),
                                (per_epoc_mean[i] + per_epoc_std[i]),
                                color='b',
                                alpha=.1) for i in range(len(data_loc))]

ax.set_yscale('log')

ax.legend()
fig.savefig('plots/ece_MNIST', dpi=400)

plt.show()